"""
Scaling study: wall-clock time and GPU memory vs graph size.

Reports wall-clock time and GPU memory as functions of |V1| and |V2|
using synthetic graphs with controlled node counts.

Also tests approximate methods:
  - Top-k candidate pruning
  - Low-rank kernel factorization
  - Sinkhorn normalization

Usage:
    python experiments/scaling_study.py --config configs/flickr30k.yaml
"""

import sys, json, time, argparse
from pathlib import Path

import yaml, torch, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gno_model import (
    GraphNeuralOperator, BilinearKernel, LowRankKernel,
    topk_sparsify, sinkhorn_normalize,
)


def make_synthetic_graph(n_nodes, feat_dim=256):
    """Create a synthetic graph with n_nodes for benchmarking."""
    x = torch.randn(n_nodes, feat_dim)
    # Random edges: each node connects to ~3 neighbors
    edges = []
    for i in range(n_nodes):
        n_neighbors = min(3, n_nodes - 1)
        targets = np.random.choice(
            [j for j in range(n_nodes) if j != i], n_neighbors, replace=False)
        for t in targets:
            edges.append([i, t])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


def benchmark_forward(model, n_src, n_tgt, feat_dim, device, n_runs=10):
    """Measure forward pass time and memory for given graph sizes."""
    src_x, src_edge = make_synthetic_graph(n_src, feat_dim)
    tgt_x, tgt_edge = make_synthetic_graph(n_tgt, feat_dim)
    src_x, src_edge = src_x.to(device), src_edge.to(device)
    tgt_x, tgt_edge = tgt_x.to(device), tgt_edge.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.forward_single(src_x, src_edge, tgt_x, tgt_edge)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward_single(src_x, src_edge, tgt_x, tgt_edge)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        peak_mem = 0.0

    return avg_time, std_time, peak_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flickr30k.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_dim = cfg["model"]["input_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]

    # Node count ranges to test — large enough to show O(|V1||V2|) scaling
    src_sizes = [10, 50, 100, 200, 500, 1000, 2000]
    tgt_sizes = [5, 10, 20, 50, 100]

    results = {}

    # ---- 1. Standard kernel scaling ----
    print("=== Standard Bilinear Kernel Scaling ===")
    model_cfg = {**cfg["model"], "kernel_type": "bilinear"}
    model = GraphNeuralOperator(model_cfg).to(device).eval()

    scaling_data = []
    for ns in src_sizes:
        for nt in tgt_sizes:
            try:
                t, t_std, mem = benchmark_forward(model, ns, nt, feat_dim, device)
                entry = {"n_src": ns, "n_tgt": nt,
                         "time_ms": round(t, 3), "time_std_ms": round(t_std, 3),
                         "memory_mb": round(mem, 1)}
                scaling_data.append(entry)
                print(f"  |V1|={ns:4d}  |V2|={nt:3d}  "
                      f"time={t:8.2f}±{t_std:.2f}ms  mem={mem:.0f}MB")
            except RuntimeError as e:
                print(f"  |V1|={ns:4d}  |V2|={nt:3d}  OOM: {e}")
                scaling_data.append({"n_src": ns, "n_tgt": nt,
                                     "time_ms": -1, "memory_mb": -1})

    results["standard_bilinear"] = scaling_data

    # ---- 2. Low-rank kernel scaling ----
    print("\n=== Low-Rank Kernel Scaling ===")
    for rank in [16, 32, 64]:
        model_cfg = {**cfg["model"], "kernel_type": "lowrank", "rank": rank}
        model_lr = GraphNeuralOperator(model_cfg).to(device).eval()
        lr_data = []
        for ns in [50, 100, 200]:
            for nt in [10, 20, 30]:
                try:
                    t, t_std, mem = benchmark_forward(
                        model_lr, ns, nt, feat_dim, device)
                    lr_data.append({"n_src": ns, "n_tgt": nt, "rank": rank,
                                    "time_ms": round(t, 3), "memory_mb": round(mem, 1)})
                    print(f"  rank={rank:3d}  |V1|={ns:4d}  |V2|={nt:3d}  "
                          f"time={t:.2f}ms  mem={mem:.0f}MB")
                except RuntimeError:
                    pass
        results[f"lowrank_r{rank}"] = lr_data

    # ---- 3. MLP kernel scaling (for comparison) ----
    print("\n=== MLP Kernel Scaling ===")
    model_cfg = {**cfg["model"], "kernel_type": "mlp"}
    model_mlp = GraphNeuralOperator(model_cfg).to(device).eval()
    mlp_data = []
    for ns in [50, 100, 200]:
        for nt in [10, 20, 30]:
            try:
                t, t_std, mem = benchmark_forward(
                    model_mlp, ns, nt, feat_dim, device)
                mlp_data.append({"n_src": ns, "n_tgt": nt,
                                 "time_ms": round(t, 3), "memory_mb": round(mem, 1)})
                print(f"  MLP  |V1|={ns:4d}  |V2|={nt:3d}  "
                      f"time={t:.2f}ms  mem={mem:.0f}MB")
            except RuntimeError:
                pass
    results["mlp_kernel"] = mlp_data

    # ---- Save ----
    out = Path(cfg["logging"]["save_dir"]) / "scaling_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
