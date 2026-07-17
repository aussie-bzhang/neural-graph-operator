"""
Robustness study for GNO.

Part A — Node-count imbalance stress test:
    Sweep |V1|/|V2| ratio by subsampling / adding distractor nodes.

Part B — Input noise robustness:
    Inject realistic noise: node deletion, insertion, edge deletion,
    relation flipping, entity-name perturbation.

Usage:
    python experiments/robustness_study.py --config configs/flickr30k.yaml
"""

import sys, json, copy, random, argparse
from pathlib import Path

import yaml, torch, numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gno_model import GraphNeuralOperator
from src.data.flickr30k_dataset import (
    Flickr30KGraphDataset, flickr30k_collate, text_to_feature,
)
from src.losses import gno_total_loss
from src.utils import compute_alignment_metrics, aggregate_metrics


# ============================================================
#  Graph perturbation utilities
# ============================================================

def drop_nodes(data: Data, drop_rate: float, feat_dim: int = 256) -> Data:
    """Randomly remove a fraction of nodes."""
    n = data.x.size(0)
    n_keep = max(2, int(n * (1 - drop_rate)))
    keep_idx = sorted(random.sample(range(n), n_keep))
    idx_map = {old: new for new, old in enumerate(keep_idx)}
    x = data.x[keep_idx]
    new_edges = []
    for i in range(data.edge_index.size(1)):
        s, t = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        if s in idx_map and t in idx_map:
            new_edges.append([idx_map[s], idx_map[t]])
    ei = torch.tensor(new_edges, dtype=torch.long).t().contiguous() if new_edges \
        else torch.zeros(2, 0, dtype=torch.long)
    return Data(x=x, edge_index=ei)


def insert_distractor_nodes(data: Data, insert_rate: float,
                            feat_dim: int = 256) -> Data:
    """Add random distractor nodes (not connected to existing graph)."""
    n = data.x.size(0)
    n_add = max(1, int(n * insert_rate))
    extra_x = torch.randn(n_add, feat_dim) * 0.1
    x = torch.cat([data.x, extra_x], dim=0)
    # Distractors get random edges among themselves
    extra_edges = []
    for i in range(n_add - 1):
        extra_edges.extend([[n + i, n + i + 1], [n + i + 1, n + i]])
    if extra_edges:
        extra_ei = torch.tensor(extra_edges, dtype=torch.long).t()
        ei = torch.cat([data.edge_index, extra_ei], dim=1)
    else:
        ei = data.edge_index
    return Data(x=x, edge_index=ei)


def drop_edges(data: Data, drop_rate: float) -> Data:
    """Randomly remove a fraction of edges."""
    n_edges = data.edge_index.size(1)
    n_keep = max(0, int(n_edges * (1 - drop_rate)))
    perm = torch.randperm(n_edges)[:n_keep]
    ei = data.edge_index[:, perm]
    return Data(x=data.x.clone(), edge_index=ei)


def perturb_features(data: Data, noise_std: float = 0.3) -> Data:
    """Add Gaussian noise to node features (name perturbation proxy)."""
    x = data.x + torch.randn_like(data.x) * noise_std
    return Data(x=x, edge_index=data.edge_index.clone())


NOISE_FNS = {
    "node_deletion": drop_nodes,
    "node_insertion": insert_distractor_nodes,
    "edge_deletion": drop_edges,
    "feature_noise": perturb_features,
}


# ============================================================
#  Evaluation with perturbations
# ============================================================

def evaluate_with_noise(model, val_ds, noise_type, noise_level, cfg, device):
    """Evaluate model on perturbed validation set."""
    model.eval()
    metrics = []
    feat_dim = cfg["model"]["input_dim"]

    for i in range(len(val_ds)):
        sample = val_ds[i]
        g1, g2, Y = sample["graph1"], sample["graph2"], sample["alignment"]

        # Apply noise to graph1 (scene graph, typically larger)
        if noise_type == "node_deletion":
            g1 = drop_nodes(g1, noise_level, feat_dim)
        elif noise_type == "node_insertion":
            g1 = insert_distractor_nodes(g1, noise_level, feat_dim)
        elif noise_type == "edge_deletion":
            g1 = drop_edges(g1, noise_level)
        elif noise_type == "feature_noise":
            g1 = perturb_features(g1, noise_level)

        # Forward
        with torch.no_grad():
            g1_dev = Data(x=g1.x.to(device), edge_index=g1.edge_index.to(device))
            g2_dev = Data(x=g2.x.to(device), edge_index=g2.edge_index.to(device))
            K, _, _, _ = model.forward_single(
                g1_dev.x, g1_dev.edge_index,
                g2_dev.x, g2_dev.edge_index)
            Y_dev = Y.to(device)

            # Handle shape mismatch after node changes
            r = min(K.shape[0], Y_dev.shape[0])
            c = min(K.shape[1], Y_dev.shape[1])
            if r > 0 and c > 0:
                m = compute_alignment_metrics(K[:r, :c], Y_dev[:r, :c])
                metrics.append(m)

    return aggregate_metrics(metrics)


# ============================================================
#  Part A: Node-count ratio sweep 
# ============================================================

def ratio_sweep(model, val_ds, cfg, device):
    """Sweep |V1|/|V2| ratio by subsampling visual graph."""
    print("\n=== Part A: Node-Count Ratio Stress Test  ===")
    results = {}

    # Increase visual graph by adding distractors
    for extra_frac in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        mets = evaluate_with_noise(
            model, val_ds, "node_insertion", extra_frac, cfg, device)
        avg_ratio = 1.0 + extra_frac  # approximate |V1|/|V2| increase
        results[f"ratio_x{avg_ratio:.1f}"] = mets
        print(f"  +{extra_frac*100:.0f}% distractor nodes  "
              f"(~{avg_ratio:.1f}x ratio)  F1={mets.get('f1',0):.4f}")

    # Decrease visual graph by dropping nodes
    for drop in [0.0, 0.2, 0.4, 0.6, 0.8]:
        mets = evaluate_with_noise(
            model, val_ds, "node_deletion", drop, cfg, device)
        results[f"drop_{drop:.0%}"] = mets
        print(f"  -{drop*100:.0f}% node deletion  F1={mets.get('f1',0):.4f}")

    return results


# ============================================================
#  Part B: Noise robustness 
# ============================================================

def noise_robustness(model, val_ds, cfg, device):
    """Test robustness to various noise types and levels."""
    print("\n=== Part B: Input Noise Robustness  ===")
    results = {}

    noise_types = ["node_deletion", "node_insertion",
                   "edge_deletion", "feature_noise"]
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    for nt in noise_types:
        print(f"\n  Noise: {nt}")
        for nl in noise_levels:
            mets = evaluate_with_noise(model, val_ds, nt, nl, cfg, device)
            key = f"{nt}_{nl:.1f}"
            results[key] = mets
            print(f"    level={nl:.1f}  F1={mets.get('f1',0):.4f}  "
                  f"P={mets.get('precision',0):.4f}  R={mets.get('recall',0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flickr30k.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained model checkpoint")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GraphNeuralOperator(cfg["model"]).to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint loaded — using random weights")

    model.eval()

    # Load validation data
    val_ds = Flickr30KGraphDataset(
        cfg["data"]["flickr30k_root"], "val",
        max_samples=cfg["split"]["val_samples"],
        seed=cfg["split"]["seed"],
        feature_dim=cfg["model"]["input_dim"])

    # Run studies
    ratio_results = ratio_sweep(model, val_ds, cfg, device)
    noise_results = noise_robustness(model, val_ds, cfg, device)

    # Save
    all_results = {"ratio_sweep": ratio_results, "noise_robustness": noise_results}
    out = Path(cfg["logging"]["save_dir"]) / "robustness_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
