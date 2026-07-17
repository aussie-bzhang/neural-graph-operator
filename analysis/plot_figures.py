"""
Generate all paper figures from experiment results.

Usage:
    python analysis/plot_figures.py --results-dir outputs/flickr30k
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def plot_training_curves(history, save_path):
    """Figure 2 equivalent: training dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Loss curves
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", color="#2196F3")
    ax.plot(history["val_loss"], label="Validation", color="#F44336")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("(a) Training Dynamics"); ax.legend(); ax.grid(alpha=0.3)

    # (b) F1 curves
    ax = axes[1]
    ax.plot(history["train_f1"], label="Train F1", color="#2196F3")
    ax.plot(history["val_f1"], label="Val F1", color="#F44336")
    best_ep = np.argmax(history["val_f1"])
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.5)
    ax.scatter([best_ep], [history["val_f1"][best_ep]], marker="*",
               s=100, color="#F44336", zorder=5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
    ax.set_title("(b) Alignment F1"); ax.legend(); ax.grid(alpha=0.3)

    # (c) Precision / Recall
    ax = axes[2]
    ax.plot(history["val_precision"], label="Precision", color="#4CAF50")
    ax.plot(history["val_recall"], label="Recall", color="#FF9800")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("(c) Precision & Recall"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ablation_table(ablation_data, save_path):
    """Ablation bar chart."""
    # Filter key ablations
    key_names = {
        "full_model": "Full Model",
        "no_weighted_loss": "w/o Weighted Loss",
        "no_multilevel": "w/o Multi-Level",
        "layers_1": "1-layer GCN",
        "kernel_mlp": "MLP Kernel",
        "kernel_attention": "Attn Kernel",
        "sinkhorn_5": "+ Sinkhorn",
        "topk_3": "+ Top-3 Sparsity",
    }
    names, f1s = [], []
    for k, label in key_names.items():
        if k in ablation_data:
            names.append(label)
            f1s.append(ablation_data[k]["f1"])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3"] + ["#FF9800"] * (len(names) - 1)
    colors[0] = "#4CAF50"  # Full model in green
    bars = ax.barh(names[::-1], f1s[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("F1 Score")
    ax.set_title("Ablation Study Results")
    for bar, val in zip(bars, f1s[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlim(0, max(f1s) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_scaling(scaling_data, save_path):
    """Scaling study: time vs node count."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if "standard_bilinear" in scaling_data:
        data = scaling_data["standard_bilinear"]
        # Group by n_tgt
        by_tgt = {}
        for d in data:
            nt = d["n_tgt"]
            if nt not in by_tgt:
                by_tgt[nt] = {"ns": [], "time": []}
            if d["time_ms"] > 0:
                by_tgt[nt]["ns"].append(d["n_src"])
                by_tgt[nt]["time"].append(d["time_ms"])

        ax = axes[0]
        for nt, vals in sorted(by_tgt.items()):
            ax.plot(vals["ns"], vals["time"], "o-", label=f"|V2|={nt}")
        ax.set_xlabel("|V₁| (source nodes)")
        ax.set_ylabel("Time (ms)")
        ax.set_title("(a) Inference Time vs Graph Size")
        ax.legend(); ax.grid(alpha=0.3)

        # Memory
        ax = axes[1]
        by_tgt_mem = {}
        for d in data:
            nt = d["n_tgt"]
            if nt not in by_tgt_mem:
                by_tgt_mem[nt] = {"ns": [], "mem": []}
            if d.get("memory_mb", 0) > 0:
                by_tgt_mem[nt]["ns"].append(d["n_src"])
                by_tgt_mem[nt]["mem"].append(d["memory_mb"])
        for nt, vals in sorted(by_tgt_mem.items()):
            if vals["mem"]:
                ax.plot(vals["ns"], vals["mem"], "s-", label=f"|V2|={nt}")
        ax.set_xlabel("|V₁| (source nodes)")
        ax.set_ylabel("GPU Memory (MB)")
        ax.set_title("(b) Memory vs Graph Size")
        ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_robustness(robustness_data, save_path):
    """Noise robustness curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Noise types
    if "noise_robustness" in robustness_data:
        nr = robustness_data["noise_robustness"]
        noise_types = ["node_deletion", "node_insertion",
                       "edge_deletion", "feature_noise"]
        levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

        ax = axes[0]
        for nt, c in zip(noise_types, colors):
            f1s = []
            for nl in levels:
                key = f"{nt}_{nl:.1f}"
                f1s.append(nr.get(key, {}).get("f1", 0))
            ax.plot(levels, f1s, "o-", color=c,
                    label=nt.replace("_", " ").title())
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("F1 Score")
        ax.set_title("(a) Noise Robustness")
        ax.legend(); ax.grid(alpha=0.3)

    # Ratio sweep
    if "ratio_sweep" in robustness_data:
        rs = robustness_data["ratio_sweep"]
        ax = axes[1]
        names, f1s = [], []
        for k, v in rs.items():
            names.append(k.replace("ratio_x", "x").replace("drop_", "-"))
            f1s.append(v.get("f1", 0))
        ax.bar(range(len(names)), f1s, color="#2196F3", edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1 Score")
        ax.set_title("(b) Node-Count Ratio Stress Test")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/flickr30k")
    args = parser.parse_args()
    rdir = Path(args.results_dir)
    fig_dir = rdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    rpath = rdir / "results.json"
    if rpath.exists():
        with open(rpath, encoding="utf-8") as f:
            data = json.load(f)
        if "history" in data:
            plot_training_curves(data["history"], fig_dir / "training_curves.pdf")

    # Ablation
    apath = rdir / "ablation_results.json"
    if apath.exists():
        with open(apath, encoding="utf-8") as f:
            plot_ablation_table(json.load(f), fig_dir / "ablation.pdf")

    # Scaling
    spath = rdir / "scaling_study.json"
    if spath.exists():
        with open(spath, encoding="utf-8") as f:
            plot_scaling(json.load(f), fig_dir / "scaling.pdf")

    # Robustness
    rbpath = rdir / "robustness_study.json"
    if rbpath.exists():
        with open(rbpath, encoding="utf-8") as f:
            plot_robustness(json.load(f), fig_dir / "robustness.pdf")

    print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
