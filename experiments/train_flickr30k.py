"""
Main training script for Flickr30K experiments.

Usage:
    python experiments/train_flickr30k.py --config configs/flickr30k.yaml
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gno_model import GraphNeuralOperator
from src.data.flickr30k_dataset import Flickr30KGraphDataset, flickr30k_collate
from src.losses import gno_total_loss
from src.utils import compute_alignment_metrics, aggregate_metrics


def train_one_epoch(model, loader, optimizer, cfg, device):
    """Train for one epoch."""
    model.train()
    all_losses = []
    all_metrics = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        scene_batch = batch["scene_batch"].to(device)
        text_batch = batch["text_batch"].to(device)
        alignments = batch["alignments"]

        optimizer.zero_grad()

        # Forward
        kernels, v_hats, tgt_feats = model(scene_batch, text_batch)

        # Compute loss over batch
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_valid = 0

        for K, v_hat, v_true, Y in zip(kernels, v_hats, tgt_feats, alignments):
            Y = Y.to(device)
            # Ensure shapes match
            if K.shape != Y.shape:
                min_r = min(K.shape[0], Y.shape[0])
                min_c = min(K.shape[1], Y.shape[1])
                K_sub = K[:min_r, :min_c]
                Y_sub = Y[:min_r, :min_c]
            else:
                K_sub, Y_sub = K, Y

            loss_dict = gno_total_loss(
                K_sub, Y_sub, v_hat, v_true,
                pos_weight=cfg["train"]["pos_weight"],
                gamma=cfg["train"]["lambda_recon"],
                alpha=cfg["train"]["alpha_reg"],
                beta=cfg["train"]["beta_entropy"],
            )
            total_loss = total_loss + loss_dict["total"]
            n_valid += 1

            # Metrics
            m = compute_alignment_metrics(K_sub, Y_sub, cfg["eval"]["threshold"])
            all_metrics.append(m)

        if n_valid > 0:
            total_loss = total_loss / n_valid
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
            all_losses.append(total_loss.item())

        pbar.set_postfix(loss=f"{np.mean(all_losses[-10:]):.4f}")

    avg_loss = np.mean(all_losses) if all_losses else float("inf")
    avg_metrics = aggregate_metrics(all_metrics)
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Evaluate on validation/test set."""
    model.eval()
    all_losses = []
    all_metrics = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        scene_batch = batch["scene_batch"].to(device)
        text_batch = batch["text_batch"].to(device)
        alignments = batch["alignments"]

        kernels, v_hats, tgt_feats = model(scene_batch, text_batch)

        for K, v_hat, v_true, Y in zip(kernels, v_hats, tgt_feats, alignments):
            Y = Y.to(device)
            if K.shape != Y.shape:
                min_r = min(K.shape[0], Y.shape[0])
                min_c = min(K.shape[1], Y.shape[1])
                K, Y = K[:min_r, :min_c], Y[:min_r, :min_c]

            loss_dict = gno_total_loss(
                K, Y, v_hat, v_true,
                pos_weight=cfg["train"]["pos_weight"],
                gamma=cfg["train"]["lambda_recon"],
                alpha=cfg["train"]["alpha_reg"],
                beta=cfg["train"]["beta_entropy"],
            )
            all_losses.append(loss_dict["total"].item())
            all_metrics.append(
                compute_alignment_metrics(K, Y, cfg["eval"]["threshold"]))

    avg_loss = np.mean(all_losses) if all_losses else float("inf")
    avg_metrics = aggregate_metrics(all_metrics)
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GNO on Flickr30K")
    parser.add_argument("--config", type=str,
                        default="configs/flickr30k.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Output directory
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    print("\n=== Loading Flickr30K data ===")
    train_ds = Flickr30KGraphDataset(
        cfg["data"]["flickr30k_root"], split="train",
        max_samples=cfg["split"]["train_samples"],
        seed=cfg["split"]["seed"],
        feature_dim=cfg["model"]["input_dim"],
        exact_weight=cfg["train"]["exact_match_weight"],
        synonym_weight=cfg["train"]["synonym_match_weight"],
        category_weight=cfg["train"]["category_match_weight"],
    )
    val_ds = Flickr30KGraphDataset(
        cfg["data"]["flickr30k_root"], split="val",
        max_samples=cfg["split"]["val_samples"],
        seed=cfg["split"]["seed"],
        feature_dim=cfg["model"]["input_dim"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, collate_fn=flickr30k_collate, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, collate_fn=flickr30k_collate, num_workers=0)

    # ---- Model ----
    print("\n=== Building model ===")
    model = GraphNeuralOperator(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ---- Optimizer ----
    opt_cls = torch.optim.AdamW if cfg["train"]["optimizer"] == "adamw" \
        else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg["train"]["lr"],
                        weight_decay=cfg["train"]["weight_decay"])

    scheduler = None
    if cfg["train"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"])

    # ---- Training loop ----
    print(f"\n=== Training for {cfg['train']['epochs']} epochs ===\n")

    history = {"train_loss": [], "val_loss": [],
               "train_f1": [], "val_f1": [],
               "train_precision": [], "val_precision": [],
               "train_recall": [], "val_recall": []}
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, cfg, device)
        val_loss, val_m = evaluate(model, val_loader, cfg, device)

        if scheduler:
            scheduler.step()

        dt = time.time() - t0

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in ("f1", "precision", "recall"):
            history[f"train_{k}"].append(train_m.get(k, 0))
            history[f"val_{k}"].append(val_m.get(k, 0))

        print(f"Epoch {epoch:3d}/{cfg['train']['epochs']}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_F1={train_m.get('f1', 0):.4f}  "
              f"val_F1={val_m.get('f1', 0):.4f}  "
              f"val_P={val_m.get('precision', 0):.4f}  "
              f"val_R={val_m.get('recall', 0):.4f}  "
              f"({dt:.1f}s)")

        # Save best
        if val_m.get("f1", 0) > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "config": cfg,
            }, save_dir / "best_model.pth")
            print(f"  ★ New best val_F1 = {best_val_f1:.4f}")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % cfg["logging"]["save_every"] == 0:
            torch.save(model.state_dict(),
                       save_dir / f"checkpoint_epoch{epoch}.pth")

        # Early stopping
        if patience_counter >= cfg["train"]["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ---- Save results ----
    results = {
        "best_val_f1": best_val_f1,
        "config": cfg,
        "history": history,
        "n_params": n_params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Training complete.  Best val F1 = {best_val_f1:.4f}")
    print(f"Results saved to {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
