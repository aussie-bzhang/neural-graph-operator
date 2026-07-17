"""
Main training script for Visual Genome experiments.

Usage:
    python experiments/train_vg.py --config configs/visual_genome.yaml
    python experiments/train_vg.py --config configs/visual_genome.yaml --text-mode natural
"""

import os, sys, json, time, argparse
from pathlib import Path
from datetime import datetime

import yaml, torch, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gno_model import GraphNeuralOperator
from src.data.vg_dataset import VGGraphDataset, vg_collate
from src.losses import gno_total_loss
from src.utils import compute_alignment_metrics, aggregate_metrics


def train_one_epoch(model, loader, optimizer, cfg, device):
    model.train()
    losses, metrics = [], []
    for batch in tqdm(loader, desc="Train", leave=False):
        sb = batch["scene_batch"].to(device)
        tb = batch["text_batch"].to(device)
        aligns = batch["alignments"]
        optimizer.zero_grad()
        kernels, v_hats, tgt_feats = model(sb, tb)
        total = torch.tensor(0.0, device=device, requires_grad=True)
        nv = 0
        for K, vh, vt, Y in zip(kernels, v_hats, tgt_feats, aligns):
            Y = Y.to(device)
            mr, mc = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
            Ks, Ys = K[:mr, :mc], Y[:mr, :mc]
            ld = gno_total_loss(Ks, Ys, vh, vt,
                                pos_weight=cfg["train"]["pos_weight"],
                                gamma=cfg["train"]["lambda_recon"],
                                alpha=cfg["train"]["alpha_reg"],
                                beta=cfg["train"]["beta_entropy"])
            total = total + ld["total"]; nv += 1
            metrics.append(compute_alignment_metrics(Ks, Ys))
        if nv > 0:
            total = total / nv; total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step(); losses.append(total.item())
    return np.mean(losses) if losses else float("inf"), aggregate_metrics(metrics)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    losses, metrics = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        sb = batch["scene_batch"].to(device)
        tb = batch["text_batch"].to(device)
        kernels, v_hats, tgt_feats = model(sb, tb)
        for K, vh, vt, Y in zip(kernels, v_hats, tgt_feats, batch["alignments"]):
            Y = Y.to(device)
            mr, mc = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
            Ks, Ys = K[:mr, :mc], Y[:mr, :mc]
            ld = gno_total_loss(Ks, Ys, vh, vt,
                                pos_weight=cfg["train"]["pos_weight"],
                                gamma=cfg["train"]["lambda_recon"],
                                alpha=cfg["train"]["alpha_reg"],
                                beta=cfg["train"]["beta_entropy"])
            losses.append(ld["total"].item())
            metrics.append(compute_alignment_metrics(Ks, Ys))
    return np.mean(losses) if losses else float("inf"), aggregate_metrics(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_genome.yaml")
    parser.add_argument("--text-mode", default="template",
                        choices=["template", "natural"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(cfg["logging"]["save_dir"]) / args.text_mode
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== VG Training (text_mode={args.text_mode}) ===")
    train_ds = VGGraphDataset(cfg["data"]["vg_root"], "train",
                              cfg["split"]["train_samples"], cfg["split"]["seed"],
                              cfg["model"]["input_dim"], text_mode=args.text_mode)
    val_ds = VGGraphDataset(cfg["data"]["vg_root"], "val",
                            cfg["split"]["val_samples"], cfg["split"]["seed"],
                            cfg["model"]["input_dim"], text_mode=args.text_mode)

    tl = DataLoader(train_ds, cfg["train"]["batch_size"], shuffle=True,
                    collate_fn=vg_collate, num_workers=0)
    vl = DataLoader(val_ds, cfg["train"]["batch_size"], shuffle=False,
                    collate_fn=vg_collate, num_workers=0)

    model = GraphNeuralOperator(cfg["model"]).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                 weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["train"]["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1 = 0; patience = 0

    for ep in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tl_loss, tm = train_one_epoch(model, tl, optimizer, cfg, device)
        vl_loss, vm = evaluate(model, vl, cfg, device)
        scheduler.step()
        history["train_loss"].append(tl_loss); history["val_loss"].append(vl_loss)
        history["train_f1"].append(tm.get("f1", 0)); history["val_f1"].append(vm.get("f1", 0))
        print(f"Ep {ep:3d}  tL={tl_loss:.4f} vL={vl_loss:.4f} "
              f"tF1={tm.get('f1',0):.4f} vF1={vm.get('f1',0):.4f} ({time.time()-t0:.1f}s)")
        if vm.get("f1", 0) > best_f1:
            best_f1 = vm["f1"]; patience = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  ★ best vF1={best_f1:.4f}")
        else:
            patience += 1
        if patience >= cfg["train"]["early_stopping_patience"]:
            print("Early stop"); break

    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_f1": best_f1, "history": history,
                    "text_mode": args.text_mode, "config": cfg}, f, indent=2, default=str)
    print(f"Done. best_F1={best_f1:.4f}  saved to {save_dir}")


if __name__ == "__main__":
    main()
