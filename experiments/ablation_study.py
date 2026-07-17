"""
Ablation study: systematically test each component.

Usage:  python experiments/ablation_study.py --config configs/flickr30k.yaml
"""

import sys, json, copy, argparse
from pathlib import Path
import yaml, torch, numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gno_model import GraphNeuralOperator
from src.data.flickr30k_dataset import Flickr30KGraphDataset, flickr30k_collate
from src.losses import gno_total_loss
from src.utils import compute_alignment_metrics, aggregate_metrics


def quick_train_eval(cfg, device, epochs=20):
    """Train briefly and return best val F1."""
    train_ds = Flickr30KGraphDataset(
        cfg["data"]["flickr30k_root"], "train",
        max_samples=cfg["split"]["train_samples"],
        seed=cfg["split"]["seed"], feature_dim=cfg["model"]["input_dim"],
        exact_weight=cfg["train"]["exact_match_weight"],
        synonym_weight=cfg["train"]["synonym_match_weight"],
        category_weight=cfg["train"]["category_match_weight"])
    val_ds = Flickr30KGraphDataset(
        cfg["data"]["flickr30k_root"], "val",
        max_samples=cfg["split"]["val_samples"],
        seed=cfg["split"]["seed"], feature_dim=cfg["model"]["input_dim"])
    tl = DataLoader(train_ds, cfg["train"]["batch_size"], True,
                    collate_fn=flickr30k_collate, num_workers=0)
    vl = DataLoader(val_ds, cfg["train"]["batch_size"], False,
                    collate_fn=flickr30k_collate, num_workers=0)
    model = GraphNeuralOperator(cfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])
    best_f1, last_m = 0, {}
    for ep in range(epochs):
        model.train()
        for batch in tl:
            sb = batch["scene_batch"].to(device)
            tb = batch["text_batch"].to(device)
            opt.zero_grad()
            Ks, vhs, tfs = model(sb, tb)
            loss = torch.tensor(0., device=device, requires_grad=True)
            nv = 0
            for K, vh, vt, Y in zip(Ks, vhs, tfs, batch["alignments"]):
                Y = Y.to(device)
                r, c = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
                ld = gno_total_loss(K[:r,:c], Y[:r,:c], vh, vt,
                                    pos_weight=cfg["train"]["pos_weight"],
                                    gamma=cfg["train"]["lambda_recon"])
                loss = loss + ld["total"]; nv += 1
            if nv: (loss/nv).backward(); opt.step()
        model.eval(); mets = []
        with torch.no_grad():
            for batch in vl:
                sb = batch["scene_batch"].to(device)
                tb = batch["text_batch"].to(device)
                Ks, vhs, tfs = model(sb, tb)
                for K, vh, vt, Y in zip(Ks, vhs, tfs, batch["alignments"]):
                    Y = Y.to(device)
                    r, c = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
                    mets.append(compute_alignment_metrics(K[:r,:c], Y[:r,:c]))
        am = aggregate_metrics(mets)
        if am.get("f1", 0) > best_f1:
            best_f1 = am["f1"]; last_m = am
    return best_f1, last_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flickr30k.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    ablations = [
        # (name, config_overrides)
        ("full_model", {}),
        # Kernel types
        ("kernel_bilinear", {"model.kernel_type": "bilinear"}),
        ("kernel_mlp",      {"model.kernel_type": "mlp"}),
        ("kernel_attention", {"model.kernel_type": "attention"}),
        ("kernel_lowrank",  {"model.kernel_type": "lowrank"}),
        # Encoder types
        ("encoder_gcn", {"model.encoder_type": "gcn"}),
        ("encoder_gat", {"model.encoder_type": "gat"}),
        # Layer depth
        ("layers_1", {"model.num_encoder_layers": 1}),
        ("layers_2", {"model.num_encoder_layers": 2}),
        ("layers_3", {"model.num_encoder_layers": 3}),
        ("layers_4", {"model.num_encoder_layers": 4}),
        # Positive weight sensitivity
        ("pw_1",  {"train.pos_weight": 1.0}),
        ("pw_3",  {"train.pos_weight": 3.0}),
        ("pw_5",  {"train.pos_weight": 5.0}),
        ("pw_10", {"train.pos_weight": 10.0}),
        ("pw_20", {"train.pos_weight": 20.0}),
        # Without weighted loss
        ("no_weighted_loss", {"train.pos_weight": 1.0}),
        # Without multi-level matching
        ("no_multilevel", {"train.synonym_match_weight": 0.0,
                           "train.category_match_weight": 0.0}),
        # Sinkhorn normalization 
        ("sinkhorn_3",  {"model.use_sinkhorn": True, "model.sinkhorn_iters": 3}),
        ("sinkhorn_5",  {"model.use_sinkhorn": True, "model.sinkhorn_iters": 5}),
        ("sinkhorn_10", {"model.use_sinkhorn": True, "model.sinkhorn_iters": 10}),
        # Top-k sparsity 
        ("topk_2", {"model.use_topk": True, "model.topk_k": 2}),
        ("topk_3", {"model.use_topk": True, "model.topk_k": 3}),
        ("topk_5", {"model.use_topk": True, "model.topk_k": 5}),
    ]

    for name, overrides in ablations:
        cfg = copy.deepcopy(base)
        for k, v in overrides.items():
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = v
        print(f"\n--- {name} ---")
        f1, m = quick_train_eval(cfg, dev, args.epochs)
        results[name] = {"f1": f1, **m}
        print(f"  F1={f1:.4f}  P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}")

    out = Path(base["logging"]["save_dir"]) / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
