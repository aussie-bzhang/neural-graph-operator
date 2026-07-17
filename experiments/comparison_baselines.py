"""
Baseline comparison experiment.

Key design: each method selects its own optimal threshold on a
validation split, then reports metrics on the held-out test split.
This prevents the "all-positive" problem and gives a fair comparison.

Methods:
  1. Random
  2. Cosine similarity (no learning)
  3. Hungarian matching (optimal 1-to-1)
  4. GNO-Bilinear  (ours — uses pre-trained checkpoint)
  5. GNO-MLP       (ours — trained from scratch)
  6. GNO-Attention  (ours — trained from scratch)
  7. GNO-LowRank   (ours — trained from scratch)

Usage:
    python experiments/comparison_baselines.py --config configs/flickr30k.yaml \
        --checkpoint outputs/flickr30k/best_model.pth --epochs 50
"""

import sys, json, argparse, copy
from pathlib import Path

import yaml, torch, numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.gno_model import GraphNeuralOperator
from src.data.flickr30k_dataset import Flickr30KGraphDataset, flickr30k_collate
from src.losses import gno_total_loss
from src.utils import compute_alignment_metrics, aggregate_metrics, hungarian_matching


# ============================================================
#  Optimal threshold selection
# ============================================================

def find_optimal_threshold(scores_list, labels_list):
    """Find threshold that maximizes F1 on validation set."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        f1s = []
        for K_np, Y_np in zip(scores_list, labels_list):
            pred = (K_np >= t).astype(int)
            true = (Y_np > 0).astype(int)
            tp = ((pred == 1) & (true == 1)).sum()
            fp = ((pred == 1) & (true == 0)).sum()
            fn = ((pred == 0) & (true == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1s.append(f1)
        avg_f1 = np.mean(f1s)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_t = t
    return best_t


def eval_with_threshold(scores_list, labels_list, threshold):
    """Evaluate with a specific threshold."""
    metrics = []
    for K_np, Y_np in zip(scores_list, labels_list):
        K_t = torch.tensor(K_np)
        Y_t = torch.tensor(Y_np)
        metrics.append(compute_alignment_metrics(K_t, Y_t, threshold))
    return aggregate_metrics(metrics)


# ============================================================
#  Score generation for each method
# ============================================================

def get_scores_random(dataset):
    """Random alignment scores."""
    scores, labels = [], []
    for i in range(len(dataset)):
        Y = dataset[i]["alignment"].numpy()
        K = np.random.rand(*Y.shape).astype(np.float32)
        scores.append(K); labels.append(Y)
    return scores, labels


def get_scores_cosine(dataset):
    """Cosine similarity scores."""
    scores, labels = [], []
    for i in range(len(dataset)):
        s = dataset[i]
        g1, g2, Y = s["graph1"], s["graph2"], s["alignment"]
        s1 = torch.nn.functional.normalize(g1.x, dim=-1)
        s2 = torch.nn.functional.normalize(g2.x, dim=-1)
        sim = (s1 @ s2.T).numpy()
        # Rescale from [-1,1] to [0,1]
        K = (sim + 1.0) / 2.0
        r, c = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
        scores.append(K[:r,:c]); labels.append(Y.numpy()[:r,:c])
    return scores, labels


def get_scores_hungarian(dataset):
    """Hungarian matching (returns binary)."""
    scores, labels = [], []
    for i in range(len(dataset)):
        s = dataset[i]
        g1, g2, Y = s["graph1"], s["graph2"], s["alignment"]
        s1 = torch.nn.functional.normalize(g1.x, dim=-1)
        s2 = torch.nn.functional.normalize(g2.x, dim=-1)
        sim = torch.sigmoid(s1 @ s2.T)
        K = hungarian_matching(sim).numpy()
        r, c = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
        scores.append(K[:r,:c]); labels.append(Y.numpy()[:r,:c])
    return scores, labels


def get_scores_gno(dataset, model, device):
    """Get GNO alignment scores."""
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            s = dataset[i]
            g1, g2, Y = s["graph1"], s["graph2"], s["alignment"]
            K, _, _, _ = model.forward_single(
                g1.x.to(device), g1.edge_index.to(device),
                g2.x.to(device), g2.edge_index.to(device))
            K = K.cpu().numpy()
            Y = Y.numpy()
            r, c = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
            scores.append(K[:r,:c]); labels.append(Y[:r,:c])
    return scores, labels


def train_gno_variant(cfg, device, kernel_type, epochs=50):
    """Train a GNO variant and return the trained model."""
    mcfg = copy.deepcopy(cfg)
    mcfg["model"]["kernel_type"] = kernel_type

    train_ds = Flickr30KGraphDataset(
        mcfg["data"]["flickr30k_root"], "train",
        max_samples=mcfg["split"]["train_samples"],
        seed=mcfg["split"]["seed"],
        feature_dim=mcfg["model"]["input_dim"])
    val_ds = Flickr30KGraphDataset(
        mcfg["data"]["flickr30k_root"], "val",
        max_samples=mcfg["split"]["val_samples"],
        seed=mcfg["split"]["seed"],
        feature_dim=mcfg["model"]["input_dim"])

    tl = DataLoader(train_ds, mcfg["train"]["batch_size"], shuffle=True,
                    collate_fn=flickr30k_collate, num_workers=0)

    model = GraphNeuralOperator(mcfg["model"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["train"]["lr"],
                           weight_decay=mcfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_sd, best_f1 = None, 0
    patience, patience_limit = 0, 10

    for ep in range(epochs):
        model.train()
        for batch in tl:
            sb = batch["scene_batch"].to(device)
            tb = batch["text_batch"].to(device)
            opt.zero_grad()
            Ks, vhs, tfs = model(sb, tb)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            nv = 0
            for K, vh, vt, Y in zip(Ks, vhs, tfs, batch["alignments"]):
                Y = Y.to(device)
                r, c = min(K.shape[0], Y.shape[0]), min(K.shape[1], Y.shape[1])
                ld = gno_total_loss(K[:r,:c], Y[:r,:c], vh, vt,
                                    pos_weight=mcfg["train"]["pos_weight"],
                                    gamma=mcfg["train"]["lambda_recon"],
                                    alpha=mcfg["train"]["alpha_reg"],
                                    beta=mcfg["train"]["beta_entropy"])
                loss = loss + ld["total"]; nv += 1
            if nv:
                (loss / nv).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        sched.step()

        # Val check
        val_scores, val_labels = get_scores_gno(val_ds, model, device)
        t = find_optimal_threshold(val_scores, val_labels)
        vm = eval_with_threshold(val_scores, val_labels, t)
        vf1 = vm.get("f1", 0)
        if vf1 > best_f1:
            best_f1 = vf1; best_sd = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break

    if best_sd:
        model.load_state_dict(best_sd)
    print(f"  Trained {kernel_type}: best val F1={best_f1:.4f} (ep={ep+1-patience})")
    return model


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flickr30k.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Pre-trained bilinear checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load val + test
    val_ds = Flickr30KGraphDataset(
        cfg["data"]["flickr30k_root"], "val",
        max_samples=cfg["split"]["val_samples"],
        seed=cfg["split"]["seed"], feature_dim=cfg["model"]["input_dim"])
    test_ds = Flickr30KGraphDataset(
        cfg["data"]["flickr30k_root"], "test",
        max_samples=cfg["split"].get("test_samples", 200),
        seed=cfg["split"]["seed"], feature_dim=cfg["model"]["input_dim"])

    results = {}

    # ---- 1. Random ----
    print("\n[1/7] Random...")
    sc, lb = get_scores_random(test_ds)
    t = find_optimal_threshold(*get_scores_random(val_ds))
    results["Random"] = eval_with_threshold(sc, lb, t)
    results["Random"]["threshold"] = t
    print(f"  F1={results['Random']['f1']:.4f} (t={t:.2f})")

    # ---- 2. Cosine ----
    print("[2/7] Cosine...")
    val_sc, val_lb = get_scores_cosine(val_ds)
    t = find_optimal_threshold(val_sc, val_lb)
    sc, lb = get_scores_cosine(test_ds)
    results["Cosine"] = eval_with_threshold(sc, lb, t)
    results["Cosine"]["threshold"] = t
    print(f"  F1={results['Cosine']['f1']:.4f} (t={t:.2f})")

    # ---- 3. Hungarian ----
    print("[3/7] Hungarian...")
    sc, lb = get_scores_hungarian(test_ds)
    results["Hungarian"] = eval_with_threshold(sc, lb, 0.5)
    results["Hungarian"]["threshold"] = 0.5
    print(f"  F1={results['Hungarian']['f1']:.4f}")

    # ---- 4. GNO-Bilinear (from checkpoint) ----
    print("\n[4/7] GNO-Bilinear...")
    if args.checkpoint and Path(args.checkpoint).exists():
        model_bi = GraphNeuralOperator(cfg["model"]).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model_bi.load_state_dict(sd)
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        model_bi = train_gno_variant(cfg, device, "bilinear", args.epochs)

    val_sc, val_lb = get_scores_gno(val_ds, model_bi, device)
    t = find_optimal_threshold(val_sc, val_lb)
    sc, lb = get_scores_gno(test_ds, model_bi, device)
    results["GNO-Bilinear"] = eval_with_threshold(sc, lb, t)
    results["GNO-Bilinear"]["threshold"] = t
    print(f"  F1={results['GNO-Bilinear']['f1']:.4f} (t={t:.2f})")

    # ---- 5-7. Other GNO variants (train from scratch) ----
    for idx, kt in enumerate(["mlp", "attention", "lowrank"], 5):
        name = f"GNO-{kt.capitalize()}"
        print(f"\n[{idx}/7] {name}...")
        model_v = train_gno_variant(cfg, device, kt, args.epochs)
        val_sc, val_lb = get_scores_gno(val_ds, model_v, device)
        t = find_optimal_threshold(val_sc, val_lb)
        sc, lb = get_scores_gno(test_ds, model_v, device)
        results[name] = eval_with_threshold(sc, lb, t)
        results[name]["threshold"] = t
        print(f"  F1={results[name]['f1']:.4f} (t={t:.2f})")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print(f"{'Method':<20s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} "
          f"{'AUC':>8s} {'Thresh':>8s}")
    print("-" * 80)
    for name, m in results.items():
        print(f"{name:<20s} {m.get('precision',0):>8.4f} "
              f"{m.get('recall',0):>8.4f} {m.get('f1',0):>8.4f} "
              f"{m.get('auc',0):>8.4f} {m.get('threshold',0.5):>8.2f}")
    print("=" * 80)

    out = Path(cfg["logging"]["save_dir"]) / "baseline_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
