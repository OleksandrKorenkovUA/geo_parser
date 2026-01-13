#!/usr/bin/env python3
"""
Pick a CNN gate threshold based on target recall or best F1.
Writes a JSON report and a threshold curve for reproducibility.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights


def _parse_floats(value: str, label: str):
    if not value:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"{label} must have 3 floats, got: {value}")
    return [float(p) for p in parts]


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_model(ckpt, device, mean=None, std=None):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    norm = weights.transforms().transforms[-1]
    default_mean, default_std = norm.mean, norm.std
    if mean is None:
        mean = default_mean
    if std is None:
        std = default_std
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tf, mean, std


@torch.no_grad()
def collect_probs(model, dl, device, pos_indices):
    ps = []
    ys = []
    pos_indices = np.asarray(pos_indices, dtype=np.int64)
    for x, y in dl:
        x = x.to(device)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        ps.append(p)
        ys.append(np.isin(y.numpy(), pos_indices))
    return np.concatenate(ps), np.concatenate(ys)


def compute_curve(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)
    if p.size == 0:
        raise ValueError("empty validation set")

    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    boundaries = np.flatnonzero(p_sorted[1:] != p_sorted[:-1])
    last_indices = np.concatenate([boundaries, [p_sorted.size - 1]])
    thresholds = p_sorted[last_indices]

    tp = tp_cum[last_indices]
    fp = fp_cum[last_indices]
    total_pos = int(tp_cum[-1])
    total_neg = int(p_sorted.size - total_pos)
    fn = total_pos - tp
    tn = total_neg - fp
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / np.maximum(1, tp + fn)
    f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
    pass_rate = (tp + fp) / p_sorted.size
    return thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn, total_pos, total_neg


def pack_metrics(idx, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn):
    return {
        "thr": float(thresholds[idx]),
        "precision": float(prec[idx]),
        "recall": float(rec[idx]),
        "f1": float(f1[idx]),
        "pass_rate": float(pass_rate[idx]),
        "tp": int(tp[idx]),
        "fp": int(fp[idx]),
        "fn": int(fn[idx]),
        "tn": int(tn[idx]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data_cnn_gate")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pos-class", default="interesting")
    ap.add_argument("--pos-classes", default="")
    ap.add_argument("--mean", default="", help="comma- or space-separated mean, e.g. 0.485,0.456,0.406")
    ap.add_argument("--std", default="", help="comma- or space-separated std, e.g. 0.229,0.224,0.225")
    ap.add_argument("--objective", default="target_recall_min_pass",
                    choices=["target_recall_min_pass", "max_f1", "max_recall_then_min_pass"])
    ap.add_argument("--out-report", default="")
    ap.add_argument("--out-curve", default="")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and "cuda" in args.device
    device = torch.device(args.device if use_cuda else "cpu")
    mean = _parse_floats(args.mean, "mean")
    std = _parse_floats(args.std, "std")
    model, tf, mean, std = load_model(args.ckpt, device, mean=mean, std=std)
    ds = datasets.ImageFolder(os.path.join(args.data, args.split), transform=tf)
    if args.pos_classes:
        pos_classes = [p.strip() for p in args.pos_classes.replace(",", " ").split() if p.strip()]
    else:
        pos_classes = [args.pos_class]
    if not pos_classes:
        raise ValueError("no positive classes provided")
    missing = [c for c in pos_classes if c not in ds.class_to_idx]
    if missing:
        raise ValueError(f"missing class folders: {missing}")
    pos_indices = [ds.class_to_idx[c] for c in pos_classes]
    print("class_to_idx:", ds.class_to_idx)
    print("positive_classes:", pos_classes)
    print("mean/std:", mean, std)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=use_cuda)
    p, y = collect_probs(model, dl, device, pos_indices)

    thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn, total_pos, total_neg = compute_curve(p, y)
    best_f1_idx = int(np.argmax(f1))
    best_f1 = pack_metrics(best_f1_idx, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn)

    target_mask = rec >= args.target_recall
    best_target = None
    if np.any(target_mask):
        target_indices = np.flatnonzero(target_mask)
        target_idx = target_indices[np.argmin(pass_rate[target_mask])]
        best_target = pack_metrics(target_idx, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn)

    chosen = None
    if args.objective == "target_recall_min_pass":
        chosen = best_target
    elif args.objective == "max_f1":
        chosen = best_f1
    elif args.objective == "max_recall_then_min_pass":
        max_rec = rec.max()
        recall_mask = np.isclose(rec, max_rec)
        recall_indices = np.flatnonzero(recall_mask)
        recall_idx = recall_indices[np.argmin(pass_rate[recall_mask])]
        chosen = pack_metrics(recall_idx, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn)

    out_report = args.out_report or f"{os.path.splitext(args.ckpt)[0]}.{args.split}.thr_report.json"
    out_curve = args.out_curve or f"{os.path.splitext(args.ckpt)[0]}.{args.split}.thr_curve.jsonl"
    _ensure_parent_dir(out_report)
    _ensure_parent_dir(out_curve)

    with open(out_curve, "w", encoding="utf-8") as f:
        for i in range(thresholds.size):
            row = pack_metrics(i, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn)
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    report = {
        "ckpt": args.ckpt,
        "data": args.data,
        "split": args.split,
        "device": str(device),
        "pos_classes": pos_classes,
        "pos_indices": pos_indices,
        "mean": [float(m) for m in mean],
        "std": [float(s) for s in std],
        "target_recall": float(args.target_recall),
        "objective": args.objective,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_samples": int(p.size),
        "num_pos": int(total_pos),
        "num_neg": int(total_neg),
        "threshold_count": int(thresholds.size),
        "best_f1": best_f1,
        "best_target_recall": best_target,
        "chosen": chosen,
        "target_recall_met": bool(best_target is not None),
        "curve_path": out_curve,
        "curve_order": "desc",
    }
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print("best_f1:", best_f1)
    if best_target:
        print("min_pass_rate_with_target_recall:", best_target)
    else:
        print("no threshold reaches target_recall; lower target or improve model/data")
    print("chosen:", chosen)
    print("report:", out_report)
    print("curve:", out_curve)


if __name__ == "__main__":
    main()
