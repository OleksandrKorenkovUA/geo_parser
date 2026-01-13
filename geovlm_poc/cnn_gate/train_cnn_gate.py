#!/usr/bin/env python3
"""
Train MobileNetV3-small for CNN gate (binary classification).
Dataset: ImageFolder with classes "interesting" and "boring".
"""
import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights


def make_model(pretrained: bool = True):
    if pretrained:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        norm = weights.transforms().transforms[-1]
        mean, std = norm.mean, norm.std
    else:
        model = models.mobilenet_v3_small(weights=None)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model, mean, std


@torch.no_grad()
def eval_epoch(model, dl, device, loss_fn, pos_index: int):
    model.eval()
    tot_loss = 0.0
    n = 0
    ps = []
    ys = []
    for x, y in dl:
        x = x.to(device)
        y = (y == pos_index).float().to(device)
        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)

        p = torch.sigmoid(logits).cpu().numpy()
        ps.append(p)
        ys.append((y == 1).cpu().numpy())

    if not ps:
        raise ValueError("empty validation set")
    return tot_loss / max(1, n), np.concatenate(ps), np.concatenate(ys)


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def class_counts(ds):
    counts = np.bincount(ds.targets, minlength=len(ds.class_to_idx))
    return {name: int(counts[idx]) for name, idx in ds.class_to_idx.items()}


def stats_at_threshold(p, y, thr):
    pred = (p >= thr).astype(np.int32)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    pass_rate = (pred == 1).mean()
    return {
        "thr": float(thr),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pass_rate": float(pass_rate),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


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


def select_threshold(thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn, target_recall, objective):
    best_f1_idx = int(np.argmax(f1))
    best_f1 = pack_metrics(best_f1_idx, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn)

    max_rec = rec.max()
    rec_mask = np.isclose(rec, max_rec)
    rec_indices = np.flatnonzero(rec_mask)
    max_rec_idx = rec_indices[np.argmin(pass_rate[rec_mask])]
    max_recall = pack_metrics(max_rec_idx, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn)

    target_mask = rec >= target_recall
    best_target = None
    if np.any(target_mask):
        target_indices = np.flatnonzero(target_mask)
        target_idx = target_indices[np.argmin(pass_rate[target_mask])]
        best_target = pack_metrics(target_idx, thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn)

    if objective == "target_recall_min_pass":
        chosen = best_target if best_target is not None else max_recall
    elif objective == "best_f1":
        chosen = best_f1
    elif objective == "max_recall":
        chosen = max_recall
    else:
        raise ValueError(f"unknown objective: {objective}")

    return best_f1, best_target, max_recall, chosen, best_target is not None


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data_cnn_gate")
    ap.add_argument("--out", default="cnn_gate_mnv3s.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--select-metric", default="target_recall_min_pass",
                    choices=["target_recall_min_pass", "best_f1", "max_recall"])
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed, args.deterministic)

    use_cuda = torch.cuda.is_available() and "cuda" in args.device
    device = torch.device(args.device if use_cuda else "cpu")
    model, mean, std = make_model(pretrained=not args.no_pretrained)
    model.to(device)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.imgsz, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.imgsz * 1.14)),
        transforms.CenterCrop(args.imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tr = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    va = datasets.ImageFolder(os.path.join(args.data, "val"), transform=val_tf)
    pos_index = tr.class_to_idx.get("interesting")
    if pos_index is None:
        raise ValueError("expected class folder named 'interesting'")
    if va.class_to_idx != tr.class_to_idx:
        raise ValueError("train/val class_to_idx mismatch")
    train_counts = class_counts(tr)
    val_counts = class_counts(va)
    print("class_to_idx:", tr.class_to_idx)
    print("train_counts:", train_counts)
    print("val_counts:", val_counts)

    tr_dl = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    va_dl = DataLoader(
        va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    n_pos = train_counts.get("interesting", 0)
    n_neg = sum(train_counts.values()) - n_pos
    if n_pos <= 0:
        raise ValueError("no positive samples in train split")
    pos_weight = None
    if not args.no_pos_weight:
        pos_weight = float(n_neg) / max(1, n_pos)
        print("pos_weight:", pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(tr_dl))
    )

    best_record = None
    best_target_met = False
    bad_epochs = 0
    meta_path = f"{os.path.splitext(args.out)[0]}.train_meta.json"
    _ensure_parent_dir(meta_path)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        n = 0
        for x, y in tr_dl:
            x = x.to(device)
            y = (y == pos_index).float().to(device)
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)

        vloss, p, y = eval_epoch(model, va_dl, device, loss_fn, pos_index)
        thr_half = stats_at_threshold(p, y, 0.5)
        thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn, total_pos, total_neg = compute_curve(p, y)
        best_f1, best_target, max_recall, chosen, target_met = select_threshold(
            thresholds, prec, rec, f1, pass_rate, tp, fp, fn, tn, args.target_recall, args.select_metric
        )
        print(
            f"ep={ep} train_loss={tot / max(1, n):.4f} "
            f"val_loss={vloss:.4f} "
            f"thr05_p={thr_half['precision']:.3f} thr05_r={thr_half['recall']:.3f} thr05_f1={thr_half['f1']:.3f} "
            f"chosen_thr={chosen['thr']:.3f} chosen_p={chosen['precision']:.3f} "
            f"chosen_r={chosen['recall']:.3f} chosen_f1={chosen['f1']:.3f} "
            f"pass_rate={chosen['pass_rate']:.3f} target_met={target_met}"
        )
        if args.select_metric == "target_recall_min_pass":
            if target_met:
                improved = (not best_target_met) or (best_record is None) or (
                    chosen["pass_rate"] < best_record["chosen"]["pass_rate"]
                )
            else:
                improved = (not best_target_met) and (
                    best_record is None
                    or (chosen["recall"] > best_record["chosen"]["recall"])
                    or (
                        np.isclose(chosen["recall"], best_record["chosen"]["recall"])
                        and chosen["pass_rate"] < best_record["chosen"]["pass_rate"]
                    )
                )
        elif args.select_metric == "best_f1":
            improved = best_record is None or chosen["f1"] > best_record["chosen"]["f1"]
        else:  # max_recall
            improved = best_record is None or (
                (chosen["recall"], -chosen["pass_rate"])
                > (best_record["chosen"]["recall"], -best_record["chosen"]["pass_rate"])
            )

        if improved:
            best_record = {
                "epoch": int(ep),
                "val_loss": float(vloss),
                "thr_0_5": thr_half,
                "best_f1": best_f1,
                "best_target_recall": best_target,
                "max_recall": max_recall,
                "chosen": chosen,
                "target_recall_met": bool(target_met),
            }
            best_target_met = best_target_met or target_met
            torch.save(model.state_dict(), args.out)
            meta = {
                "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "ckpt_path": args.out,
                "meta_path": meta_path,
                "data": args.data,
                "split": "val",
                "pretrained": not args.no_pretrained,
                "imgsz": args.imgsz,
                "class_to_idx": tr.class_to_idx,
                "train_counts": train_counts,
                "val_counts": val_counts,
                "pos_index": int(pos_index),
                "pos_weight": float(pos_weight) if pos_weight is not None else None,
                "mean": [float(m) for m in mean],
                "std": [float(s) for s in std],
                "target_recall": float(args.target_recall),
                "select_metric": args.select_metric,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "best": best_record,
                "num_val": int(p.size),
                "num_val_pos": int(total_pos),
                "num_val_neg": int(total_neg),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=True, indent=2)
            print(f"saved: {args.out}")
            print(f"meta: {meta_path}")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                print(f"early stop: no improvement for {bad_epochs} epochs")
                break


if __name__ == "__main__":
    main()
