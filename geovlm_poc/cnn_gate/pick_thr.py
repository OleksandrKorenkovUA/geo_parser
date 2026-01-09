#!/usr/bin/env python3
"""
Pick a CNN gate threshold based on target recall or best F1.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights


def load_model(ckpt, device):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    norm = weights.transforms().transforms[-1]
    mean, std = norm.mean, norm.std
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return model, tf


@torch.no_grad()
def collect_probs(model, dl, device, pos_index: int):
    ps = []
    ys = []
    for x, y in dl:
        x = x.to(device)
        logits = model(x).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        ps.append(p)
        ys.append((y == pos_index).numpy())
    return np.concatenate(ps), np.concatenate(ys)


def stats(p, y, thr):
    pred = (p >= thr).astype(np.int32)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    pass_rate = (pred == 1).mean()
    return prec, rec, f1, pass_rate, tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data_cnn_gate")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target-recall", type=float, default=0.98)
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and "cuda" in args.device
    device = torch.device(args.device if use_cuda else "cpu")
    model, tf = load_model(args.ckpt, device)
    ds = datasets.ImageFolder(os.path.join(args.data, args.split), transform=tf)
    pos_index = ds.class_to_idx.get("interesting")
    if pos_index is None:
        raise ValueError("expected class folder named 'interesting'")
    print("class_to_idx:", ds.class_to_idx)

    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=use_cuda)
    p, y = collect_probs(model, dl, device, pos_index)

    best = (0.0, -1.0)
    chosen = None
    for thr in np.linspace(0.05, 0.95, 91):
        prec, rec, f1, pass_rate, tp, fp, fn, tn = stats(p, y, thr)
        if rec >= args.target_recall:
            if chosen is None or pass_rate < chosen[0]:
                chosen = (pass_rate, thr, prec, rec, f1, tp, fp, fn, tn)
        if f1 > best[1]:
            best = (thr, f1, prec, rec, pass_rate, tp, fp, fn, tn)

    print("best_f1:", best)
    if chosen:
        print("min_pass_rate_with_target_recall:", chosen)
    else:
        print("no threshold reaches target_recall; lower target or improve model/data")


if __name__ == "__main__":
    main()
