#!/usr/bin/env python3
"""
Train MobileNetV3-small for CNN gate (binary classification).
Dataset: ImageFolder with classes "interesting" and "boring".
"""
import argparse
import os

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
    tp = tn = fp = fn = 0
    for x, y in dl:
        x = x.to(device)
        y = (y == pos_index).float().to(device)
        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * x.size(0)
        n += x.size(0)

        p = torch.sigmoid(logits)
        pred = (p >= 0.5).long()
        yy = y.long()
        tp += ((pred == 1) & (yy == 1)).sum().item()
        tn += ((pred == 0) & (yy == 0)).sum().item()
        fp += ((pred == 1) & (yy == 0)).sum().item()
        fn += ((pred == 0) & (yy == 1)).sum().item()

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return tot_loss / max(1, n), acc, prec, rec, f1


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
    args = ap.parse_args()

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
    print("class_to_idx:", tr.class_to_idx)

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

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(tr_dl))
    )

    best_f1 = -1.0
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

        vloss, vacc, vprec, vrec, vf1 = eval_epoch(model, va_dl, device, loss_fn, pos_index)
        print(
            f"ep={ep} train_loss={tot / max(1, n):.4f} "
            f"val_loss={vloss:.4f} acc={vacc:.3f} p={vprec:.3f} r={vrec:.3f} f1={vf1:.3f}"
        )
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), args.out)
            print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
