#!/usr/bin/env python3
"""
Minimal OpenCLIP fine-tune on (image_path, text) pairs from a CSV file.
CSV must include columns: image_path,text
"""
import argparse
import csv
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip


class CsvPairs(Dataset):
    def __init__(self, csv_path: str, preprocess):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append((row["image_path"], row["text"]))
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, text = self.rows[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), text


def _collate(batch, tokenizer):
    imgs, txts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    txts = tokenizer(list(txts))
    return imgs, txts


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train.csv")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="out/openclip_finetuned.pt")
    return ap.parse_args()


def main():
    args = _parse_args()
    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda else "cpu")

    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)

    ds = CsvPairs(args.csv, preprocess_train)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        collate_fn=lambda b: _collate(b, tokenizer),
    )

    loss_fn = open_clip.loss.ClipLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    model.train()
    step = 0
    for ep in range(args.epochs):
        for imgs, txts in dl:
            imgs = imgs.to(device, non_blocking=True)
            txts = txts.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                img_f = model.encode_image(imgs)
                txt_f = model.encode_text(txts)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                logits_per_image = model.logit_scale.exp() * img_f @ txt_f.t()
                logits_per_text = logits_per_image.t()
                loss = loss_fn(logits_per_image, logits_per_text)

            loss = loss / args.accum
            scaler.scale(loss).backward()

            if (step + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step % 50 == 0:
                print(f"ep={ep} step={step} loss={loss.item() * args.accum:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
