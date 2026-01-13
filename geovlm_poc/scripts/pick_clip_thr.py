#!/usr/bin/env python3
"""
Pick a CLIP gate threshold using labeled keep/drop tiles.
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _parse_prompt_list(value: str, fallback):
    if not value:
        return fallback
    sep = "|" if "|" in value else ","
    items = [p.strip() for p in value.split(sep)]
    items = [p for p in items if p]
    return items if items else fallback


def _list_images(root: str):
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"path not found: {root}")
    if p.is_file():
        return [p]
    out = []
    for ext in IMG_EXTS:
        out.extend(p.rglob(f"*{ext}"))
    return sorted(out)


def _sample_paths(paths, max_items: int, seed: int):
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, max_items)


class TileDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), str(path)


def _describe_scores(label: str, scores: np.ndarray):
    if scores.size == 0:
        print(f"{label}: empty")
        return
    qs = np.quantile(scores, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print(
        f"{label}: n={scores.size} min={scores.min():.4f} mean={scores.mean():.4f} "
        f"std={scores.std():.4f} max={scores.max():.4f}"
    )
    print(
        "  quantiles: p01={:.4f} p05={:.4f} p10={:.4f} p25={:.4f} "
        "p50={:.4f} p75={:.4f} p90={:.4f} p95={:.4f} p99={:.4f}".format(*qs)
    )


def _threshold_for_recall(scores: np.ndarray, target_recall: float):
    if scores.size == 0:
        return None
    target_recall = min(max(target_recall, 0.0), 1.0)
    n = scores.size
    k = int(np.floor((1.0 - target_recall) * n))
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])


def _stats(keep_scores: np.ndarray, drop_scores: np.ndarray, thr: float):
    keep_recall = float((keep_scores >= thr).mean()) if keep_scores.size else 0.0
    drop_reject = float((drop_scores < thr).mean()) if drop_scores.size else 0.0
    return keep_recall, drop_reject


@torch.no_grad()
def _collect_scores(model, keep_text, drop_text, dl, device):
    scores = []
    keep_sims = []
    drop_sims = []
    paths = []
    for x, p in dl:
        x = x.to(device)
        vi = model.encode_image(x)
        vi /= vi.norm(dim=-1, keepdim=True)
        k = (vi @ keep_text.T).max(dim=1).values
        d = (vi @ drop_text.T).max(dim=1).values
        s = (k - d).cpu().numpy()
        scores.append(s)
        keep_sims.append(k.cpu().numpy())
        drop_sims.append(d.cpu().numpy())
        paths.extend(p)
    if not scores:
        return np.array([]), np.array([]), np.array([]), []
    return (
        np.concatenate(scores),
        np.concatenate(keep_sims),
        np.concatenate(drop_sims),
        paths,
    )


def _load_model_and_texts(model_name: str, pretrained: str, keep_prompts, drop_prompts, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    with torch.no_grad():
        kt = tokenizer(keep_prompts).to(device)
        dt = tokenizer(drop_prompts).to(device)
        keep_text = model.encode_text(kt)
        keep_text /= keep_text.norm(dim=-1, keepdim=True)
        drop_text = model.encode_text(dt)
        drop_text /= drop_text.norm(dim=-1, keepdim=True)
    return model, preprocess, keep_text, drop_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-dir", required=True, help="Directory with tiles you want to keep")
    ap.add_argument("--drop-dir", required=True, help="Directory with trash tiles to drop")
    ap.add_argument("--max-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=os.environ.get("CLIP_DEVICE", "cpu"))
    ap.add_argument("--model", default=os.environ.get("CLIP_MODEL", "ViT-B-32"))
    ap.add_argument("--pretrained", default=os.environ.get("CLIP_PRETRAINED", "openai") or "openai")
    ap.add_argument("--keep-prompts", default=os.environ.get("CLIP_KEEP_PROMPTS", ""))
    ap.add_argument("--drop-prompts", default=os.environ.get("CLIP_DROP_PROMPTS", ""))
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    keep_default = ["dense urban area", "buildings and roads", "parking lot with many cars", "industrial site"]
    drop_default = ["forest", "agricultural field", "water surface", "clouds"]
    keep_prompts = _parse_prompt_list(args.keep_prompts, keep_default)
    drop_prompts = _parse_prompt_list(args.drop_prompts, drop_default)
    if not keep_prompts or not drop_prompts:
        raise ValueError("keep/drop prompts must be non-empty")

    keep_paths = _list_images(args.keep_dir)
    drop_paths = _list_images(args.drop_dir)
    keep_paths = _sample_paths(keep_paths, args.max_per_class, args.seed)
    drop_paths = _sample_paths(drop_paths, args.max_per_class, args.seed)
    print(f"keep_tiles={len(keep_paths)} drop_tiles={len(drop_paths)}")

    use_cuda = torch.cuda.is_available() and "cuda" in args.device
    device = torch.device(args.device if use_cuda else "cpu")
    model, preprocess, keep_text, drop_text = _load_model_and_texts(
        args.model, args.pretrained, keep_prompts, drop_prompts, device
    )

    keep_ds = TileDataset(keep_paths, preprocess)
    drop_ds = TileDataset(drop_paths, preprocess)
    keep_dl = DataLoader(
        keep_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )
    drop_dl = DataLoader(
        drop_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
    )

    keep_scores, keep_sim, drop_sim, keep_out_paths = _collect_scores(
        model, keep_text, drop_text, keep_dl, device
    )
    drop_scores, drop_keep_sim, drop_drop_sim, drop_out_paths = _collect_scores(
        model, keep_text, drop_text, drop_dl, device
    )

    _describe_scores("keep clip_score", keep_scores)
    _describe_scores("drop clip_score", drop_scores)

    thr = _threshold_for_recall(keep_scores, args.target_recall)
    if thr is not None:
        keep_recall, drop_reject = _stats(keep_scores, drop_scores, thr)
        print(
            f"target_recall={args.target_recall:.2f} -> thr={thr:.4f} "
            f"keep_recall={keep_recall:.3f} drop_reject={drop_reject:.3f}"
        )

    print("recall_targets:")
    for r in [0.995, 0.99, 0.98, 0.95, 0.9]:
        t = _threshold_for_recall(keep_scores, r)
        if t is None:
            continue
        keep_recall, drop_reject = _stats(keep_scores, drop_scores, t)
        print(f"  r={r:.3f} thr={t:.4f} keep_recall={keep_recall:.3f} drop_reject={drop_reject:.3f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("path,label,clip_score,clip_keep_sim,clip_drop_sim\n")
            for p, s, ks, ds in zip(keep_out_paths, keep_scores, keep_sim, drop_sim):
                f.write(f"{p},keep,{s:.6f},{ks:.6f},{ds:.6f}\n")
            for p, s, ks, ds in zip(drop_out_paths, drop_scores, drop_keep_sim, drop_drop_sim):
                f.write(f"{p},drop,{s:.6f},{ks:.6f},{ds:.6f}\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
