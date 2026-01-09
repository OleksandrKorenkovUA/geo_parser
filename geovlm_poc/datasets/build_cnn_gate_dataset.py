#!/usr/bin/env python3
"""
Build an ImageFolder dataset for CNN gate training (interesting vs boring).
Uses OSM GeoJSON for positives and heuristic filters for clean negatives.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import Point, box, shape
from shapely.strtree import STRtree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geovlm_poc.tiling import ImageTiler


DEFAULT_RULES = [
    {"key": "building", "values": ["warehouse"]},
    {"key": "landuse", "values": ["industrial"]},
    {"key": "industrial", "values": ["*"]},
    {"key": "man_made", "values": ["works"]},
    {"key": "railway", "values": ["rail", "yard"]},
    {"key": "amenity", "values": ["parking"]},
    {"key": "highway", "values": ["*"]},
]


def _parse_rules(path: str) -> List[Dict[str, List[str]]]:
    if not path:
        return DEFAULT_RULES
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _match_rule(props: Dict[str, str], rule: Dict[str, List[str]]) -> bool:
    key = rule.get("key")
    values = rule.get("values", [])
    val = props.get(key)
    if val is None:
        return False
    if isinstance(val, (list, tuple, set)):
        val = ",".join(str(v) for v in val)
    val = str(val)
    if "*" in values:
        return True
    return val in values


def _load_osm_features(path: str, src_crs: CRS, dst_crs: CRS):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data.get("features", [])
    geoms = []
    props = []
    for feat in feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        if src_crs != dst_crs:
            geom = transform_geom(src_crs, dst_crs, geom, antimeridian_cutting=True)
        g = shape(geom)
        if g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
            if g.is_empty:
                continue
        geoms.append(g)
        props.append(feat.get("properties", {}))
    return geoms, props


def _build_pos_index(
    geoms: List, props: List[Dict[str, str]], rules: List[Dict[str, List[str]]], line_buffer: float
) -> Tuple[STRtree, List]:
    pos = []
    for g, p in zip(geoms, props):
        if not any(_match_rule(p, r) for r in rules):
            continue
        if g.geom_type in ("LineString", "MultiLineString"):
            g = g.buffer(line_buffer)
        pos.append(g)
    if not pos:
        return None, []
    return STRtree(pos), pos


def _heuristic_metrics(rgb: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    score = 0.6 * _norm(edge_density, 0.01, 0.14) + 0.4 * _norm(entropy, 2.0, 6.5)
    return {"edge_density": edge_density, "entropy": entropy, "score": float(score)}


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def _split_from_center(
    cx: float, cy: float, grid_size: float, ratios: Tuple[float, float, float], seed: int
) -> str:
    if grid_size > 0:
        gx = int(cx // grid_size)
        gy = int(cy // grid_size)
        key = f"{gx}_{gy}_{seed}".encode("utf-8")
        h = hashlib.md5(key).hexdigest()
        r = int(h[:8], 16) / 0xFFFFFFFF
    else:
        key = f"{cx}_{cy}_{seed}".encode("utf-8")
        h = hashlib.md5(key).hexdigest()
        r = int(h[:8], 16) / 0xFFFFFFFF
    if r < ratios[0]:
        return "train"
    if r < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _parse_split(split: str) -> Tuple[float, float, float]:
    parts = [float(p.strip()) for p in split.split(",")]
    if len(parts) != 3 or not np.isclose(sum(parts), 1.0):
        raise ValueError("split must be three ratios that sum to 1.0, e.g. 0.8,0.1,0.1")
    return parts[0], parts[1], parts[2]


def _iter_images(images: List[str], list_file: str) -> Iterable[str]:
    if list_file:
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    yield p
    for p in images:
        yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=[], help="GeoTIFF path (repeatable)")
    ap.add_argument("--images-list", help="Text file with GeoTIFF paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--bands", default="1,2,3")
    ap.add_argument("--osm", help="OSM GeoJSON with properties")
    ap.add_argument("--osm-crs", default="EPSG:4326")
    ap.add_argument("--rules-json", help="JSON list of tag rules")
    ap.add_argument("--min-intersection", type=float, default=0.05)
    ap.add_argument("--center-in", action="store_true")
    ap.add_argument("--line-buffer", type=float, default=20.0)
    ap.add_argument("--neg-buffer", type=float, default=200.0)
    ap.add_argument("--boring-max-score", type=float, default=0.24)
    ap.add_argument("--boring-max-edge", type=float, default=0.05)
    ap.add_argument("--boring-max-entropy", type=float, default=3.2)
    ap.add_argument("--keep-ambiguous", action="store_true")
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--grid-size", type=float, default=5000.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--format", default="png")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ratios = _parse_split(args.split)
    bands = tuple(int(b) for b in args.bands.split(","))
    rules = _parse_rules(args.rules_json)

    csv_path = os.path.join(args.out, "labels.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow([
        "split",
        "label",
        "image_path",
        "source_image",
        "tile_id",
        "center_x",
        "center_y",
        "intersect_ratio",
        "edge_density",
        "entropy",
        "score",
    ])

    total = 0
    for img_path in _iter_images(args.image, args.images_list):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        tiler = ImageTiler(tile_size=args.tile_size, overlap=args.overlap, bands=bands)
        with rasterio.open(img_path) as ds:
            dst_crs = ds.crs or CRS.from_string(args.osm_crs)

        pos_tree = None
        pos_geoms = []
        if args.osm:
            src_crs = CRS.from_string(args.osm_crs)
            geoms, props = _load_osm_features(args.osm, src_crs, dst_crs)
            pos_tree, pos_geoms = _build_pos_index(geoms, props, rules, args.line_buffer)

        for tref in tiler.iter_tiles(img_path, image_id):
            if args.limit and total >= args.limit:
                break
            rgb = tiler.read_tile_rgb(img_path, tref)
            metrics = _heuristic_metrics(rgb)

            minx, miny, maxx, maxy = tref.bounds
            tile_poly = box(minx, miny, maxx, maxy)
            tile_area = tile_poly.area
            center = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)

            intersect_ratio = 0.0
            is_pos = False
            if pos_tree:
                candidates = pos_tree.query(tile_poly)
                for g in candidates:
                    inter = tile_poly.intersection(g)
                    if not inter.is_empty:
                        intersect_ratio = max(intersect_ratio, inter.area / max(tile_area, 1e-9))
                        if intersect_ratio >= args.min_intersection:
                            is_pos = True
                            break
                    if args.center_in and g.contains(center):
                        is_pos = True
                        break
            if is_pos:
                label = "interesting"
            else:
                near_pos = False
                if pos_tree and args.neg_buffer > 0:
                    buf = tile_poly.buffer(args.neg_buffer)
                    if pos_tree.query(buf):
                        near_pos = True
                if near_pos and not args.keep_ambiguous:
                    continue
                if (
                    metrics["score"] <= args.boring_max_score
                    and metrics["edge_density"] <= args.boring_max_edge
                    and metrics["entropy"] <= args.boring_max_entropy
                ):
                    label = "boring"
                elif args.keep_ambiguous:
                    label = "boring"
                else:
                    continue

            split = _split_from_center(center.x, center.y, args.grid_size, ratios, seed=17)
            out_dir = os.path.join(args.out, split, label)
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{image_id}_{tref.tile_id}.{args.format}"
            out_path = os.path.join(out_dir, fname)
            Image.fromarray(rgb).save(out_path)

            writer.writerow([
                split,
                label,
                out_path,
                img_path,
                tref.tile_id,
                f"{center.x:.3f}",
                f"{center.y:.3f}",
                f"{intersect_ratio:.4f}",
                f"{metrics['edge_density']:.6f}",
                f"{metrics['entropy']:.4f}",
                f"{metrics['score']:.4f}",
            ])
            total += 1

    csv_f.close()


if __name__ == "__main__":
    main()
