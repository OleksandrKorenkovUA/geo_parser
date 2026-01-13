#!/usr/bin/env python3
"""
Build a weakly labeled retrieval dataset from GeoTIFF + OSM GeoJSON.
Outputs tiles.jsonl and optional train.csv for CLIP fine-tune.
"""
import argparse
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
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import Point, box, shape
from shapely.strtree import STRtree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geovlm_poc.tiling import ImageTiler
from geovlm_poc.geo_utils import require_projected_crs, strtree_query_geoms


LABEL_RULES = {
    "warehouse": [{"key": "building", "values": ["warehouse"]}],
    "industrial_area": [{"key": "landuse", "values": ["industrial"]}, {"key": "industrial", "values": ["*"]}],
    "railway": [{"key": "railway", "values": ["rail", "yard"]}],
    "parking_lot": [{"key": "amenity", "values": ["parking"]}],
    "road": [{"key": "highway", "values": ["*"]}],
}


PROMPT_TEMPLATES = {
    "warehouse": [
        "satellite view of a warehouse",
        "overhead view of warehouse buildings",
        "industrial warehouse area from above",
    ],
    "industrial_area": [
        "industrial area from above",
        "industrial site with large buildings",
    ],
    "railway": [
        "railway tracks from above",
        "rail yard seen from above",
    ],
    "parking_lot": [
        "parking lot seen from above",
    ],
    "road": [
        "road network from above",
    ],
}


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


def _build_label_indices(
    geoms: List, props: List[Dict[str, str]], line_buffer: float
) -> Dict[str, Tuple[STRtree, List]]:
    out = {}
    for label, rules in LABEL_RULES.items():
        label_geoms = []
        for g, p in zip(geoms, props):
            if not any(_match_rule(p, r) for r in rules):
                continue
            if g.geom_type in ("LineString", "MultiLineString"):
                g = g.buffer(line_buffer)
            label_geoms.append(g)
        out[label] = (STRtree(label_geoms) if label_geoms else None, label_geoms)
    return out


def _build_building_index(geoms: List, props: List[Dict[str, str]]):
    buildings = []
    for g, p in zip(geoms, props):
        if p.get("building"):
            if g.geom_type in ("Polygon", "MultiPolygon"):
                buildings.append(g)
    return STRtree(buildings) if buildings else None, buildings


def _blue_roof_ratio(rgb: np.ndarray, mask: np.ndarray, h_min: int, h_max: int, s_min: int, v_min: int) -> float:
    if mask.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    blue = (h >= h_min) & (h <= h_max) & (s >= s_min) & (v >= v_min) & (mask > 0)
    return float(blue.sum() / max(1, mask.sum()))


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


def _make_prompts(labels: List[str], blue_roof: bool, max_prompts: int) -> List[str]:
    prompts = []
    for label in labels:
        tmpl = PROMPT_TEMPLATES.get(label, [])
        for t in tmpl:
            if blue_roof and label in ("warehouse", "industrial_area"):
                prompts.append(f"{t} with a blue roof")
            else:
                prompts.append(t)
    return prompts[:max_prompts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=[], help="GeoTIFF path (repeatable)")
    ap.add_argument("--images-list", help="Text file with GeoTIFF paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--bands", default="1,2,3")
    ap.add_argument("--osm", required=True)
    ap.add_argument("--osm-crs", default="EPSG:4326")
    ap.add_argument("--min-intersection", type=float, default=0.05)
    ap.add_argument("--center-in", action="store_true")
    ap.add_argument("--line-buffer", type=float, default=20.0)
    ap.add_argument("--near-buffer", type=float, default=200.0)
    ap.add_argument("--blue-roof", action="store_true")
    ap.add_argument("--blue-h-min", type=int, default=90)
    ap.add_argument("--blue-h-max", type=int, default=130)
    ap.add_argument("--blue-s-min", type=int, default=50)
    ap.add_argument("--blue-v-min", type=int, default=40)
    ap.add_argument("--blue-ratio", type=float, default=0.15)
    ap.add_argument("--emit-csv", action="store_true")
    ap.add_argument("--max-prompts", type=int, default=3)
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--grid-size", type=float, default=5000.0)
    ap.add_argument("--format", default="png")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ratios = _parse_split(args.split)
    bands = tuple(int(b) for b in args.bands.split(","))

    jsonl_path = os.path.join(args.out, "tiles.jsonl")
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")
    csv_f = None
    if args.emit_csv:
        csv_f = open(os.path.join(args.out, "train.csv"), "w", encoding="utf-8", newline="")
        csv_f.write("image_path,text\n")

    for img_path in _iter_images(args.image, args.images_list):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        tiler = ImageTiler(tile_size=args.tile_size, overlap=args.overlap, bands=bands)
        with rasterio.open(img_path) as ds:
            dst_crs = ds.crs
        require_projected_crs(dst_crs, "GeoTIFF")

        src_crs = CRS.from_string(args.osm_crs)
        geoms, props = _load_osm_features(args.osm, src_crs, dst_crs)
        label_indices = _build_label_indices(geoms, props, args.line_buffer)
        bldg_tree, bldg_geoms = _build_building_index(geoms, props)

        for tref in tiler.iter_tiles(img_path, image_id):
            rgb = tiler.read_tile_rgb(img_path, tref)
            minx, miny, maxx, maxy = tref.bounds
            tile_poly = box(minx, miny, maxx, maxy)
            tile_area = tile_poly.area
            center = Point((minx + maxx) / 2.0, (miny + maxy) / 2.0)
            labels = []
            for label, (tree, geoms) in label_indices.items():
                if not tree:
                    continue
                candidates = strtree_query_geoms(tree, geoms, tile_poly)
                hit = False
                for g in candidates:
                    inter = tile_poly.intersection(g)
                    if not inter.is_empty:
                        if inter.area / max(tile_area, 1e-9) >= args.min_intersection:
                            hit = True
                            break
                    if args.center_in and g.contains(center):
                        hit = True
                        break
                if hit:
                    labels.append(label)

            railway_near = False
            if "railway" in label_indices and args.near_buffer > 0:
                tree, geoms = label_indices["railway"]
                if tree and strtree_query_geoms(tree, geoms, tile_poly.buffer(args.near_buffer)):
                    railway_near = True

            blue_ratio = 0.0
            blue_roof = False
            if args.blue_roof and bldg_tree:
                candidates = strtree_query_geoms(bldg_tree, bldg_geoms, tile_poly)
                if candidates:
                    affine = Affine(*tref.transform)
                    mask = rasterize(
                        [(g, 1) for g in candidates],
                        out_shape=(rgb.shape[0], rgb.shape[1]),
                        transform=affine,
                        fill=0,
                        dtype=np.uint8,
                    )
                    blue_ratio = _blue_roof_ratio(
                        rgb, mask, args.blue_h_min, args.blue_h_max, args.blue_s_min, args.blue_v_min
                    )
                    blue_roof = blue_ratio >= args.blue_ratio

            split = _split_from_center(center.x, center.y, args.grid_size, ratios, seed=23)
            out_dir = os.path.join(args.out, "images", split)
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{image_id}_{tref.tile_id}.{args.format}"
            out_path = os.path.join(out_dir, fname)
            Image.fromarray(rgb).save(out_path)

            rec = {
                "split": split,
                "image_path": out_path,
                "source_image": img_path,
                "tile_id": tref.tile_id,
                "bounds": [minx, miny, maxx, maxy],
                "labels": labels,
                "attrs": {
                    "railway_near": railway_near,
                    "blue_roof": blue_roof,
                    "blue_ratio": round(blue_ratio, 4),
                },
            }
            jsonl_f.write(json.dumps(rec) + "\n")

            if csv_f and labels:
                prompts = _make_prompts(labels, blue_roof, args.max_prompts)
                for p in prompts:
                    csv_f.write(f"{out_path},{p}\n")

    jsonl_f.close()
    if csv_f:
        csv_f.close()


if __name__ == "__main__":
    main()
