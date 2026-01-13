import os, json, asyncio, logging
from typing import Optional
from .tiling import ImageTiler
from .gates import HeuristicGate, CLIPGate, CNNGate, NoopGate
from .detector import YOLODetector
from .vlm import VLMAnnotator
from .geo import GeoAggregator
from .pipeline import TileCache, process_image, load_geoobjects_jsonl
from .change import Coregistrator, ChangeDetector
from .geo_utils import require_projected_crs
from .semantic_index import EmbeddingClient, VectorIndex, SemanticSearcher, build_semantic_index
from .telemetry import init_logging, init_tracing, get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


def _env_flag(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _require_file(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _parse_prompt_list(value: str, fallback):
    if not value:
        return fallback
    sep = "|" if "|" in value else ","
    items = [p.strip() for p in value.split(sep)]
    items = [p for p in items if p]
    return items if items else fallback


def _build_gate():
    g = os.environ.get("GATE", "heuristic").strip().lower()
    if g == "none":
        logger.info("Gate selection: none")
        return NoopGate()
    if g == "clip":
        keep_default = ["dense urban area", "buildings and roads", "parking lot with many cars", "industrial site"]
        drop_default = ["forest", "agricultural field", "water surface", "clouds"]
        model_name = os.environ.get("CLIP_MODEL", "ViT-B-32").strip()
        pretrained = os.environ.get("CLIP_PRETRAINED", "openai").strip() or "openai"
        logger.info("Gate selection: clip model=%s threshold=%s", model_name, os.environ.get("CLIP_GATE_THR", "0.15"))
        return CLIPGate(
            keep_prompts=_parse_prompt_list(os.environ.get("CLIP_KEEP_PROMPTS", ""), keep_default),
            drop_prompts=_parse_prompt_list(os.environ.get("CLIP_DROP_PROMPTS", ""), drop_default),
            threshold=float(os.environ.get("CLIP_GATE_THR", "0.15")),
            device=os.environ.get("CLIP_DEVICE", "cpu"),
            model_name=model_name,
            pretrained=pretrained,
        )
    if g == "cnn":
        ckpt = os.environ.get("CNN_GATE_CKPT")
        if not ckpt:
            raise RuntimeError("CNN gate selected but CNN_GATE_CKPT is not set")
        logger.info("Gate selection: cnn ckpt=%s threshold=%s", ckpt, os.environ.get("CNN_GATE_THR", "0.5"))
        return CNNGate(
            ckpt_path=ckpt,
            threshold=float(os.environ.get("CNN_GATE_THR", "0.5")),
            device=os.environ.get("CNN_DEVICE", "cpu"),
        )
    logger.info("Gate selection: heuristic score=%s edge=%s entropy=%s",
                os.environ.get("H_SCORE", "0.24"), os.environ.get("H_EDGE", "0.05"), os.environ.get("H_ENT", "3.2"))
    return HeuristicGate(
        min_score=float(os.environ.get("H_SCORE", "0.24")),
        min_edge_density=float(os.environ.get("H_EDGE", "0.05")),
        min_entropy=float(os.environ.get("H_ENT", "3.2")),
    )


async def cmd_analyze(a_path: str, b_path: str, out_dir: str):
    with tracer.start_as_current_span("cmd.analyze") as span:
        span.set_attribute("input.a", a_path)
        span.set_attribute("input.b", b_path)
        span.set_attribute("output.dir", out_dir)
        logger.info("Analyze start a=%s b=%s out=%s", a_path, b_path, out_dir)
        _require_file(a_path, "GeoTIFF A")
        _require_file(b_path, "GeoTIFF B")
        os.makedirs(out_dir, exist_ok=True)
        tiler = ImageTiler(tile_size=int(os.environ.get("TILE_SIZE", "1024")),
                           overlap=int(os.environ.get("OVERLAP", "256")))
        gate = _build_gate()
        detector = YOLODetector(model_path=os.environ.get("YOLO_MODEL", "yolo12n.pt"),
                                conf=float(os.environ.get("YOLO_CONF", "0.25")),
                                max_det=int(os.environ.get("YOLO_MAX_DET", "400")),
                                device=os.environ.get("YOLO_DEVICE", "cpu"))
        aggregator = GeoAggregator(nms_iou=float(os.environ.get("NMS_IOU", "0.6")))
        cache = TileCache(os.path.join(out_dir, "cache"))
        max_inflight = int(os.environ.get("MAX_INFLIGHT_TILES", "32"))

        base_url = _require_env("VLM_BASE_URL")
        api_key = _require_env("VLM_API_KEY")
        model = os.environ.get("VLM_MODEL", "gpt-4o")
        vlm = VLMAnnotator(base_url=base_url, api_key=api_key, model=model,
                           concurrency=int(os.environ.get("VLM_CONCURRENCY", "6")))

        coreg = Coregistrator()
        shift_xy = coreg.estimate_shift(a_path, b_path)
        logger.info("Coregistration shift dx=%s dy=%s", shift_xy[0], shift_xy[1])

        try:
            out_a = os.path.join(out_dir, "a.objects.jsonl")
            out_b = os.path.join(out_dir, "b.objects.jsonl")
            a_id, crs_wkt, a_objs = await process_image(
                a_path, out_a, tiler, gate, detector, vlm, cache, aggregator, max_inflight=max_inflight,
                allow_empty_vlm=_env_flag("ALLOW_EMPTY_VLM", default=False)
            )
            b_id, _, b_objs = await process_image(
                b_path, out_b, tiler, gate, detector, vlm, cache, aggregator, max_inflight=max_inflight,
                allow_empty_vlm=_env_flag("ALLOW_EMPTY_VLM", default=False)
            )
            if not crs_wkt:
                raise RuntimeError("GeoTIFF CRS is missing; distance buffers require projected CRS in meters.")
            require_projected_crs(crs_wkt, "Change detection")

            cd = ChangeDetector(iou_match=float(os.environ.get("MATCH_IOU", "0.35")),
                                buffer_tol=float(os.environ.get("BUFFER_TOL", "4.0")),
                                vehicle_cluster_radius_m=float(os.environ.get("VEH_R", "25.0")))
            report = cd.diff(a_id, b_id, crs_wkt, a_objs, b_objs, b_shift_xy=shift_xy)

            out_report = os.path.join(out_dir, "change_report.json")
            with open(out_report, "w", encoding="utf-8") as f:
                json.dump(report.model_dump(), f, ensure_ascii=False)
            logger.info("Analyze outputs objects_a=%s objects_b=%s report=%s", out_a, out_b, out_report)
            print(out_a)
            print(out_b)
            print(out_report)
        finally:
            await vlm.close()


async def cmd_analyze_single(image_path: str, out_dir: str):
    with tracer.start_as_current_span("cmd.analyze_single") as span:
        span.set_attribute("input.image", image_path)
        span.set_attribute("output.dir", out_dir)
        logger.info("Analyze single start image=%s out=%s", image_path, out_dir)
        _require_file(image_path, "GeoTIFF")
        os.makedirs(out_dir, exist_ok=True)
        tiler = ImageTiler(tile_size=int(os.environ.get("TILE_SIZE", "1024")),
                           overlap=int(os.environ.get("OVERLAP", "256")))
        gate = _build_gate()
        detector = YOLODetector(model_path=os.environ.get("YOLO_MODEL", "yolo12n.pt"),
                                conf=float(os.environ.get("YOLO_CONF", "0.25")),
                                max_det=int(os.environ.get("YOLO_MAX_DET", "400")),
                                device=os.environ.get("YOLO_DEVICE", "cpu"))
        aggregator = GeoAggregator(nms_iou=float(os.environ.get("NMS_IOU", "0.6")))
        cache = TileCache(os.path.join(out_dir, "cache"))
        max_inflight = int(os.environ.get("MAX_INFLIGHT_TILES", "32"))

        base_url = _require_env("VLM_BASE_URL")
        api_key = _require_env("VLM_API_KEY")
        model = os.environ.get("VLM_MODEL", "gpt-4o")
        vlm = VLMAnnotator(base_url=base_url, api_key=api_key, model=model,
                           concurrency=int(os.environ.get("VLM_CONCURRENCY", "6")))

        try:
            out_path = os.path.join(out_dir, "objects.jsonl")
            await process_image(image_path, out_path, tiler, gate, detector, vlm, cache, aggregator,
                                max_inflight=max_inflight, allow_empty_vlm=_env_flag("ALLOW_EMPTY_VLM", default=False))
            logger.info("Analyze single output objects=%s", out_path)
            print(out_path)
        finally:
            await vlm.close()


def cmd_build_index(objects_jsonl: str, out_index_dir: str, change_report_json: Optional[str],
                    rail_geojson: Optional[str], tile_manifest_jsonl: Optional[str], crs_wkt: Optional[str]):
    with tracer.start_as_current_span("cmd.build_index") as span:
        span.set_attribute("input.objects", objects_jsonl)
        span.set_attribute("output.index_dir", out_index_dir)
        logger.info("Build index start objects=%s out=%s", objects_jsonl, out_index_dir)
        _require_file(objects_jsonl, "Objects JSONL")
        if change_report_json:
            _require_file(change_report_json, "Change report")
        if rail_geojson:
            _require_file(rail_geojson, "Rail GeoJSON")
        if tile_manifest_jsonl:
            _require_file(tile_manifest_jsonl, "Tile manifest")
        base_url = _require_env("EMB_BASE_URL")
        api_key = _require_env("EMB_API_KEY")
        model = os.environ.get("EMB_MODEL", "text-embedding-3-large")
        emb = EmbeddingClient(base_url=base_url, api_key=api_key, model=model)
        try:
            p = build_semantic_index(objects_jsonl, out_index_dir, emb, change_report_json=change_report_json,
                                     rail_geojson=rail_geojson, tile_manifest_jsonl=tile_manifest_jsonl, crs_wkt=crs_wkt)
            logger.info("Build index output index_dir=%s", p)
            print(p)
        finally:
            emb.close()


def cmd_search(index_dir: str, q: str):
    with tracer.start_as_current_span("cmd.search") as span:
        span.set_attribute("input.index_dir", index_dir)
        span.set_attribute("query", q)
        logger.info("Search start index=%s query=%s", index_dir, q)
        if not os.path.isdir(index_dir):
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        base_url = _require_env("EMB_BASE_URL")
        api_key = _require_env("EMB_API_KEY")
        model = os.environ.get("EMB_MODEL", "text-embedding-3-large")
        emb = EmbeddingClient(base_url=base_url, api_key=api_key, model=model)
        try:
            idx = VectorIndex.load(index_dir)
            s = SemanticSearcher(idx, emb)
            hits = s.query(q, top_k=int(os.environ.get("TOP_K", "10")))
            logger.info("Search hits count=%s", len(hits))
            for h in hits:
                print(json.dumps({"score": h.score, "doc_id": h.doc_id, "kind": h.kind, "image_id": h.image_id,
                                  "centroid": h.centroid, "meta": h.meta, "preview": h.preview}, ensure_ascii=False))
        finally:
            emb.close()


def main():
    import argparse
    init_logging()
    init_tracing("geovlm-poc")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("--a", required=True)
    a.add_argument("--b", required=True)
    a.add_argument("--out", required=True)

    ao = sub.add_parser("analyze-single")
    ao.add_argument("--image", required=True)
    ao.add_argument("--out", required=True)

    bi = sub.add_parser("build-index")
    bi.add_argument("--objects", required=True)
    bi.add_argument("--out-index", required=True)
    bi.add_argument("--changes", default=None)
    bi.add_argument("--rail", default=None)
    bi.add_argument("--tiles-manifest", default=None)
    bi.add_argument("--crs-wkt", default=None)

    s = sub.add_parser("search")
    s.add_argument("--index", required=True)
    s.add_argument("--q", required=True)

    args = ap.parse_args()
    logger.info("Command start cmd=%s", args.cmd)
    if args.cmd == "analyze":
        asyncio.run(cmd_analyze(args.a, args.b, args.out))
    elif args.cmd == "analyze-single":
        asyncio.run(cmd_analyze_single(args.image, args.out))
    elif args.cmd == "build-index":
        cmd_build_index(args.objects, args.out_index, args.changes, args.rail, args.tiles_manifest, args.crs_wkt)
    elif args.cmd == "search":
        cmd_search(args.index, args.q)


if __name__ == "__main__":
    main()
