import os
import sys
import json
import time
import threading
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

APP_ROOT = os.path.dirname(__file__)
SRC_ROOT = os.path.join(APP_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from geovlm_poc.tiling import ImageTiler
from geovlm_poc.gates import HeuristicGate, CLIPGate, CNNGate, NoopGate
from geovlm_poc.detector import YOLODetector
from geovlm_poc.vlm import VLMAnnotator
from geovlm_poc.geo import GeoAggregator
from geovlm_poc.pipeline import TileCache, NullTileCache, process_image, load_geoobjects_jsonl
from geovlm_poc.change import Coregistrator, ChangeDetector
from geovlm_poc.geo_utils import is_geographic_crs, require_projected_crs
from geovlm_poc.semantic_index import EmbeddingClient, VectorIndex, SemanticSearcher, build_semantic_index

try:
    import folium
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover - optional at runtime
    folium = None
    st_folium = None

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional at runtime
    CRS = None
    Transformer = None


def _list_images(input_dir: str) -> List[str]:
    if not input_dir or not os.path.isdir(input_dir):
        return []
    exts = (".tif", ".tiff", ".jp2", ".png")
    files = []
    for entry in os.scandir(input_dir):
        if entry.is_file() and entry.name.lower().endswith(exts):
            files.append(entry.path)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _format_file_label(path: str) -> str:
    stinfo = os.stat(path)
    size_mb = stinfo.st_size / (1024 * 1024)
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(stinfo.st_mtime))
    return f"{os.path.basename(path)} | {size_mb:.1f} MB | {mtime}"


def _parse_int_list(raw: str, fallback: Tuple[int, ...]) -> Tuple[int, ...]:
    if not raw:
        return fallback
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            return fallback
    return tuple(out) if out else fallback


def _parse_prompt_lines(raw: str, fallback: List[str]) -> List[str]:
    if not raw:
        return fallback
    lines = [l.strip() for l in raw.splitlines()]
    out = [l for l in lines if l]
    return out if out else fallback


def _safe_run_id(prefix: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)[:40].strip("_")
    return f"{ts}_{base}" if base else ts


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tail_lines(path: str, n: int = 200) -> str:
    if not path or not os.path.exists(path):
        return ""
    if n <= 0:
        return ""
    chunk_size = 4096
    data = b""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        while end > 0 and data.count(b"\n") <= n:
            read_size = min(chunk_size, end)
            end -= read_size
            f.seek(end)
            data = f.read(read_size) + data
            if end == 0:
                break
    lines = data.splitlines()
    tail = lines[-n:] if n else lines
    text = "\n".join(line.decode("utf-8", errors="replace") for line in tail)
    if data.endswith(b"\n"):
        text += "\n"
    return text


def _attach_run_file_handler(log_path: str) -> logging.Handler:
    root = logging.getLogger()
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if not root.level or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return fh


def _detach_run_file_handler(handler: logging.Handler) -> None:
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()


def _ensure_file_logger(log_path: str) -> logging.Handler:
    root = logging.getLogger()
    abs_path = os.path.abspath(log_path)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(handler.baseFilename) == abs_path:
                return handler
    return _attach_run_file_handler(log_path)


def _build_gate(mode: str, params: Dict[str, Any]):
    if mode == "none":
        return NoopGate()
    if mode == "clip":
        return CLIPGate(
            keep_prompts=params["keep_prompts"],
            drop_prompts=params["drop_prompts"],
            threshold=params["threshold"],
            device=params["device"],
            model_name=params["model_name"],
            pretrained=params["pretrained"],
        )
    if mode == "cnn":
        return CNNGate(
            ckpt_path=params["ckpt_path"],
            threshold=params["threshold"],
            device=params["device"],
        )
    return HeuristicGate(
        min_score=params["min_score"],
        min_edge_density=params["min_edge_density"],
        min_entropy=params["min_entropy"],
    )


def _job_state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "job_state.json")


def _start_analysis_job(cfg: Dict[str, Any]) -> None:
    run_dir = cfg["run_dir"]
    log_path = os.path.join(run_dir, "logs.txt")
    job_path = _job_state_path(run_dir)

    def _worker():
        _ensure_file_logger(log_path)
        job_state = {
            "status": "running",
            "started_at": time.time(),
            "run_dir": run_dir,
            "image_path": cfg["image_path"],
            "event": "start",
        }

        def _progress_cb(update: Dict[str, Any]) -> None:
            job_state.update(update)
            job_state["updated_at"] = time.time()
            _write_json(job_path, job_state)

        async def _run():
            tiler = ImageTiler(tile_size=cfg["tile_size"], overlap=cfg["overlap"], bands=cfg["bands"])
            gate = _build_gate(cfg["gate_mode"], cfg["gate_params"])
            detector = YOLODetector(
                model_path=cfg["yolo_model"],
                conf=cfg["yolo_conf"],
                max_det=cfg["yolo_max_det"],
                device=cfg["yolo_device"],
            )
            aggregator = GeoAggregator(nms_iou=cfg["nms_iou"])
            cache = TileCache(cfg["cache_dir"]) if cfg["use_cache"] else NullTileCache()
            vlm = VLMAnnotator(
                base_url=cfg["vlm_base_url"],
                api_key=cfg["vlm_api_key"],
                model=cfg["vlm_model"],
                timeout_s=cfg["vlm_timeout_s"],
                concurrency=cfg["vlm_concurrency"],
            )
            try:
                image_id, crs_wkt, objs = await process_image(
                    cfg["image_path"],
                    cfg["out_jsonl"],
                    tiler,
                    gate,
                    detector,
                    vlm,
                    cache,
                    aggregator,
                    max_tiles=cfg["max_tiles"],
                    max_inflight=cfg["max_inflight"],
                    save_tiles=cfg["save_tiles"],
                    tiles_dir=cfg["tiles_dir"],
                    tile_manifest_path=cfg["tile_manifest"],
                    progress_cb=_progress_cb,
                    allow_empty_vlm=cfg["allow_empty_vlm"],
                )
                job_state.update({
                    "status": "done",
                    "finished_at": time.time(),
                    "image_id": image_id,
                    "crs_wkt": crs_wkt,
                    "objects_total": len(objs),
                    "geoobjects_jsonl": cfg["out_jsonl"],
                    "tile_manifest": cfg["tile_manifest"],
                })
                _write_json(job_path, job_state)
                cfg_update = dict(cfg)
                cfg_update["image_id"] = image_id
                cfg_update["crs_wkt"] = crs_wkt
                _write_json(os.path.join(run_dir, "run_config.json"), cfg_update)
            except Exception as exc:
                job_state.update({
                    "status": "error",
                    "finished_at": time.time(),
                    "error": str(exc),
                })
                _write_json(job_path, job_state)
            finally:
                await vlm.close()

        _write_json(job_path, job_state)
        asyncio.run(_run())

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _collect_index_dirs(base_dir: str) -> List[str]:
    if not base_dir or not os.path.isdir(base_dir):
        return []
    out = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            idx = os.path.join(entry.path, "index")
            if os.path.isdir(idx):
                out.append(idx)
    out.sort()
    return out


def _collect_runs(base_dir: str) -> List[str]:
    if not base_dir or not os.path.isdir(base_dir):
        return []
    out = [e.path for e in os.scandir(base_dir) if e.is_dir()]
    out.sort()
    return out


@st.cache_resource
def _load_index(index_dir: str) -> VectorIndex:
    return VectorIndex.load(index_dir)


@st.cache_data
def _load_tile_manifest_map(path: str) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    out = {}
    base = os.path.dirname(path) or "."
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tile_id = item.get("tile_id")
            if not tile_id:
                continue
            png_path = item.get("png_path")
            if png_path and not os.path.isabs(png_path):
                png_path = os.path.join(base, png_path)
            item["png_path"] = png_path
            out[tile_id] = item
    return out


def _geom_to_wgs84(geom: Dict[str, Any], crs_wkt: Optional[str]) -> Optional[Dict[str, Any]]:
    if not geom:
        return None
    if not crs_wkt:
        return None
    if is_geographic_crs(crs_wkt):
        return geom
    if CRS is None or Transformer is None:
        return None
    try:
        src = CRS.from_wkt(crs_wkt)
        dst = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(src, dst, always_xy=True)
        g = shape(geom)
        g2 = shp_transform(transformer.transform, g)
        return mapping(g2)
    except Exception:
        return None


def _render_tile_preview(png_path: str, dets: List[Dict[str, Any]], highlight_det_id: Optional[int]) -> Image.Image:
    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in dets or []:
        x1, y1, x2, y2 = d.get("bbox_px", [0, 0, 0, 0])
        color = "red" if highlight_det_id is not None and d.get("det_id") == highlight_det_id else "yellow"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = d.get("cls") or ""
        if label:
            draw.text((x1 + 2, y1 + 2), label, fill=color)
    return img


def _collect_filter_values(docs: List[Any]) -> Dict[str, List[str]]:
    labels = set()
    roof_colors = set()
    change_types = set()
    for d in docs:
        m = d.meta or {}
        lab = m.get("label")
        if isinstance(lab, (list, tuple, set)):
            labels.update([str(x) for x in lab if x])
        elif lab:
            labels.add(str(lab))
        rc = m.get("roof_color")
        if isinstance(rc, (list, tuple, set)):
            roof_colors.update([str(x) for x in rc if x])
        elif rc:
            roof_colors.add(str(rc))
        ct = m.get("change_type")
        if ct:
            change_types.add(str(ct))
    return {
        "labels": sorted(labels),
        "roof_colors": sorted(roof_colors),
        "change_types": sorted(change_types),
    }


def _build_map(features: List[Dict[str, Any]], center: Optional[Tuple[float, float]]):
    if folium is None or st_folium is None:
        st.warning("Folium/streamlit-folium не встановлені. Встановіть extras для мапи.")
        return
    if not center:
        center = (50.45, 30.52)
    m = folium.Map(location=[center[1], center[0]], zoom_start=12, tiles="CartoDB positron")
    color_map = {
        "added": "#2ca02c",
        "removed": "#d62728",
        "modified": "#ff7f0e",
        "object": "#1f77b4",
        "tile": "#9467bd",
    }
    for ft in features:
        props = ft.get("properties") or {}
        kind = props.get("kind")
        change_type = props.get("change_type")
        label = props.get("label")
        color = color_map.get(change_type or kind, "#1f77b4")
        score = props.get("score") or 0.0
        tooltip = f"{label or kind} | score={score:.3f}"
        folium.GeoJson(
            ft,
            tooltip=tooltip,
            style_function=lambda x, c=color: {"color": c, "weight": 2, "fillOpacity": 0.25},
        ).add_to(m)
    st_folium(m, width=None, height=520)


def _hit_center_wgs84(hit_geom: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not hit_geom:
        return None
    g = shape(hit_geom)
    c = g.centroid
    return (float(c.x), float(c.y))


def _parse_det_id_from_doc_id(doc_id: str) -> Optional[int]:
    if not doc_id or not doc_id.startswith("obj:"):
        return None
    obj_id = doc_id.split("obj:", 1)[-1]
    m = re.search(r"_d(\d+)$", obj_id)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


st.set_page_config(page_title="GeoVLM UI", layout="wide")
st.title("GeoVLM UI")

tab_analyze, tab_search, tab_compare = st.tabs(["Analyze", "Search", "Compare"])

with tab_analyze:
    st.subheader("Analyze")
    runs_root = st.text_input("Runs directory", value=os.path.join(APP_ROOT, "runs"))
    input_dir = st.text_input("Input directory (GeoTIFF/JP2/PNG)", value=os.path.join(APP_ROOT, "data"))
    refresh = st.button("Refresh list")
    if refresh:
        st.session_state["image_list"] = _list_images(input_dir)
    if "image_list" not in st.session_state:
        st.session_state["image_list"] = _list_images(input_dir)
    images = st.session_state.get("image_list") or []
    image_map = {(_format_file_label(p)): p for p in images}
    if not images:
        st.info("Немає файлів у папці.")
    selected_label = st.selectbox("Satellite image", options=list(image_map.keys())) if images else None
    image_path = image_map.get(selected_label) if selected_label else None

    st.sidebar.header("Параметри аналізу")
    with st.sidebar.expander("Tiling", expanded=True):
        tile_size = st.number_input("tile_size", min_value=128, max_value=4096, value=1024, step=64)
        overlap = st.number_input("overlap", min_value=0, max_value=2048, value=256, step=32)
        bands_raw = st.text_input("bands (comma-separated)", value="1,2,3")
        st.caption("step = tile_size - overlap")
    with st.sidebar.expander("Gate", expanded=True):
        gate_mode = st.selectbox("GATE_MODE", options=["none", "heuristic", "clip", "cnn"])
        gate_params: Dict[str, Any] = {}
        if gate_mode == "heuristic":
            gate_params["min_score"] = st.number_input("H_SCORE", value=0.24, step=0.01)
            gate_params["min_edge_density"] = st.number_input("H_EDGE", value=0.05, step=0.01)
            gate_params["min_entropy"] = st.number_input("H_ENT", value=3.2, step=0.1)
            st.caption("чим вище — тим строгіше")
        elif gate_mode == "clip":
            gate_params["model_name"] = st.text_input("CLIP model_name", value="ViT-B-32")
            gate_params["pretrained"] = st.text_input("CLIP pretrained", value="openai")
            gate_params["device"] = st.text_input("CLIP device", value="cpu")
            gate_params["threshold"] = st.number_input("CLIP threshold", value=0.15, step=0.01)
            keep_default = "dense urban area\nbuildings and roads\nparking lot with many cars\nindustrial site"
            drop_default = "forest\nagricultural field\nwater surface\nclouds"
            gate_params["keep_prompts"] = _parse_prompt_lines(
                st.text_area("CLIP keep prompts (one per line)", value=keep_default), []
            )
            gate_params["drop_prompts"] = _parse_prompt_lines(
                st.text_area("CLIP drop prompts (one per line)", value=drop_default), []
            )
        elif gate_mode == "cnn":
            gate_params["ckpt_path"] = st.text_input("CNN_GATE_CKPT", value=os.path.join(APP_ROOT, "cnn_gate", "model.pt"))
            gate_params["threshold"] = st.number_input("CNN_GATE_THR", value=0.5, step=0.05)
            gate_params["device"] = st.text_input("CNN_DEVICE", value="cpu")
            if st.button("Validate ckpt"):
                if os.path.exists(gate_params["ckpt_path"]):
                    st.success("Checkpoint exists.")
                else:
                    st.error("Checkpoint not found.")
    with st.sidebar.expander("Detector (YOLO)", expanded=True):
        yolo_model = st.text_input("model_path", value="yolo12s.pt")
        yolo_conf = st.number_input("conf", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        yolo_max_det = st.number_input("max_det", min_value=1, max_value=2000, value=400, step=50)
        yolo_device = st.text_input("device", value="cpu")
        nms_iou = st.number_input("NMS IOU", min_value=0.1, max_value=0.95, value=0.6, step=0.05)
    with st.sidebar.expander("VLM", expanded=True):
        vlm_base_url = st.text_input("VLM base_url", value="http://localhost:8000/v1")
        vlm_api_key = st.text_input("VLM api_key", type="password", value="")
        vlm_model = st.text_input("VLM model", value="gpt-4o")
        vlm_timeout_s = st.number_input("timeout_s", min_value=5.0, max_value=300.0, value=90.0, step=5.0)
        vlm_concurrency = st.number_input("concurrency", min_value=1, max_value=32, value=6, step=1)
        allow_empty_vlm = st.checkbox("Run VLM on empty tiles", value=False)
    with st.sidebar.expander("Cache", expanded=True):
        use_cache = st.checkbox("Use cache", value=True)
        cache_dir = st.text_input("cache_dir", value=os.path.join(APP_ROOT, "out", "cache"))
    with st.sidebar.expander("Output", expanded=True):
        max_tiles = st.number_input("max_tiles (0 = all)", min_value=0, max_value=100000, value=0, step=50)
        max_inflight = st.number_input("max_inflight", min_value=1, max_value=256, value=32, step=4)
        save_tiles = st.checkbox("Save tile PNG previews", value=True)
        st.caption("Artifacts stored in runs/<run_id>/")
    with st.sidebar.expander("Index", expanded=False):
        emb_base_url = st.text_input("Embeddings base_url", value="http://localhost:8000/v1", key="emb_base_url_analyze")
        emb_api_key = st.text_input("Embeddings api_key", type="password", value="", key="emb_api_key_analyze")
        emb_model = st.text_input("Embeddings model", value="text-embedding-3-large", key="emb_model_analyze")
        rail_geojson = st.text_input("rail_geojson (optional)", value="", key="rail_geojson_analyze")
        index_tiles = st.checkbox("INDEX_TILES (tile-level search)", value=True, key="index_tiles_analyze")

    cols = st.columns(4)
    run_clicked = cols[0].button("Run analysis")
    build_index_clicked = cols[1].button("Build / rebuild index")
    refresh_status = cols[2].button("Refresh status")
    open_last = cols[3].button("Open last results")

    if run_clicked and image_path:
        run_id = _safe_run_id(os.path.basename(image_path))
        run_dir = os.path.join(runs_root, run_id)
        tiles_dir = os.path.join(run_dir, "tiles")
        tile_manifest = os.path.join(run_dir, "tile_manifest.jsonl")
        out_jsonl = os.path.join(run_dir, "geoobjects.jsonl")
        cfg = {
            "run_dir": run_dir,
            "image_path": image_path,
            "tile_size": int(tile_size),
            "overlap": int(overlap),
            "bands": _parse_int_list(bands_raw, (1, 2, 3)),
            "gate_mode": gate_mode,
            "gate_params": gate_params,
            "yolo_model": yolo_model,
            "yolo_conf": float(yolo_conf),
            "yolo_max_det": int(yolo_max_det),
            "yolo_device": yolo_device,
            "nms_iou": float(nms_iou),
            "vlm_base_url": vlm_base_url,
            "vlm_api_key": vlm_api_key,
            "vlm_model": vlm_model,
            "vlm_timeout_s": float(vlm_timeout_s),
            "vlm_concurrency": int(vlm_concurrency),
            "allow_empty_vlm": bool(allow_empty_vlm),
            "use_cache": bool(use_cache),
            "cache_dir": cache_dir,
            "max_tiles": int(max_tiles) if int(max_tiles) > 0 else None,
            "max_inflight": int(max_inflight),
            "save_tiles": bool(save_tiles),
            "tiles_dir": tiles_dir,
            "tile_manifest": tile_manifest,
            "out_jsonl": out_jsonl,
        }
        _write_json(os.path.join(run_dir, "run_config.json"), cfg)
        _start_analysis_job(cfg)
        st.session_state["last_run_dir"] = run_dir
        st.success(f"Analysis started: {run_dir}")
    elif run_clicked and not image_path:
        st.warning("Оберіть знімок.")

    if refresh_status:
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    if open_last and st.session_state.get("last_run_dir"):
        st.write(f"Last run: {st.session_state['last_run_dir']}")

    current_run = st.session_state.get("last_run_dir")
    if current_run:
        job_state = _read_json(_job_state_path(current_run))
        if job_state:
            status = job_state.get("status", "unknown")
            st.write(f"Status: {status}")
            total = job_state.get("total_tiles", 0)
            processed = job_state.get("processed_tiles", 0)
            if total:
                st.progress(min(processed / max(total, 1), 1.0))
            metrics = st.columns(5)
            metrics[0].metric("tiles", f"{processed}/{total}")
            metrics[1].metric("kept", job_state.get("kept_tiles", 0))
            metrics[2].metric("dets", job_state.get("dets_total", 0))
            metrics[3].metric("objects", job_state.get("objects_total", 0))
            hits = job_state.get("cache_hits", 0)
            misses = job_state.get("cache_misses", 0)
            metrics[4].metric("cache hit %", f"{(hits / max(hits + misses, 1)) * 100:.1f}")
            log_tail = _tail_lines(os.path.join(current_run, "logs.txt"), n=200)
            st.text_area("Logs (last 200 lines)", value=log_tail, height=240)

            geoobjects_jsonl = job_state.get("geoobjects_jsonl")
            crs_wkt = job_state.get("crs_wkt")
            if status == "done" and geoobjects_jsonl and os.path.exists(geoobjects_jsonl):
                if st.checkbox("Show map preview", value=False):
                    objs = load_geoobjects_jsonl(geoobjects_jsonl)
                    features = []
                    centers = []
                    for o in objs:
                        g_wgs = _geom_to_wgs84(o.geometry, crs_wkt)
                        if not g_wgs:
                            continue
                        ctr = _hit_center_wgs84(g_wgs)
                        if ctr:
                            centers.append(ctr)
                        features.append({
                            "type": "Feature",
                            "geometry": g_wgs,
                            "properties": {"label": o.label, "score": o.confidence, "kind": "object"},
                        })
                    center = centers[0] if centers else None
                    if features:
                        _build_map(features, center)
                    else:
                        st.info("Неможливо показати мапу (CRS або конвертація).")

        if build_index_clicked and current_run:
            cfg = _read_json(os.path.join(current_run, "run_config.json")) or {}
            objects_jsonl = cfg.get("out_jsonl") or os.path.join(current_run, "geoobjects.jsonl")
            tile_manifest = cfg.get("tile_manifest") or os.path.join(current_run, "tile_manifest.jsonl")
            crs_wkt = cfg.get("crs_wkt")
            if not os.path.exists(objects_jsonl):
                st.error("Не знайдено geoobjects.jsonl для цього запуску.")
            else:
                index_dir = os.path.join(current_run, "index")
                os.environ["INDEX_TILES"] = "1" if index_tiles else "0"
                emb = EmbeddingClient(base_url=emb_base_url, api_key=emb_api_key, model=emb_model)
                try:
                    with st.spinner("Building index..."):
                        build_semantic_index(
                            objects_jsonl,
                            index_dir,
                            emb,
                            change_report_json=None,
                            rail_geojson=rail_geojson or None,
                            tile_manifest_jsonl=tile_manifest,
                            crs_wkt=crs_wkt,
                        )
                    st.success(f"Index built: {index_dir}")
                    st.session_state["last_index_dir"] = index_dir
                finally:
                    emb.close()

with tab_search:
    st.subheader("Search")
    runs_root = st.text_input("Runs directory (search)", value=os.path.join(APP_ROOT, "runs"))
    index_dirs = _collect_index_dirs(runs_root)
    index_dir = st.selectbox("Index dir", options=index_dirs) if index_dirs else None
    emb_base_url = st.text_input("Embeddings base_url", value="http://localhost:8000/v1")
    emb_api_key = st.text_input("Embeddings api_key", type="password", value="")
    emb_model = st.text_input("Embeddings model", value="text-embedding-3-large")
    query_text = st.text_area("Запит (українською)", value="знайди склади з синіми дахами біля залізниці")
    top_k = st.slider("top_k", min_value=1, max_value=100, value=15, step=1)

    docs = []
    idx = None
    if index_dir:
        try:
            idx = _load_index(index_dir)
            docs = idx.docs
        except Exception as exc:
            st.error(f"Не вдалося завантажити індекс: {exc}")

    filters = _collect_filter_values(docs) if docs else {"labels": [], "roof_colors": [], "change_types": []}
    cols = st.columns(4)
    scope = cols[0].selectbox("Scope", options=["all", "object", "tile", "change"])
    label_filter = cols[1].selectbox("label", options=[""] + filters["labels"])
    roof_filter = cols[2].selectbox("roof_color", options=[""] + filters["roof_colors"])
    change_filter = cols[3].selectbox("change_type", options=[""] + filters["change_types"])
    rail_dist_max = st.number_input("rail_dist_max_m (optional)", min_value=0.0, value=0.0, step=25.0)

    run_search = st.button("Search")
    hits = []
    if run_search and index_dir and query_text.strip():
        emb = EmbeddingClient(base_url=emb_base_url, api_key=emb_api_key, model=emb_model)
        try:
            searcher = SemanticSearcher(idx, emb)
            flt: Dict[str, Any] = {}
            if scope != "all":
                flt["kind"] = scope
            if label_filter:
                flt["label"] = label_filter
            if roof_filter:
                flt["roof_color"] = roof_filter
            if change_filter:
                flt["change_type"] = change_filter
            if rail_dist_max > 0:
                flt["rail_dist_max_m"] = rail_dist_max
            hits = searcher.query(query_text, top_k=top_k, filters=flt if flt else None)
            st.session_state["search_hits"] = hits
        finally:
            emb.close()
    elif st.session_state.get("search_hits"):
        hits = st.session_state["search_hits"]

    if hits:
        meta = _read_json(os.path.join(index_dir, "index_meta.json")) if index_dir else {}
        crs_wkt = meta.get("crs_wkt")
        tile_manifest = meta.get("tile_manifest_jsonl")
        tiles_map = _load_tile_manifest_map(tile_manifest) if tile_manifest else {}
        options = [f"{i+1}. {h.kind} | score={h.score:.3f} | {h.preview}" for i, h in enumerate(hits)]
        sel = st.radio("Results", options=options, index=0)
        sel_idx = options.index(sel) if sel in options else 0
        hit = hits[sel_idx]

        cols = st.columns([0.35, 0.65])
        with cols[0]:
            st.write(f"doc_id: {hit.doc_id}")
            st.write(f"kind: {hit.kind}")
            st.write(f"score: {hit.score:.3f}")
            st.json(hit.meta)
            tile_id = (hit.meta or {}).get("tile_id")
            tile_entry = tiles_map.get(tile_id) if tile_id else None
            if tile_entry and tile_entry.get("png_path") and os.path.exists(tile_entry.get("png_path")):
                det_id = _parse_det_id_from_doc_id(hit.doc_id)
                if st.checkbox("Show detections overlay", value=True):
                    img = _render_tile_preview(tile_entry["png_path"], tile_entry.get("dets", []), det_id)
                    st.image(img, caption=f"Tile {tile_id}")
                else:
                    st.image(tile_entry["png_path"], caption=f"Tile {tile_id}")
        with cols[1]:
            features = []
            centers = []
            map_ok = True
            for h in hits:
                g = _geom_to_wgs84(h.geometry, crs_wkt)
                if not g:
                    map_ok = False
                    break
                ctr = _hit_center_wgs84(g)
                if ctr:
                    centers.append(ctr)
                props = dict(h.meta or {})
                props["score"] = h.score
                props["kind"] = h.kind
                features.append({"type": "Feature", "geometry": g, "properties": props})
            if map_ok and features:
                center = centers[0] if centers else None
                _build_map(features, center)
            else:
                st.info("CRS не географічний або немає pyproj. Показуємо тільки превʼю.")
    else:
        st.info("Немає результатів. Виконайте пошук.")

with tab_compare:
    st.subheader("Compare")
    runs_root = st.text_input("Runs directory (compare)", value=os.path.join(APP_ROOT, "runs"))
    run_dirs = _collect_runs(runs_root)
    if run_dirs:
        a_run = st.selectbox("Run A", options=run_dirs)
        b_run = st.selectbox("Run B", options=run_dirs)
    else:
        a_run = None
        b_run = None
        st.info("Немає запусків у runs/.")

    match_iou = st.number_input("MATCH_IOU", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    buffer_tol = st.number_input("BUFFER_TOL (m)", min_value=0.0, max_value=50.0, value=4.0, step=1.0)
    veh_r = st.number_input("VEH_R (m)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
    run_compare = st.button("Run compare")

    if run_compare and a_run and b_run:
        out_id = _safe_run_id("compare")
        out_dir = os.path.join(runs_root, out_id)
        os.makedirs(out_dir, exist_ok=True)
        cfg_a = _read_json(os.path.join(a_run, "run_config.json")) or {}
        cfg_b = _read_json(os.path.join(b_run, "run_config.json")) or {}
        obj_a = cfg_a.get("out_jsonl") or os.path.join(a_run, "geoobjects.jsonl")
        obj_b = cfg_b.get("out_jsonl") or os.path.join(b_run, "geoobjects.jsonl")
        if not (os.path.exists(obj_a) and os.path.exists(obj_b)):
            st.error("Не знайдено geoobjects.jsonl для одного з запусків.")
        else:
            _write_json(os.path.join(out_dir, "compare_config.json"), {
                "run_a": a_run,
                "run_b": b_run,
                "objects_a": obj_a,
                "objects_b": obj_b,
                "match_iou": match_iou,
                "buffer_tol": buffer_tol,
                "vehicle_cluster_radius_m": veh_r,
            })
            a_objs = load_geoobjects_jsonl(obj_a)
            b_objs = load_geoobjects_jsonl(obj_b)
            crs_wkt = cfg_a.get("crs_wkt") or cfg_b.get("crs_wkt") or ""
            try:
                require_projected_crs(crs_wkt, "Compare")
            except Exception as exc:
                st.warning(f"CRS не метричний: {exc}")
            shift_xy = (0.0, 0.0)
            if cfg_a.get("image_path") and cfg_b.get("image_path"):
                coreg = Coregistrator()
                shift_xy = coreg.estimate_shift(cfg_a["image_path"], cfg_b["image_path"])
            cd = ChangeDetector(iou_match=match_iou, buffer_tol=buffer_tol, vehicle_cluster_radius_m=veh_r)
            report = cd.diff(os.path.basename(a_run), os.path.basename(b_run), crs_wkt, a_objs, b_objs, b_shift_xy=shift_xy)
            report_path = os.path.join(out_dir, "change_report.json")
            _write_json(report_path, report.model_dump())
            st.success(f"Change report saved: {report_path}")

            features = []
            centers = []
            for ev in report.changes:
                g = _geom_to_wgs84(ev.geometry, crs_wkt)
                if not g:
                    continue
                ctr = _hit_center_wgs84(g)
                if ctr:
                    centers.append(ctr)
                features.append({
                    "type": "Feature",
                    "geometry": g,
                    "properties": {
                        "label": ev.label,
                        "kind": "change",
                        "change_type": ev.change_type,
                        "score": ev.score or 0.0,
                    },
                })
            if features:
                center = centers[0] if centers else None
                _build_map(features, center)
            else:
                st.info("Немає геометрій для мапи або CRS не підтримується.")
