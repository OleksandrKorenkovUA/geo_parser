import os, json, asyncio, hashlib, logging, time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from .types import DetBox, GeoObject, TileRef, TileSemantics
from .tiling import ImageTiler
from .gates import TileGate
from .detector import Detector
from .vlm import VLMAnnotator
from .geo import GeoAggregator
from .normalize import normalize_label, normalize_roof_color
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class TileCache:
    def __init__(self, dir_path: str):
        self.dir = dir_path
        self.enabled = True
        os.makedirs(self.dir, exist_ok=True)
        self.key_mode = os.environ.get("CACHE_KEY_MODE", "tile").strip().lower()
        if self.key_mode not in {"pixels", "source", "tile"}:
            logger.warning("Unknown CACHE_KEY_MODE=%s; defaulting to tile", self.key_mode)
            self.key_mode = "tile"
        self.max_files = int(os.environ.get("CACHE_MAX_FILES", "0")) or None
        self.max_mb = float(os.environ.get("CACHE_MAX_MB", "0") or 0.0) or None
        self.prune_every = max(1, int(os.environ.get("CACHE_PRUNE_EVERY", "25")))
        self._puts = 0
        self._prune()

    def key(self, image_id: str, tile_id: str, rgb: np.ndarray, model_id: str,
            dets_hash: Optional[str] = None) -> str:
        h = hashlib.sha256()
        h.update(model_id.encode("utf-8"))
        h.update(image_id.encode("utf-8"))
        h.update(tile_id.encode("utf-8"))
        if self.key_mode == "source":
            if not dets_hash:
                raise ValueError("dets_hash required for CACHE_KEY_MODE=source")
            h.update(dets_hash.encode("utf-8"))
        elif self.key_mode == "pixels":
            mv = memoryview(rgb)
            if not mv.contiguous:
                mv = memoryview(np.ascontiguousarray(rgb))
            h.update(mv)
        return h.hexdigest()

    def get(self, key: str) -> Optional[Dict]:
        p = os.path.join(self.dir, f"{key}.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def put(self, key: str, obj: Dict):
        p = os.path.join(self.dir, f"{key}.json")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, p)
        self._puts += 1
        if self._puts % self.prune_every == 0:
            self._prune()

    def _prune(self):
        if not self.max_files and not self.max_mb:
            return
        files = []
        total = 0
        try:
            names = os.listdir(self.dir)
        except FileNotFoundError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, path))
            total += st.st_size
        files.sort()
        max_bytes = int(self.max_mb * 1024 * 1024) if self.max_mb else None
        while files and ((self.max_files and len(files) > self.max_files) or (max_bytes and total > max_bytes)):
            _, size, path = files.pop(0)
            try:
                os.remove(path)
                total -= size
                logger.info("Cache prune removed path=%s", path)
            except OSError:
                logger.warning("Cache prune failed path=%s", path)


class NullTileCache:
    enabled = False

    def key(self, image_id: str, tile_id: str, rgb: np.ndarray, model_id: str,
            dets_hash: Optional[str] = None) -> str:
        return ""

    def get(self, key: str) -> Optional[Dict]:
        return None

    def put(self, key: str, obj: Dict):
        return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _stable_image_id(path: str) -> str:
    base = os.path.basename(path)
    h = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    return f"{base}__{h}"


def _hash_dets(dets: List[DetBox]) -> str:
    h = hashlib.sha256()
    for d in sorted(dets, key=lambda x: x.det_id):
        h.update(str(d.det_id).encode("utf-8"))
        h.update(b"|")
        h.update(str(d.cls or "").encode("utf-8"))
        h.update(b"|")
        h.update(f"{float(d.score):.4f}".encode("utf-8"))
        h.update(b"|")
        h.update(",".join(str(x) for x in d.bbox_px).encode("utf-8"))
        h.update(b";")
    return h.hexdigest()


def _hash_rgb(rgb: np.ndarray) -> str:
    h = hashlib.sha256()
    mv = memoryview(rgb)
    if not mv.contiguous:
        mv = memoryview(np.ascontiguousarray(rgb))
    h.update(mv)
    return h.hexdigest()


def _normalize_semantics(sem: TileSemantics) -> None:
    for ann in sem.annotations:
        ann.label = normalize_label(ann.label)
        attrs = ann.attributes
        if attrs is None:
            attrs = {}
            ann.attributes = attrs
        if "roof_color" in attrs:
            norm_color = normalize_roof_color(attrs.get("roof_color"))
            if norm_color:
                attrs["roof_color"] = norm_color


def _save_tile_png(rgb: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.fromarray(rgb, mode="RGB")
    im.save(path, format="PNG", optimize=True)


async def process_image(path: str, out_jsonl: str,
                        tiler: ImageTiler, gate: TileGate,
                        detector: Detector, vlm: VLMAnnotator,
                        cache: TileCache, aggregator: GeoAggregator,
                        max_tiles: Optional[int] = None,
                        max_inflight: int = 32,
                        save_tiles: bool = False,
                        tiles_dir: Optional[str] = None,
                        tile_manifest_path: Optional[str] = None,
                        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
                        allow_empty_vlm: bool = False) -> Tuple[str, str, List[GeoObject]]:
    with tracer.start_as_current_span("process_image") as span:
        span.set_attribute("image.path", path)
        span.set_attribute("output.jsonl", out_jsonl)
        max_inflight = max(1, int(max_inflight))
        span.set_attribute("max_inflight", max_inflight)
        image_id = _stable_image_id(path)
        tile_refs: List[TileRef] = []
        for i, t in enumerate(tiler.iter_tiles(path, image_id=image_id)):
            tile_refs.append(t)
            if max_tiles is not None and i + 1 >= max_tiles:
                break
        logger.info("Process image start image=%s tiles=%s out=%s max_inflight=%s",
                    image_id, len(tile_refs), out_jsonl, max_inflight)

        offload_cpu = _env_flag("ASYNC_OFFLOAD_CPU", default=False)
        image_mode = os.environ.get("VLM_IMAGE_MODE", "base64").strip().lower()
        cache_enabled = bool(getattr(cache, "enabled", True))
        if image_mode in {"url", "path"} and not save_tiles:
            save_tiles = True
            if not tiles_dir:
                tiles_dir = os.path.join(os.path.dirname(out_jsonl) or ".", "tiles")
            logger.info("VLM image mode=%s requires tile PNGs; enabling save_tiles tiles_dir=%s", image_mode, tiles_dir)
        if save_tiles and not tiles_dir:
            tiles_dir = os.path.join(os.path.dirname(out_jsonl) or ".", "tiles")
        tile_img_dir = os.path.join(tiles_dir, image_id) if (save_tiles and tiles_dir) else None
        if tile_img_dir:
            os.makedirs(tile_img_dir, exist_ok=True)
        if tile_manifest_path:
            os.makedirs(os.path.dirname(tile_manifest_path) or ".", exist_ok=True)
            with open(tile_manifest_path, "w", encoding="utf-8"):
                pass
        manifest_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        progress = {
            "total_tiles": len(tile_refs),
            "processed_tiles": 0,
            "kept_tiles": 0,
            "dets_total": 0,
            "objects_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        detector_workers = max(1, int(os.environ.get("DETECTOR_WORKERS", "4")))
        vlm_workers = max(1, int(os.environ.get("VLM_WORKERS", "4")))
        detector_workers = min(detector_workers, max_inflight)
        vlm_workers = min(vlm_workers, max_inflight)
        tile_queue = asyncio.Queue(maxsize=max_inflight)
        vlm_queue = asyncio.Queue(maxsize=max_inflight)
        objs = []
        objs_lock = asyncio.Lock()
        detector_model = getattr(detector, "model_path", None)
        gate_name = type(gate).__name__
        vlm_model = getattr(vlm, "model", None)
        vlm_prompt_id = getattr(vlm, "prompt_id", None)

        def _relpath(p: Optional[str]) -> Optional[str]:
            if not p or not tile_manifest_path:
                return p
            base = os.path.dirname(tile_manifest_path) or "."
            try:
                return os.path.relpath(p, base)
            except ValueError:
                return p

        async def _write_manifest(entry: Dict[str, Any]) -> None:
            if not tile_manifest_path:
                return
            async with manifest_lock:
                with open(tile_manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        async def _emit_progress(event: str, extra: Optional[Dict[str, Any]] = None) -> None:
            if not progress_cb:
                return
            payload = dict(progress)
            payload["event"] = event
            if extra:
                payload.update(extra)
            progress_cb(payload)

        async def _finalize_tile(
            tref: TileRef,
            gate_ok: bool,
            metrics: Dict[str, float],
            class_counts: Dict[str, int],
            dets: List[DetBox],
            cache_hit: Optional[bool],
            status: str,
            png_path: Optional[str],
            sem_tile: Optional[TileSemantics],
            tile_hash: Optional[str],
            prompt_id: Optional[str],
            geo: Optional[List[GeoObject]] = None,
        ) -> None:
            if geo:
                async with objs_lock:
                    objs.extend(geo)
            entry = {
                "image_id": tref.image_id,
                "tile_id": tref.tile_id,
                "row": tref.row,
                "col": tref.col,
                "window": list(tref.window),
                "bounds": list(tref.bounds),
                "transform": list(tref.transform),
                "crs_wkt": tref.crs_wkt,
                "gate_ok": gate_ok,
                "gate_metrics": metrics,
                "class_counts": class_counts,
                "dets": [d.model_dump() for d in dets] if dets else [],
                "cache_hit": cache_hit,
                "status": status,
                "png_path": _relpath(png_path),
                "semantics": sem_tile.model_dump() if sem_tile else None,
                "tile_hash": tile_hash,
                "prompt_id": prompt_id,
                "detector_model": detector_model,
                "vlm_model": vlm_model,
                "vlm_prompt_id": vlm_prompt_id,
                "gate_name": gate_name,
            }
            await _write_manifest(entry)
            async with progress_lock:
                progress["processed_tiles"] += 1
                if gate_ok:
                    progress["kept_tiles"] += 1
                progress["dets_total"] += len(dets)
                if cache_enabled and cache_hit is not None:
                    if cache_hit:
                        progress["cache_hits"] += 1
                    else:
                        progress["cache_misses"] += 1
            await _emit_progress("tile", {"tile_id": tref.tile_id, "status": status})

        async def _detector_worker():
            while True:
                tref = await tile_queue.get()
                if tref is None:
                    tile_queue.task_done()
                    break
                gate_ok = False
                dets: List[DetBox] = []
                class_counts: Dict[str, int] = {}
                metrics: Dict[str, float] = {}
                cache_hit = None
                sem_tile: Optional[TileSemantics] = None
                png_path = None
                status = "error"
                tile_hash = None
                key = ""
                try:
                    with tracer.start_as_current_span("tile.handle") as tspan:
                        tspan.set_attribute("tile.id", tref.tile_id)
                        tspan.set_attribute("tile.window", str(tref.window))
                        t0 = time.perf_counter()
                        logger.info("Tile start image=%s tile=%s window=%s", tref.image_id, tref.tile_id, tref.window)
                        rgb = tiler.read_tile_rgb(path, tref)
                        tile_hash = _hash_rgb(rgb)
                        logger.info("Tile read tile=%s shape=%s", tref.tile_id, rgb.shape)
                        if offload_cpu:
                            ok, metrics = await asyncio.to_thread(gate.keep, rgb)
                        else:
                            ok, metrics = gate.keep(rgb)
                        logger.info("Tile gate tile=%s ok=%s metrics=%s", tref.tile_id, ok, metrics)
                        if not ok:
                            tspan.set_attribute("tile.skipped", True)
                            logger.info("Tile skipped by gate tile=%s", tref.tile_id)
                            status = "skipped_gate"
                            await _finalize_tile(
                                tref, gate_ok, metrics, class_counts, dets, cache_hit, status,
                                png_path, sem_tile, tile_hash, None, None
                            )
                            continue
                        gate_ok = True
                        if tile_img_dir:
                            png_path = os.path.join(tile_img_dir, f"{tref.tile_id}.png")
                            _save_tile_png(rgb, png_path)
                        if offload_cpu:
                            dets, class_counts = await asyncio.to_thread(detector.detect, rgb)
                        else:
                            dets, class_counts = detector.detect(rgb)
                        logger.info("Tile detect tile=%s dets=%s class_counts=%s", tref.tile_id, len(dets), class_counts)
                        if not dets and not allow_empty_vlm:
                            tspan.set_attribute("tile.skipped", True)
                            logger.info("Tile skipped no_dets tile=%s", tref.tile_id)
                            status = "no_dets"
                            await _finalize_tile(
                                tref, gate_ok, metrics, class_counts, dets, cache_hit, status,
                                png_path, sem_tile, tile_hash, None, None
                            )
                            continue
                        dets_hash = _hash_dets(dets)
                        cached = None
                        if cache_enabled:
                            model_id = getattr(vlm, "cache_id", vlm.model)
                            key = cache.key(tref.image_id, tref.tile_id, rgb, model_id=model_id, dets_hash=dets_hash)
                            cached = cache.get(key)
                        if cached is not None:
                            cache_hit = True
                            logger.info("Tile cache hit tile=%s key=%s", tref.tile_id, key)
                            sem_tile = TileSemantics.model_validate(cached["semantics"])
                            _normalize_semantics(sem_tile)
                            geo = aggregator.build_geo_objects(tref, dets, sem_tile) if dets else []
                            logger.info("Tile geo objects tile=%s count=%s", tref.tile_id, len(geo))
                            logger.info("Tile done tile=%s elapsed_s=%.3f", tref.tile_id, time.perf_counter() - t0)
                            status = "ok"
                            await _finalize_tile(
                                tref, gate_ok, metrics, class_counts, dets, cache_hit, status,
                                png_path, sem_tile, tile_hash, vlm_prompt_id, geo
                            )
                            continue
                        cache_hit = False if cache_enabled else None
                        logger.info("Tile cache miss tile=%s key=%s", tref.tile_id, key)
                        await vlm_queue.put({
                            "tref": tref,
                            "rgb": rgb,
                            "dets": dets,
                            "class_counts": class_counts,
                            "metrics": metrics,
                            "gate_ok": gate_ok,
                            "cache_hit": cache_hit,
                            "png_path": png_path,
                            "tile_hash": tile_hash,
                            "key": key,
                        })
                except Exception:
                    logger.exception("Tile failed tile=%s", tref.tile_id)
                    status = "error"
                    await _finalize_tile(
                        tref, gate_ok, metrics, class_counts, dets, cache_hit, status,
                        png_path, sem_tile, tile_hash, None, None
                    )
                finally:
                    tile_queue.task_done()

        async def _vlm_worker():
            while True:
                item = await vlm_queue.get()
                if item is None:
                    vlm_queue.task_done()
                    break
                tref = item["tref"]
                rgb = item["rgb"]
                dets = item["dets"]
                class_counts = item["class_counts"]
                metrics = item["metrics"]
                gate_ok = item["gate_ok"]
                cache_hit = item["cache_hit"]
                png_path = item["png_path"]
                tile_hash = item["tile_hash"]
                key = item["key"]
                sem_tile = None
                status = "error"
                geo = []
                try:
                    sem_tile = await vlm.annotate(tref.tile_id, rgb, dets, class_counts, image_path=png_path)
                    _normalize_semantics(sem_tile)
                    if cache_enabled:
                        cache.put(key, {"tref": asdict(tref), "metrics": metrics, "dets": [d.model_dump() for d in dets],
                                        "class_counts": class_counts, "semantics": sem_tile.model_dump()})
                    geo = aggregator.build_geo_objects(tref, dets, sem_tile) if dets else []
                    logger.info("Tile geo objects tile=%s count=%s", tref.tile_id, len(geo))
                    status = "ok"
                except Exception:
                    logger.exception("VLM failed tile=%s", tref.tile_id)
                    status = "error"
                finally:
                    await _finalize_tile(
                        tref, gate_ok, metrics, class_counts, dets, cache_hit, status,
                        png_path, sem_tile, tile_hash, vlm_prompt_id, geo
                    )
                    vlm_queue.task_done()

        async def _producer():
            for t in tile_refs:
                await tile_queue.put(t)
            for _ in range(detector_workers):
                await tile_queue.put(None)

        detector_tasks = [asyncio.create_task(_detector_worker()) for _ in range(detector_workers)]
        vlm_tasks = [asyncio.create_task(_vlm_worker()) for _ in range(vlm_workers)]
        await _producer()
        await tile_queue.join()
        await asyncio.gather(*detector_tasks)
        for _ in range(vlm_workers):
            await vlm_queue.put(None)
        await vlm_queue.join()
        await asyncio.gather(*vlm_tasks)
        objs = aggregator.geo_nms_global(objs)
        progress["objects_total"] = len(objs)
        await _emit_progress("done", {"objects_total": len(objs)})

        os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o.model_dump(), ensure_ascii=False) + "\n")
        logger.info("Process image write objects=%s count=%s", out_jsonl, len(objs))

        crs_wkt = tile_refs[0].crs_wkt if tile_refs else ""
        return image_id, crs_wkt, objs


def load_geoobjects_jsonl(p: str):
    with tracer.start_as_current_span("pipeline.load_geoobjects") as span:
        span.set_attribute("path", p)
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(GeoObject.model_validate(json.loads(line)))
        logger.info("Load geoobjects path=%s count=%s", p, len(out))
        return out
