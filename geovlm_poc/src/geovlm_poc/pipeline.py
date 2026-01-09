import os, json, asyncio, hashlib, logging, time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from .types import GeoObject, TileRef, TileSemantics
from .tiling import ImageTiler
from .gates import TileGate
from .detector import Detector
from .vlm import VLMAnnotator
from .geo import GeoAggregator
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class TileCache:
    def __init__(self, dir_path: str):
        self.dir = dir_path
        os.makedirs(self.dir, exist_ok=True)

    def key(self, image_id: str, tile_id: str, rgb: np.ndarray, model_id: str) -> str:
        h = hashlib.sha256()
        h.update(model_id.encode("utf-8"))
        h.update(image_id.encode("utf-8"))
        h.update(tile_id.encode("utf-8"))
        h.update(rgb.tobytes())
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


async def process_image(path: str, out_jsonl: str,
                        tiler: ImageTiler, gate: TileGate,
                        detector: Detector, vlm: VLMAnnotator,
                        cache: TileCache, aggregator: GeoAggregator,
                        max_tiles: Optional[int] = None) -> Tuple[str, str, List[GeoObject]]:
    with tracer.start_as_current_span("process_image") as span:
        span.set_attribute("image.path", path)
        span.set_attribute("output.jsonl", out_jsonl)
        image_id = os.path.basename(path)
        tile_refs = []
        for i, t in enumerate(tiler.iter_tiles(path, image_id=image_id)):
            tile_refs.append(t)
            if max_tiles is not None and i + 1 >= max_tiles:
                break
        logger.info("Process image start image=%s tiles=%s out=%s", image_id, len(tile_refs), out_jsonl)

        objs: List[GeoObject] = []

        async def handle(tref: TileRef):
            with tracer.start_as_current_span("tile.handle") as tspan:
                tspan.set_attribute("tile.id", tref.tile_id)
                tspan.set_attribute("tile.window", str(tref.window))
                t0 = time.perf_counter()
                logger.info("Tile start image=%s tile=%s window=%s", tref.image_id, tref.tile_id, tref.window)
                rgb = tiler.read_tile_rgb(path, tref)
                logger.info("Tile read tile=%s shape=%s", tref.tile_id, rgb.shape)
                ok, metrics = gate.keep(rgb)
                logger.info("Tile gate tile=%s ok=%s metrics=%s", tref.tile_id, ok, metrics)
                if not ok:
                    tspan.set_attribute("tile.skipped", True)
                    logger.info("Tile skipped by gate tile=%s", tref.tile_id)
                    return
                dets, class_counts = detector.detect(rgb)
                logger.info("Tile detect tile=%s dets=%s class_counts=%s", tref.tile_id, len(dets), class_counts)
                if not dets:
                    tspan.set_attribute("tile.skipped", True)
                    logger.info("Tile skipped no_dets tile=%s", tref.tile_id)
                    return
                key = cache.key(tref.image_id, tref.tile_id, rgb, model_id=vlm.model)
                cached = cache.get(key)
                if cached is not None:
                    logger.info("Tile cache hit tile=%s key=%s", tref.tile_id, key)
                    sem = TileSemantics.model_validate(cached["semantics"])
                else:
                    logger.info("Tile cache miss tile=%s key=%s", tref.tile_id, key)
                    sem = await vlm.annotate(tref.tile_id, rgb, dets, class_counts)
                    cache.put(key, {"tref": asdict(tref), "metrics": metrics, "dets": [d.model_dump() for d in dets],
                                    "class_counts": class_counts, "semantics": sem.model_dump()})
                geo = aggregator.build_geo_objects(tref, dets, sem)
                logger.info("Tile geo objects tile=%s count=%s", tref.tile_id, len(geo))
                objs.extend(geo)
                logger.info("Tile done tile=%s elapsed_s=%.3f", tref.tile_id, time.perf_counter() - t0)

        tasks = [asyncio.create_task(handle(t)) for t in tile_refs]
        await asyncio.gather(*tasks)

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
