from typing import Any, Dict, List, Tuple
import logging
from shapely.geometry import Polygon, shape
from .types import DetBox, GeoObject, TileRef, TileSemantics
from .telemetry import get_tracer
from .normalize import normalize_label, normalize_roof_color


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def px_bbox_to_geo_poly(bbox_px: Tuple[int, int, int, int], tref: TileRef) -> Polygon:
    x1, y1, x2, y2 = bbox_px
    a, b, c, d, e, f = tref.transform

    def pix_to_geo(x: float, y: float):
        return (a * x + b * y + c, d * x + e * y + f)

    p1 = pix_to_geo(x1, y1)
    p2 = pix_to_geo(x2, y1)
    p3 = pix_to_geo(x2, y2)
    p4 = pix_to_geo(x1, y2)
    return Polygon([p1, p2, p3, p4, p1])


def poly_to_geojson(poly: Polygon) -> Dict[str, Any]:
    return {"type": "Polygon", "coordinates": [[list(p) for p in poly.exterior.coords]]}


def iou_poly(a, b) -> float:
    if not a.is_valid or not b.is_valid:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    uni = a.union(b).area
    return float(inter / uni) if uni > 0 else 0.0


class GeoAggregator:
    def __init__(self, nms_iou: float = 0.6):
        self.nms_iou = nms_iou

    def build_geo_objects(self, tref: TileRef, dets: List[DetBox], sem: TileSemantics) -> List[GeoObject]:
        with tracer.start_as_current_span("geo.build_objects") as span:
            span.set_attribute("tile.id", tref.tile_id)
            ann_by_id = {a.det_id: a for a in sem.annotations}
            objs = []
            for d in dets:
                a = ann_by_id.get(d.det_id)
                label = a.label if a else (d.cls or "object")
                label = normalize_label(label)
                attrs = dict(a.attributes) if a else {}
                if "roof_color" in attrs:
                    norm_color = normalize_roof_color(attrs.get("roof_color"))
                    if norm_color:
                        attrs["roof_color"] = norm_color
                poly = px_bbox_to_geo_poly(d.bbox_px, tref)
                obj_id = f"{tref.tile_id}_d{d.det_id}"
                objs.append(GeoObject(
                    image_id=tref.image_id,
                    obj_id=obj_id,
                    label=label,
                    confidence=float(d.score),
                    geometry=poly_to_geojson(poly),
                    attributes=attrs,
                    tile_id=tref.tile_id,
                ))
            logger.info("Geo objects pre-nms tile=%s count=%s", tref.tile_id, len(objs))
            out = self.geo_nms(objs)
            logger.info("Geo objects post-nms tile=%s count=%s", tref.tile_id, len(out))
            return out

    def geo_nms(self, objs: List[GeoObject]) -> List[GeoObject]:
        with tracer.start_as_current_span("geo.nms") as span:
            if not objs:
                return objs
            polys = []
            for o in objs:
                try:
                    g = shape(o.geometry)
                    if not g.is_valid:
                        g = g.buffer(0)
                    if g.is_empty:
                        g = None
                except Exception:
                    g = None
                polys.append(g)
            order = sorted(range(len(objs)), key=lambda i: objs[i].confidence, reverse=True)
            keep = []
            suppressed = set()
            for pi, i in enumerate(order):
                if i in suppressed:
                    continue
                keep.append(i)
                for pj in range(pi + 1, len(order)):
                    j = order[pj]
                    if j in suppressed:
                        continue
                    if polys[i] is None or polys[j] is None:
                        continue
                    if polys[i].geom_type not in ("Polygon", "MultiPolygon"):
                        continue
                    if polys[j].geom_type not in ("Polygon", "MultiPolygon"):
                        continue
                    if objs[i].label != objs[j].label:
                        continue
                    if iou_poly(polys[i], polys[j]) >= self.nms_iou:
                        suppressed.add(j)
            span.set_attribute("nms.kept", len(keep))
            return [objs[i] for i in keep]

    def geo_nms_global(self, objs: List[GeoObject]) -> List[GeoObject]:
        return self.geo_nms(objs)
