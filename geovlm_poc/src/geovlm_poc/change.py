import math, time, logging
from typing import Dict, List, Tuple
import numpy as np
import cv2
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Polygon, Point
from shapely.affinity import translate as shp_translate
try:
    from shapely.strtree import STRtree
except Exception:
    STRtree = None
from .types import ChangeEvent, ChangeReport, GeoObject
from .geo import iou_poly, poly_to_geojson
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def _poly(o: GeoObject) -> Polygon:
    return Polygon(o.geometry["coordinates"][0])


def _poly_area(o: GeoObject) -> float:
    return float(_poly(o).area)


def _poly_orientation_deg(o: GeoObject) -> float:
    p = _poly(o)
    mrr = p.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 2:
        return 0.0
    (x1, y1), (x2, y2) = coords[0], coords[1]
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    if ang > 90.0:
        ang -= 90.0
    return float(ang)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 90.0
    return float(min(d, 90.0 - d))


def _centroid(o: GeoObject) -> Tuple[float, float]:
    c = _poly(o).centroid
    return float(c.x), float(c.y)


def _vehicle_group_sizes(objs: List[GeoObject], radius_m: float, vehicle_labels=("car", "vehicle", "truck", "bus")) -> Dict[str, int]:
    veh = []
    pts = []
    for o in objs:
        if o.label.lower() in vehicle_labels:
            x, y = _centroid(o)
            veh.append(o)
            pts.append(Point(x, y))
    if not pts:
        return {}
    if STRtree is None:
        out = {}
        for o, p in zip(veh, pts):
            cnt = 0
            for q in pts:
                if p.distance(q) <= radius_m:
                    cnt += 1
            out[o.obj_id] = cnt
        return out
    tree = STRtree(pts)
    out = {}
    for o, p in zip(veh, pts):
        cand = tree.query(p.buffer(radius_m))
        cnt = 0
        for q in cand:
            if p.distance(q) <= radius_m:
                cnt += 1
        out[o.obj_id] = cnt
    return out


class Coregistrator:
    def __init__(self, max_dim: int = 1024, band: int = 1):
        self.max_dim = max_dim
        self.band = band

    def estimate_shift(self, image_a_path: str, image_b_path: str) -> Tuple[float, float]:
        with tracer.start_as_current_span("coreg.estimate") as span:
            span.set_attribute("input.a", image_a_path)
            span.set_attribute("input.b", image_b_path)
            logger.info("Coregistration estimate start a=%s b=%s", image_a_path, image_b_path)
            with rasterio.open(image_a_path) as da, rasterio.open(image_b_path) as db:
                if da.crs and db.crs and da.crs != db.crs:
                    logger.warning("Coregistration CRS mismatch a=%s b=%s; returning zero shift", da.crs, db.crs)
                    return (0.0, 0.0)
                out_w, out_h = self._common_out_shape(da.width, da.height, db.width, db.height)
                img_a = self._read_gray(da, out_h, out_w)
                img_b = self._read_gray(db, out_h, out_w)
                shift_px, response = cv2.phaseCorrelate(img_a, img_b)
                dx_px, dy_px = float(shift_px[0]), float(shift_px[1])
                scale_x = da.width / float(out_w)
                scale_y = da.height / float(out_h)
                dx_full = dx_px * scale_x
                dy_full = dy_px * scale_y
                a, b, c, d, e, f = da.transform
                dx_geo = a * dx_full + b * dy_full
                dy_geo = d * dx_full + e * dy_full
                span.set_attribute("shift.px.dx", dx_px)
                span.set_attribute("shift.px.dy", dy_px)
                span.set_attribute("shift.geo.dx", dx_geo)
                span.set_attribute("shift.geo.dy", dy_geo)
                span.set_attribute("phase.response", float(response))
                logger.info("Coregistration shift px=(%.2f, %.2f) response=%.3f", dx_px, dy_px, response)
                logger.info("Coregistration shift geo=(%.3f, %.3f)", dx_geo, dy_geo)
                return (dx_geo, dy_geo)

    def _common_out_shape(self, w_a: int, h_a: int, w_b: int, h_b: int) -> Tuple[int, int]:
        max_dim = max(8, int(self.max_dim))
        def scale_for(w: int, h: int) -> float:
            m = max(w, h)
            return m / max_dim if m > max_dim else 1.0
        scale_a = scale_for(w_a, h_a)
        scale_b = scale_for(w_b, h_b)
        out_w_a = max(8, int(round(w_a / scale_a)))
        out_h_a = max(8, int(round(h_a / scale_a)))
        out_w_b = max(8, int(round(w_b / scale_b)))
        out_h_b = max(8, int(round(h_b / scale_b)))
        out_w = min(out_w_a, out_w_b)
        out_h = min(out_h_a, out_h_b)
        return out_w, out_h

    def _read_gray(self, ds: rasterio.io.DatasetReader, out_h: int, out_w: int) -> np.ndarray:
        arr = ds.read(self.band, out_shape=(out_h, out_w), masked=True, resampling=Resampling.bilinear)
        if np.ma.is_masked(arr):
            arr = np.ma.filled(arr, 0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.dtype != np.uint8:
            arr = self._to_uint8(arr)
        return arr.astype(np.float32)

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        a = arr.astype(np.float32)
        lo = np.nanpercentile(a, 2)
        hi = np.nanpercentile(a, 98)
        if hi <= lo:
            return np.clip(a, 0, 255).astype(np.uint8)
        a = (a - lo) / (hi - lo)
        a = np.clip(a, 0, 1)
        return (a * 255.0).astype(np.uint8)


class ChangeDetector:
    def __init__(self, iou_match: float = 0.35, buffer_tol: float = 4.0, vehicle_cluster_radius_m: float = 25.0):
        self.iou_match = iou_match
        self.buffer_tol = buffer_tol
        self.vehicle_cluster_radius_m = vehicle_cluster_radius_m
        self._vehA = {}
        self._vehB = {}

    def diff(self, image_a: str, image_b: str, crs_wkt: str,
             a_objs: List[GeoObject], b_objs: List[GeoObject],
             b_shift_xy: Tuple[float, float] = (0.0, 0.0)) -> ChangeReport:
        with tracer.start_as_current_span("change.diff") as span:
            span.set_attribute("image.a", image_a)
            span.set_attribute("image.b", image_b)
            span.set_attribute("objs.a", len(a_objs))
            span.set_attribute("objs.b", len(b_objs))
            logger.info("Change diff start a=%s b=%s a_objs=%s b_objs=%s shift=%s",
                        image_a, image_b, len(a_objs), len(b_objs), b_shift_xy)
            b_objs_shifted = self._shift_b(b_objs, b_shift_xy)
            self._vehA = _vehicle_group_sizes(a_objs, self.vehicle_cluster_radius_m)
            self._vehB = _vehicle_group_sizes(b_objs_shifted, self.vehicle_cluster_radius_m)

            changes = []
            a_polys = [_poly(o) for o in a_objs]
            b_polys = [_poly(o) for o in b_objs_shifted]

            if STRtree is None:
                used = set()
                for i, (oa, pa) in enumerate(zip(a_objs, a_polys)):
                    best = (-1.0, None)
                    for j, (ob, pb) in enumerate(zip(b_objs_shifted, b_polys)):
                        if j in used or oa.label != ob.label:
                            continue
                        s = iou_poly(pa, pb)
                        if s > best[0]:
                            best = (s, j)
                    if best[0] >= self.iou_match:
                        used.add(best[1])
                        ob = b_objs_shifted[best[1]]
                        if self._modified(oa, ob):
                            changes.append(ChangeEvent(change_type="modified", label=ob.label, geometry=ob.geometry,
                                                       before=oa.model_dump(), after=ob.model_dump(), score=best[0]))
                    else:
                        changes.append(ChangeEvent(change_type="removed", label=oa.label, geometry=oa.geometry,
                                                   before=oa.model_dump(), after=None))
                for j, ob in enumerate(b_objs_shifted):
                    if j not in used:
                        changes.append(ChangeEvent(change_type="added", label=ob.label, geometry=ob.geometry,
                                                   before=None, after=ob.model_dump()))
                report = ChangeReport(image_a=image_a, image_b=image_b, crs_wkt=crs_wkt,
                                      generated_at_unix=int(time.time()), changes=changes)
                logger.info("Change diff done total=%s", len(report.changes))
                return report

            tree = STRtree(b_polys)
            b_idx = {id(p): i for i, p in enumerate(b_polys)}
            used_b = set()
            for oa, pa in zip(a_objs, a_polys):
                q = pa.buffer(self.buffer_tol) if self.buffer_tol > 0 else pa
                cand = tree.query(q)
                best = (-1.0, None)
                for pb in cand:
                    j = b_idx.get(id(pb))
                    if j is None or j in used_b:
                        continue
                    ob = b_objs_shifted[j]
                    if oa.label != ob.label:
                        continue
                    s = iou_poly(pa, pb)
                    if s > best[0]:
                        best = (s, j)
                if best[0] >= self.iou_match:
                    used_b.add(best[1])
                    ob = b_objs_shifted[best[1]]
                    if self._modified(oa, ob):
                        changes.append(ChangeEvent(change_type="modified", label=ob.label, geometry=ob.geometry,
                                                   before=oa.model_dump(), after=ob.model_dump(), score=best[0]))
                else:
                    changes.append(ChangeEvent(change_type="removed", label=oa.label, geometry=oa.geometry,
                                               before=oa.model_dump(), after=None))
            for j, ob in enumerate(b_objs_shifted):
                if j not in used_b:
                    changes.append(ChangeEvent(change_type="added", label=ob.label, geometry=ob.geometry,
                                               before=None, after=ob.model_dump()))
            report = ChangeReport(image_a=image_a, image_b=image_b, crs_wkt=crs_wkt,
                                  generated_at_unix=int(time.time()), changes=changes)
            logger.info("Change diff done total=%s", len(report.changes))
            return report

    def _modified(self, a: GeoObject, b: GeoObject) -> bool:
        if abs(a.confidence - b.confidence) > 0.25:
            return True
        aa = _poly_area(a)
        ba = _poly_area(b)
        if aa > 0 and ba > 0:
            r = ba / aa
            if r < 0.75 or r > 1.35:
                return True
        oa = _poly_orientation_deg(a)
        ob = _poly_orientation_deg(b)
        if _angle_diff(oa, ob) > 18.0:
            return True
        ak = a.attributes or {}
        bk = b.attributes or {}
        for k in ("roof_color", "color", "surface_type", "material", "construction_state", "vehicle_type"):
            if k in ak and k in bk and str(ak[k]).lower() != str(bk[k]).lower():
                return True
        la = a.label.lower()
        if la in ("car", "vehicle", "truck", "bus"):
            ga = self._vehA.get(a.obj_id)
            gb = self._vehB.get(b.obj_id)
            if ga is not None and gb is not None and abs(ga - gb) >= 6:
                return True
        return False

    def _shift_b(self, b_objs: List[GeoObject], shift_xy: Tuple[float, float]) -> List[GeoObject]:
        dx, dy = shift_xy
        if dx == 0.0 and dy == 0.0:
            return b_objs
        logger.info("Shift B objects dx=%s dy=%s count=%s", dx, dy, len(b_objs))
        out = []
        for o in b_objs:
            p = _poly(o)
            p2 = shp_translate(p, xoff=dx, yoff=dy)
            o2 = o.model_copy(deep=True)
            o2.geometry = poly_to_geojson(p2)
            out.append(o2)
        return out
