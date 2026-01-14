import os, json, pickle, logging, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import httpx
from shapely.geometry import shape
from shapely.ops import unary_union
try:
    import faiss
except Exception:
    faiss = None
from .types import GeoObject, ChangeReport, ChangeEvent
from .normalize import normalize_label, normalize_roof_color, to_uk_color, to_uk_label
from .geo_utils import require_projected_crs
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


@dataclass
class SearchDoc:
    doc_id: str
    kind: str
    image_id: str
    geometry: Dict[str, Any]
    text: str
    meta: Dict[str, Any]


@dataclass
class SearchHit:
    score: float
    doc_id: str
    kind: str
    image_id: str
    centroid: Tuple[float, float]
    geometry: Dict[str, Any]
    meta: Dict[str, Any]
    preview: str


def _centroid_xy(geom: Dict[str, Any]) -> Tuple[float, float]:
    g = shape(geom)
    c = g.centroid
    return float(c.x), float(c.y)


def _short(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _bounds_to_geojson(bounds: List[float]) -> Dict[str, Any]:
    minx, miny, maxx, maxy = bounds
    ring = [
        [float(minx), float(miny)],
        [float(maxx), float(miny)],
        [float(maxx), float(maxy)],
        [float(minx), float(maxy)],
        [float(minx), float(miny)],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def build_object_text(o: GeoObject, scene_hint: Optional[str] = None) -> str:
    a = o.attributes or {}
    tech_parts = []
    tech_parts.append(f"object_label: {o.label}")
    tech_parts.append(f"confidence: {o.confidence:.2f}")
    if scene_hint:
        tech_parts.append(f"scene: {scene_hint}")
    key_fields = ("roof_color", "surface_type", "material", "construction_state", "vehicle_type", "color", "notes")
    for k in key_fields:
        if k in a and a[k] is not None and str(a[k]).strip():
            tech_parts.append(f"{k}: {a[k]}")
    extras = {k: v for k, v in a.items() if k not in set(key_fields)}
    if extras:
        tech_parts.append("attributes_extra: " + json.dumps(extras, ensure_ascii=False, sort_keys=True))

    uk_parts = []
    label_uk = to_uk_label(o.label)
    if label_uk:
        uk_parts.append(f"обʼєкт: {label_uk}")
    if scene_hint:
        uk_parts.append(f"сцена: {scene_hint}")
    roof_val = a.get("roof_color")
    roof_uk = to_uk_color(roof_val)
    if roof_uk or (roof_val is not None and str(roof_val).strip()):
        uk_parts.append(f"дах_колір: {roof_uk or str(roof_val)}")
    surface_val = a.get("surface_type")
    if surface_val is not None and str(surface_val).strip():
        uk_parts.append(f"тип_поверхні: {surface_val}")
    material_val = a.get("material")
    if material_val is not None and str(material_val).strip():
        uk_parts.append(f"матеріал: {material_val}")
    construction_val = a.get("construction_state")
    if construction_val is not None and str(construction_val).strip():
        uk_parts.append(f"стан_будівництва: {construction_val}")
    vehicle_val = a.get("vehicle_type")
    if vehicle_val is not None and str(vehicle_val).strip():
        uk_parts.append(f"тип_транспорту: {vehicle_val}")
    notes_val = a.get("notes")
    if notes_val is not None and str(notes_val).strip():
        uk_parts.append(f"примітки: {notes_val}")

    uk_desc = []
    if label_uk:
        uk_desc.append(label_uk)
    if roof_uk:
        uk_desc.append(f"{roof_uk} дах")
    if surface_val is not None and str(surface_val).strip():
        uk_desc.append(f"поверхня: {surface_val}")
    if material_val is not None and str(material_val).strip():
        uk_desc.append(f"матеріал: {material_val}")
    if construction_val is not None and str(construction_val).strip():
        uk_desc.append(f"стан: {construction_val}")
    if vehicle_val is not None and str(vehicle_val).strip():
        uk_desc.append(f"тип транспорту: {vehicle_val}")
    if notes_val is not None and str(notes_val).strip():
        uk_desc.append(f"примітки: {notes_val}")
    if uk_desc:
        uk_parts.append("опис_укр: " + "; ".join(uk_desc))

    mode = os.environ.get("TEXT_MODE", "bilingual").strip().lower()
    if mode == "tech":
        return "\n".join(tech_parts)
    if mode == "uk_only":
        return "\n".join(uk_parts or tech_parts)
    return "\n".join(tech_parts + uk_parts)


def build_change_text(ev: ChangeEvent) -> str:
    parts = []
    parts.append(f"change_type: {ev.change_type}")
    parts.append(f"label: {ev.label}")
    if ev.before:
        parts.append("before: " + json.dumps(ev.before, ensure_ascii=False))
    if ev.after:
        parts.append("after: " + json.dumps(ev.after, ensure_ascii=False))
    if ev.score is not None:
        parts.append(f"match_score: {ev.score:.3f}")
    return "\n".join(parts)


def build_tile_text(tile: Dict[str, Any]) -> str:
    sem = tile.get("semantics") or {}
    scene = (sem.get("scene") or "").strip()
    tile_summary = sem.get("tile_summary") or {}
    class_counts = tile.get("class_counts") or {}
    anns = sem.get("annotations") or []
    labels = [normalize_label(a.get("label")) for a in anns if a.get("label")]
    roof_colors = [
        normalize_roof_color((a.get("attributes") or {}).get("roof_color"))
        for a in anns
        if (a.get("attributes") or {}).get("roof_color")
    ]
    labels = sorted({l for l in labels if l})
    roof_colors = sorted({c for c in roof_colors if c})
    notes = [a.get("notes") for a in anns if a.get("notes")]

    tech_parts = []
    if scene:
        tech_parts.append(f"scene: {scene}")
    if tile_summary:
        tech_parts.append("tile_summary: " + json.dumps(tile_summary, ensure_ascii=False, sort_keys=True))
    if class_counts:
        tech_parts.append("class_counts: " + json.dumps(class_counts, ensure_ascii=False, sort_keys=True))
    if labels:
        tech_parts.append("labels: " + ", ".join(labels))
    if roof_colors:
        tech_parts.append("roof_colors: " + ", ".join(roof_colors))
    if notes:
        tech_parts.append("notes: " + "; ".join([str(n) for n in notes if str(n).strip()]))

    uk_parts = []
    if scene:
        uk_parts.append(f"сцена: {scene}")
    if labels:
        uk_labels = [to_uk_label(l) or l for l in labels]
        uk_parts.append("обʼєкти: " + ", ".join(uk_labels))
    if roof_colors:
        uk_colors = [to_uk_color(c) or c for c in roof_colors]
        uk_parts.append("кольори_дахів: " + ", ".join(uk_colors))
    if notes:
        uk_parts.append("примітки: " + "; ".join([str(n) for n in notes if str(n).strip()]))

    mode = os.environ.get("TEXT_MODE", "bilingual").strip().lower()
    if mode == "tech":
        return "\n".join(tech_parts)
    if mode == "uk_only":
        return "\n".join(uk_parts or tech_parts)
    return "\n".join(tech_parts + uk_parts)


def load_geoobjects_jsonl(path: str) -> List[GeoObject]:
    with tracer.start_as_current_span("index.load_geoobjects") as span:
        span.set_attribute("path", path)
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(GeoObject.model_validate(json.loads(line)))
        logger.info("Loaded geoobjects path=%s count=%s", path, len(out))
        return out


def load_tile_manifest_jsonl(path: str) -> List[Dict[str, Any]]:
    with tracer.start_as_current_span("index.load_tile_manifest") as span:
        span.set_attribute("path", path)
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        logger.info("Loaded tile manifest path=%s count=%s", path, len(out))
        return out


def load_change_report(path: str) -> ChangeReport:
    with tracer.start_as_current_span("index.load_change_report") as span:
        span.set_attribute("path", path)
        with open(path, "r", encoding="utf-8") as f:
            rep = ChangeReport.model_validate(json.load(f))
        logger.info("Loaded change report path=%s changes=%s", path, len(rep.changes))
        return rep


def load_rail_geometry(rail_geojson_path: str):
    with tracer.start_as_current_span("index.load_rail_geometry") as span:
        span.set_attribute("path", rail_geojson_path)
        with open(rail_geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features", [])
        geoms = [shape(ft["geometry"]) for ft in feats if ft.get("geometry")]
        if not geoms:
            logger.info("Rail geometry empty path=%s", rail_geojson_path)
            return None
        logger.info("Rail geometry loaded path=%s count=%s", rail_geojson_path, len(geoms))
        return unary_union(geoms)


def compute_rail_dist_m(geom: Dict[str, Any], rail_union) -> Optional[float]:
    if rail_union is None:
        return None
    g = shape(geom)
    return float(g.distance(rail_union))


class EmbeddingClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model
        self.client = httpx.Client(timeout=timeout_s)
        logger.info("Embedding client init base_url=%s model=%s timeout_s=%s", self.base_url, self.model, timeout_s)

    def embed(self, texts: List[str]) -> np.ndarray:
        with tracer.start_as_current_span("embeddings.embed") as span:
            span.set_attribute("batch.size", len(texts))
            t0 = time.perf_counter()
            url = f"{self.base_url}/embeddings"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {"model": self.model, "input": texts}
            r = self.client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()["data"]
            if data and all("index" in d for d in data):
                data = sorted(data, key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in data]
            if len(vecs) != len(texts):
                raise ValueError(f"Embedding count mismatch got={len(vecs)} expected={len(texts)}")
            arr = np.array(vecs, dtype=np.float32)
            n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            logger.info("Embedding batch size=%s elapsed_s=%.3f", len(texts), time.perf_counter() - t0)
            return arr / n

    def close(self):
        self.client.close()


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.docs = []
        self._emb = None
        self._faiss = None

    def add(self, docs: List[SearchDoc], emb: np.ndarray):
        with tracer.start_as_current_span("index.add") as span:
            span.set_attribute("docs.count", len(docs))
            self.docs.extend(docs)
            self._emb = emb if self._emb is None else np.vstack([self._emb, emb])
            if faiss is not None:
                if self._faiss is None:
                    self._faiss = faiss.IndexFlatIP(self.dim)
                    self._faiss.add(self._emb)
                else:
                    self._faiss.add(emb)
            logger.info("Index add docs=%s total=%s", len(docs), len(self.docs))

    def search(self, q: np.ndarray, top_k: int = 20):
        with tracer.start_as_current_span("index.search") as span:
            span.set_attribute("top_k", top_k)
            q = q.astype(np.float32)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            if self._emb is None or len(self.docs) == 0:
                logger.info("Index search empty")
                return []
            if self._faiss is not None:
                D, I = self._faiss.search(q, top_k)
                return [(float(D[0, i]), int(I[0, i])) for i in range(I.shape[1]) if int(I[0, i]) >= 0]
            sims = (self._emb @ q[0]).reshape(-1)
            idx = np.argsort(-sims)[:top_k]
            return [(float(sims[i]), int(i)) for i in idx]

    def save(self, dir_path: str):
        with tracer.start_as_current_span("index.save") as span:
            span.set_attribute("dir", dir_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, "docs.pkl"), "wb") as f:
                pickle.dump(self.docs, f)
            np.save(os.path.join(dir_path, "emb.npy"), self._emb)
            if self._faiss is not None:
                faiss.write_index(self._faiss, os.path.join(dir_path, "faiss.index"))
            logger.info("Index saved dir=%s docs=%s", dir_path, len(self.docs))

    @classmethod
    def load(cls, dir_path: str):
        with tracer.start_as_current_span("index.load") as span:
            span.set_attribute("dir", dir_path)
            with open(os.path.join(dir_path, "docs.pkl"), "rb") as f:
                docs = pickle.load(f)
            emb = np.load(os.path.join(dir_path, "emb.npy")).astype(np.float32)
            vi = cls(int(emb.shape[1]))
            vi.docs = docs
            vi._emb = emb
            fi = os.path.join(dir_path, "faiss.index")
            if faiss is not None and os.path.exists(fi):
                vi._faiss = faiss.read_index(fi)
            logger.info("Index loaded dir=%s docs=%s", dir_path, len(vi.docs))
            return vi


class SemanticSearcher:
    def __init__(self, index: VectorIndex, embedder: EmbeddingClient, default_top_k: int = 20):
        self.index = index
        self.embedder = embedder
        self.default_top_k = default_top_k

    def query(self, text: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        with tracer.start_as_current_span("search.query") as span:
            top_k = top_k or self.default_top_k
            span.set_attribute("top_k", top_k)
            q = self.embedder.embed([text])
            raw = self.index.search(q, top_k=top_k * 5)
            hits = []
            for score, pos in raw:
                d = self.index.docs[pos]
                if filters and not self._passes_filters(d, filters):
                    continue
                x, y = _centroid_xy(d.geometry)
                hits.append(SearchHit(score=score, doc_id=d.doc_id, kind=d.kind, image_id=d.image_id,
                                      centroid=(x, y), geometry=d.geometry, meta=d.meta, preview=_short(d.text)))
                if len(hits) >= top_k:
                    break
            logger.info("Search query done hits=%s", len(hits))
            return hits

    def _passes_filters(self, d: SearchDoc, flt: Dict[str, Any]) -> bool:
        m = d.meta or {}
        for k, v in flt.items():
            if k == "kind":
                if str(d.kind).strip().lower() != str(v).strip().lower():
                    return False
            if k == "roof_color":
                want = normalize_roof_color(v)
                have = m.get("roof_color")
                if isinstance(have, (list, tuple, set)):
                    have_norm = {normalize_roof_color(x) for x in have}
                    if want not in have_norm:
                        return False
                else:
                    if normalize_roof_color(have) != want:
                        return False
            elif k == "roof_color_any":
                vals = v if isinstance(v, (list, tuple, set)) else [v]
                want = {normalize_roof_color(x) for x in vals}
                have = m.get("roof_color")
                if isinstance(have, (list, tuple, set)):
                    have_norm = {normalize_roof_color(x) for x in have}
                    if not (have_norm & want):
                        return False
                else:
                    if normalize_roof_color(have) not in want:
                        return False
            elif k == "label":
                want = normalize_label(v)
                have = m.get("label")
                if isinstance(have, (list, tuple, set)):
                    have_norm = {normalize_label(x) for x in have}
                    if want not in have_norm:
                        return False
                else:
                    if normalize_label(have) != want:
                        return False
            elif k == "rail_dist_max_m":
                rd = m.get("rail_dist_m")
                if rd is None or float(rd) > float(v):
                    return False
            elif k == "change_type":
                if (m.get("change_type") or "").strip().lower() != str(v).strip().lower():
                    return False
        return True


def build_semantic_index(geoobjects_jsonl: str, index_dir: str, embedder: EmbeddingClient,
                         change_report_json: Optional[str] = None, rail_geojson: Optional[str] = None,
                         tile_manifest_jsonl: Optional[str] = None, crs_wkt: Optional[str] = None,
                         batch: int = 128) -> str:
    with tracer.start_as_current_span("index.build") as span:
        span.set_attribute("objects.path", geoobjects_jsonl)
        span.set_attribute("index.dir", index_dir)
        logger.info("Build index start objects=%s changes=%s rail=%s", geoobjects_jsonl, change_report_json, rail_geojson)
        objs = load_geoobjects_jsonl(geoobjects_jsonl)
        rail_union = load_rail_geometry(rail_geojson) if rail_geojson else None
        crs_wkt = crs_wkt or os.environ.get("INDEX_CRS_WKT")
        if rail_union is not None:
            try:
                require_projected_crs(crs_wkt, label="Index")
            except ValueError as exc:
                if _env_flag("STRICT_METRIC_CRS", default=False):
                    raise
                logger.warning("Skipping rail distance due to CRS: %s", exc)
                rail_union = None
        docs = []
        texts = []
        for o in objs:
            txt = build_object_text(o)
            label_norm = normalize_label(o.label)
            roof_norm = normalize_roof_color((o.attributes or {}).get("roof_color"))
            meta = {"label": label_norm, "confidence": o.confidence, "tile_id": o.tile_id,
                    "roof_color": roof_norm}
            rd = compute_rail_dist_m(o.geometry, rail_union)
            if rd is not None:
                meta["rail_dist_m"] = rd
            docs.append(SearchDoc(doc_id=f"obj:{o.obj_id}", kind="object", image_id=o.image_id,
                                  geometry=o.geometry, text=txt, meta=meta))
            texts.append(txt)
        if change_report_json:
            rep = load_change_report(change_report_json)
            for i, ev in enumerate(rep.changes):
                txt = build_change_text(ev)
                meta = {"change_type": ev.change_type, "label": ev.label, "score": ev.score}
                docs.append(SearchDoc(doc_id=f"chg:{i}", kind="change", image_id=f"{rep.image_a}__{rep.image_b}",
                                      geometry=ev.geometry, text=txt, meta=meta))
                texts.append(txt)

        index_tiles = _env_flag("INDEX_TILES", default=False)
        if not tile_manifest_jsonl:
            tile_manifest_jsonl = os.environ.get("TILE_MANIFEST_JSONL")
        if index_tiles and tile_manifest_jsonl and os.path.exists(tile_manifest_jsonl):
            tiles = load_tile_manifest_jsonl(tile_manifest_jsonl)
            for t in tiles:
                tile_id = t.get("tile_id")
                if not tile_id:
                    continue
                if not t.get("semantics"):
                    continue
                bounds = t.get("bounds")
                if not bounds:
                    continue
                geom = _bounds_to_geojson(bounds)
                txt = build_tile_text(t)
                sem = t.get("semantics") or {}
                anns = sem.get("annotations") or []
                labels = [normalize_label(a.get("label")) for a in anns if a.get("label")]
                roof_colors = [
                    normalize_roof_color((a.get("attributes") or {}).get("roof_color"))
                    for a in anns
                    if (a.get("attributes") or {}).get("roof_color")
                ]
                labels = sorted({l for l in labels if l})
                roof_colors = sorted({c for c in roof_colors if c})
                meta = {
                    "tile_id": tile_id,
                    "label": labels,
                    "roof_color": roof_colors,
                    "scene": sem.get("scene"),
                    "png_path": t.get("png_path"),
                }
                rd = compute_rail_dist_m(geom, rail_union)
                if rd is not None:
                    meta["rail_dist_m"] = rd
                docs.append(SearchDoc(doc_id=f"tile:{tile_id}", kind="tile",
                                      image_id=t.get("image_id") or "", geometry=geom, text=txt, meta=meta))
                texts.append(txt)

        idx = None
        cur = 0
        while cur < len(docs):
            chunk_docs = docs[cur:cur + batch]
            chunk_txt = texts[cur:cur + batch]
            emb = embedder.embed(chunk_txt)
            if idx is None:
                idx = VectorIndex(int(emb.shape[1]))
            idx.add(chunk_docs, emb)
            logger.info("Index batch embedded docs=%s total=%s", len(chunk_docs), min(cur + batch, len(docs)))
            cur += batch
        idx.save(index_dir)
        meta_path = os.path.join(index_dir, "index_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "geoobjects_jsonl": geoobjects_jsonl,
                "change_report_json": change_report_json,
                "tile_manifest_jsonl": tile_manifest_jsonl,
                "crs_wkt": crs_wkt,
            }, f, ensure_ascii=False)
        logger.info("Build index done dir=%s docs=%s", index_dir, len(docs))
        return index_dir
