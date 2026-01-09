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
    meta: Dict[str, Any]
    preview: str


def _centroid_xy(geom: Dict[str, Any]) -> Tuple[float, float]:
    g = shape(geom)
    c = g.centroid
    return float(c.x), float(c.y)


def _short(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def build_object_text(o: GeoObject, scene_hint: Optional[str] = None) -> str:
    a = o.attributes or {}
    parts = []
    parts.append(f"object_label: {o.label}")
    parts.append(f"confidence: {o.confidence:.2f}")
    if scene_hint:
        parts.append(f"scene: {scene_hint}")
    for k in ("roof_color", "surface_type", "material", "construction_state", "vehicle_type", "color", "notes"):
        if k in a and a[k] is not None and str(a[k]).strip():
            parts.append(f"{k}: {a[k]}")
    extras = {k: v for k, v in a.items() if k not in {"roof_color", "surface_type", "material", "construction_state", "vehicle_type", "color", "notes"}}
    if extras:
        parts.append("attributes_extra: " + json.dumps(extras, ensure_ascii=False))
    return "\n".join(parts)


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
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=timeout_s)
        logger.info("Embedding client init base_url=%s model=%s timeout_s=%s", self.base_url, self.model, timeout_s)

    def embed(self, texts: List[str]) -> np.ndarray:
        with tracer.start_as_current_span("embeddings.embed") as span:
            span.set_attribute("batch.size", len(texts))
            t0 = time.perf_counter()
            url = f"{self.base_url}/embeddings"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "input": texts}
            r = self.client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()["data"]
            vecs = [d["embedding"] for d in data]
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
                                      centroid=(x, y), meta=d.meta, preview=_short(d.text)))
                if len(hits) >= top_k:
                    break
            logger.info("Search query done hits=%s", len(hits))
            return hits

    def _passes_filters(self, d: SearchDoc, flt: Dict[str, Any]) -> bool:
        m = d.meta or {}
        for k, v in flt.items():
            if k == "roof_color":
                if (m.get("roof_color") or "").strip().lower() != str(v).strip().lower():
                    return False
            elif k == "label":
                if (m.get("label") or "").strip().lower() != str(v).strip().lower():
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
                         batch: int = 128) -> str:
    with tracer.start_as_current_span("index.build") as span:
        span.set_attribute("objects.path", geoobjects_jsonl)
        span.set_attribute("index.dir", index_dir)
        logger.info("Build index start objects=%s changes=%s rail=%s", geoobjects_jsonl, change_report_json, rail_geojson)
        objs = load_geoobjects_jsonl(geoobjects_jsonl)
        rail_union = load_rail_geometry(rail_geojson) if rail_geojson else None
        docs = []
        texts = []
        for o in objs:
            txt = build_object_text(o)
            meta = {"label": o.label, "confidence": o.confidence, "tile_id": o.tile_id,
                    "roof_color": (o.attributes or {}).get("roof_color")}
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
        logger.info("Build index done dir=%s docs=%s", index_dir, len(docs))
        return index_dir
