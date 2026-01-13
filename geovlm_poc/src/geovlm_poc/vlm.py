import io, json, base64, asyncio, logging, time
from typing import Any, Dict, List
import numpy as np
import httpx
from PIL import Image
from .types import DetBox, TileSemantics
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class VLMAnnotator:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 90.0, concurrency: int = 6):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout_s)
        self.sem = asyncio.Semaphore(concurrency)
        logger.info("VLM client init base_url=%s model=%s timeout_s=%s concurrency=%s",
                    self.base_url, self.model, timeout_s, concurrency)

    async def close(self):
        await self.client.aclose()

    async def annotate(self, tile_id: str, rgb: np.ndarray, dets: List[DetBox], class_counts: Dict[str, int]) -> TileSemantics:
        with tracer.start_as_current_span("vlm.annotate") as span:
            span.set_attribute("tile.id", tile_id)
            span.set_attribute("detections.count", len(dets))
            t0 = time.perf_counter()
            logger.info("VLM annotate start tile=%s dets=%s", tile_id, len(dets))
            png_b64 = self._to_png_b64(rgb)
            prompt = self._prompt(tile_id, dets, class_counts)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Return ONLY valid JSON. Do not change bbox. Do not invent new det_id. Use det_id mapping."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                    ]},
                ],
                "temperature": 0,
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            url = f"{self.base_url}/chat/completions"
            backoffs = [0.5, 1.5]
            attempts = len(backoffs) + 1
            for attempt in range(attempts):
                try:
                    async with self.sem:
                        r = await self.client.post(url, headers=headers, json=payload)
                    if r.status_code >= 500 or r.status_code == 429:
                        if attempt < attempts - 1:
                            logger.warning("VLM retry status=%s attempt=%s tile=%s", r.status_code, attempt + 1, tile_id)
                            await asyncio.sleep(backoffs[attempt])
                            continue
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    data = self._load_json_strict(content)
                    sem = TileSemantics.model_validate(data)
                    logger.info("VLM annotate done tile=%s annotations=%s elapsed_s=%.3f",
                                tile_id, len(sem.annotations), time.perf_counter() - t0)
                    return sem
                except (json.JSONDecodeError, ValueError, httpx.TimeoutException) as exc:
                    if attempt < attempts - 1:
                        logger.warning("VLM retry parse/timeout attempt=%s tile=%s err=%s", attempt + 1, tile_id, exc)
                        await asyncio.sleep(backoffs[attempt])
                        continue
                    raise
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status in (429,) or (status is not None and status >= 500):
                        if attempt < attempts - 1:
                            logger.warning("VLM retry http status=%s attempt=%s tile=%s", status, attempt + 1, tile_id)
                            await asyncio.sleep(backoffs[attempt])
                            continue
                    raise

    def _prompt(self, tile_id: str, dets: List[DetBox], class_counts: Dict[str, int]) -> str:
        det_list = [{"det_id": d.det_id, "bbox_px": list(d.bbox_px), "det_score": d.score, "det_cls": d.cls} for d in dets]
        schema = {
            "tile_id": tile_id,
            "scene": "string",
            "annotations": [{"det_id": 0, "label": "string", "attributes": {}, "notes": "string|null"}],
        }
        return (
            "You analyze a satellite tile. You are given YOLO detections.\n"
            "Use YOLO class summary as strong context and keep your labels consistent.\n"
            f"YOLO tile class counts:\n{json.dumps(class_counts)}\n"
            f"Detections (det_id mapping, do not change bbox):\n{json.dumps(det_list)}\n"
            f"Return ONLY JSON matching schema:\n{json.dumps(schema)}\n"
            "For each detection, fill label and attributes when visible: roof_color, surface_type, material, construction_state, vehicle_type.\n"
        )

    def _load_json_strict(self, s: str) -> Dict[str, Any]:
        s = (s or "").strip()
        if not s:
            raise ValueError("Empty VLM response")
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            snippet = self._extract_first_json_object(s)
            return json.loads(snippet)
        except Exception:
            logger.debug("VLM JSON parse failed content=%r", s)
            raise

    def _extract_first_json_object(self, s: str) -> str:
        start = s.find("{")
        if start < 0:
            raise ValueError("No JSON object start")
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                continue
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        raise ValueError("No complete JSON object found")

    def _to_png_b64(self, rgb: np.ndarray) -> str:
        im = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
