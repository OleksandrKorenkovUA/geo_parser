import io, json, base64, asyncio, logging, time, os, random, hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import httpx
from PIL import Image
from .types import DetBox, TileSemantics
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def _env_flag(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    sep = "|" if "|" in value else ","
    items = [p.strip() for p in value.split(sep)]
    return [p for p in items if p]


class VLMAnnotator:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 90.0, concurrency: int = 6):
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = model
        prompt_cfg = self._load_prompt_config()
        self.system_prompt = prompt_cfg["system_prompt"]
        self.user_prompt_template = prompt_cfg["user_prompt_template"]
        self.prompt_id = os.environ.get("VLM_PROMPT_ID", "vlm_prompt_v2")
        self.backend = os.environ.get("VLM_BACKEND", "openai").strip().lower()
        self.mode = os.environ.get("VLM_MODE", "free_text").strip().lower()
        self.choice_list = _parse_list(os.environ.get("VLM_CHOICE_LIST", ""))
        self.json_schema = None
        self.max_image_side = int(os.environ.get("VLM_MAX_IMAGE_SIDE", "0") or 0)
        self.image_mode = os.environ.get("VLM_IMAGE_MODE", "base64").strip().lower()
        self.image_url_prefix = os.environ.get("VLM_IMAGE_URL_PREFIX", "").strip()
        self.image_url_root = os.environ.get("VLM_IMAGE_URL_ROOT", "").strip()
        self._json_schema_raw = os.environ.get("VLM_JSON_SCHEMA", "").strip()
        if self._json_schema_raw:
            try:
                self.json_schema = json.loads(self._json_schema_raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"VLM_JSON_SCHEMA must be valid JSON: {exc}") from exc
        self.log_payload = _env_flag("VLM_LOG_PAYLOAD", default=False)
        self.concurrency = max(1, int(concurrency))
        self.timeout = self._build_timeout(timeout_s)
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self.sem = asyncio.Semaphore(self.concurrency)
        logger.info("VLM client init base_url=%s model=%s timeout_s=%s concurrency=%s max_image_side=%s image_mode=%s",
                    self.base_url, self.model, timeout_s, self.concurrency, self.max_image_side, self.image_mode)
        self._validate_backend()
        self._validate_image_mode()
        self._validate_mode()
        self.cache_id = self._build_cache_id()

    async def close(self):
        await self.client.aclose()

    async def annotate(self, tile_id: str, rgb: np.ndarray, dets: List[DetBox],
                       class_counts: Dict[str, int], image_path: Optional[str] = None) -> TileSemantics:
        with tracer.start_as_current_span("vlm.annotate") as span:
            span.set_attribute("tile.id", tile_id)
            span.set_attribute("detections.count", len(dets))
            t0 = time.perf_counter()
            logger.info("VLM annotate start tile=%s dets=%s", tile_id, len(dets))
            png_b64 = ""
            scale_x = 1.0
            scale_y = 1.0
            image_size = (int(rgb.shape[1]), int(rgb.shape[0]))
            if self.image_mode == "base64":
                png_b64, scale_x, scale_y, image_size = self._to_png_b64(rgb)
            prompt = self._prompt(tile_id, dets, class_counts, scale_x, scale_y, image_size)
            payload = self._build_payload(prompt, png_b64, image_path=image_path)
            structured_outputs, structured_kind = self._build_structured_outputs()
            if structured_outputs is not None:
                payload["structured_outputs"] = structured_outputs
            span.set_attribute("vlm.mode", self.mode)
            span.set_attribute("vlm.backend", self.backend)
            span.set_attribute("vlm.structured", structured_kind)
            span.set_attribute("vlm.image_width", image_size[0])
            span.set_attribute("vlm.image_height", image_size[1])
            logger.info(
                "VLM request backend=%s mode=%s structured=%s model=%s messages=%s image_mode=%s image_b64_len=%s image_size=%sx%s scale=%.3f,%.3f",
                self.backend,
                self.mode,
                structured_kind,
                self.model,
                len(payload.get("messages", [])),
                self.image_mode,
                len(png_b64),
                image_size[0],
                image_size[1],
                scale_x,
                scale_y,
            )
            self._log_payload(payload)
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
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
                    if not sem.tile_id:
                        sem.tile_id = tile_id
                    logger.info("VLM annotate done tile=%s annotations=%s elapsed_s=%.3f",
                                tile_id, len(sem.annotations), time.perf_counter() - t0)
                    return sem
                except (json.JSONDecodeError, ValueError, httpx.TimeoutException, httpx.ReadError) as exc:
                    if attempt < attempts - 1:
                        delay = backoffs[attempt]
                        if isinstance(exc, httpx.ReadError):
                            delay = random.uniform(1.0, 3.0)
                            logger.warning("VLM retry read_error attempt=%s tile=%s delay=%.2fs err=%s",
                                           attempt + 1, tile_id, delay, exc)
                        else:
                            logger.warning("VLM retry parse/timeout attempt=%s tile=%s err=%s", attempt + 1, tile_id, exc)
                        await asyncio.sleep(delay)
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

    def _validate_backend(self) -> None:
        allowed = {"openai", "vllm_qwen"}
        if self.backend not in allowed:
            raise RuntimeError(f"Unsupported VLM_BACKEND={self.backend!r}. Use one of: openai, vllm_qwen.")
        if self.backend == "vllm_qwen" and self.concurrency > 2:
            logger.warning("VLM_BACKEND=vllm_qwen with concurrency=%s may cause ReadError; consider 1-2.",
                           self.concurrency)

    def _validate_image_mode(self) -> None:
        allowed = {"base64", "url", "path"}
        if self.image_mode not in allowed:
            raise RuntimeError(f"Unsupported VLM_IMAGE_MODE={self.image_mode!r}. Use one of: base64, url, path.")
        if self.image_mode == "path" and self.backend == "openai":
            raise RuntimeError("VLM_IMAGE_MODE=path is not supported for VLM_BACKEND=openai; use base64 or url.")

    def _build_payload(self, prompt: str, png_b64: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if self.backend == "openai":
            image_url = self._build_image_ref(png_b64, image_path, for_openai=True)
            messages[1]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            return payload
        if self.backend == "vllm_qwen":
            image_ref = self._build_image_ref(png_b64, image_path, for_openai=False)
            payload["mm_processor_kwargs"] = {"images": [image_ref]}
            return payload
        raise RuntimeError(f"Unsupported VLM_BACKEND={self.backend!r}")

    def _build_timeout(self, timeout_s: float) -> httpx.Timeout:
        read = float(os.environ.get("VLM_TIMEOUT_READ", str(timeout_s)))
        connect = float(os.environ.get("VLM_TIMEOUT_CONNECT", "10"))
        write = float(os.environ.get("VLM_TIMEOUT_WRITE", "30"))
        pool = float(os.environ.get("VLM_TIMEOUT_POOL", "10"))
        return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)

    def _build_image_ref(self, png_b64: str, image_path: Optional[str], for_openai: bool) -> str:
        if self.image_mode == "base64":
            if for_openai:
                return f"data:image/png;base64,{png_b64}"
            return png_b64
        if not image_path:
            raise RuntimeError("VLM_IMAGE_MODE requires image_path. Enable tile saving or use base64.")
        if self.image_mode == "path":
            return image_path
        if self.image_mode == "url":
            if image_path.startswith("http://") or image_path.startswith("https://"):
                return image_path
            if not self.image_url_prefix:
                raise RuntimeError("VLM_IMAGE_MODE=url requires VLM_IMAGE_URL_PREFIX or an absolute URL path.")
            rel = image_path
            if self.image_url_root and os.path.isabs(image_path):
                try:
                    root = os.path.abspath(self.image_url_root)
                    path_abs = os.path.abspath(image_path)
                    if os.path.commonpath([path_abs, root]) == root:
                        rel = os.path.relpath(path_abs, root)
                    else:
                        rel = os.path.basename(image_path)
                except ValueError:
                    rel = os.path.basename(image_path)
            rel = rel.replace(os.sep, "/")
            return self.image_url_prefix.rstrip("/") + "/" + rel.lstrip("/")
        raise RuntimeError(f"Unsupported VLM_IMAGE_MODE={self.image_mode!r}")

    def _build_cache_id(self) -> str:
        parts = [self.model, self.backend, self.mode, f"max{self.max_image_side}", f"img:{self.image_mode}"]
        if self.prompt_id:
            parts.append(f"prompt:{self.prompt_id}")
        if self.mode == "json" and self.json_schema:
            blob = json.dumps(self.json_schema, sort_keys=True)
            parts.append(f"json={self._short_hash(blob)}")
        elif self.mode == "choice" and self.choice_list:
            blob = "|".join(self.choice_list)
            parts.append(f"choice={self._short_hash(blob)}")
        return "|".join(parts)

    def _short_hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

    def _validate_mode(self) -> None:
        allowed = {"free_text", "json", "choice"}
        if self.mode not in allowed:
            raise RuntimeError(f"Unsupported VLM_MODE={self.mode!r}. Use one of: free_text, json, choice.")
        if self.mode == "json":
            if self.choice_list:
                raise RuntimeError("VLM_MODE=json conflicts with VLM_CHOICE_LIST; only one constraint allowed.")
            if self.json_schema is None:
                self.json_schema = self._default_json_schema()
                logger.info("VLM json schema defaulted to built-in schema.")
        elif self.mode == "choice":
            if self._json_schema_raw:
                raise RuntimeError("VLM_MODE=choice conflicts with VLM_JSON_SCHEMA; only one constraint allowed.")
            if not self.choice_list:
                raise RuntimeError("VLM_MODE=choice requires non-empty VLM_CHOICE_LIST.")
        else:
            if self.choice_list or self._json_schema_raw:
                logger.warning("VLM_MODE=free_text ignores structured output settings.")

    def _build_structured_outputs(self) -> Tuple[Optional[Dict[str, Any]], str]:
        if self.mode == "free_text":
            return None, "none"
        if self.mode == "json":
            return {"json": self.json_schema}, "json"
        if self.mode == "choice":
            return {"choice": self.choice_list}, "choice"
        raise RuntimeError(f"Unsupported VLM_MODE={self.mode!r}")

    def _default_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tile_id": {"type": "string"},
                "scene": {"type": "string"},
                "annotations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "det_id": {"type": "integer"},
                            "label": {"type": "string"},
                            "attributes": {"type": "object"},
                            "notes": {"type": ["string", "null"]},
                        },
                        "required": ["det_id", "label", "attributes"],
                    },
                },
            },
            "required": ["tile_id", "scene", "annotations"],
        }

    def _log_payload(self, payload: Dict[str, Any]) -> None:
        if not self.log_payload:
            return
        safe = json.loads(json.dumps(payload))
        mm_kwargs = safe.get("mm_processor_kwargs")
        if isinstance(mm_kwargs, dict) and isinstance(mm_kwargs.get("images"), list):
            mm_kwargs["images"] = [f"<redacted:{len(x)}>" for x in mm_kwargs["images"]]
        for msg in safe.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url", {})
                url = image_url.get("url", "")
                image_url["url"] = f"<redacted:{len(url)}>"
        logger.info("VLM payload=%s", json.dumps(safe, ensure_ascii=True))

    def _scale_bbox(self, bbox: Tuple[int, int, int, int], scale_x: float, scale_y: float) -> List[int]:
        x1, y1, x2, y2 = bbox
        return [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        ]

    def _prompt(self, tile_id: str, dets: List[DetBox], class_counts: Dict[str, int],
                scale_x: float, scale_y: float, image_size: Tuple[int, int]) -> str:
        scaled = not (abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6)
        det_list = []
        for d in dets:
            bbox = list(d.bbox_px)
            if scaled:
                bbox = self._scale_bbox(d.bbox_px, scale_x, scale_y)
            det_list.append({"det_id": d.det_id, "bbox": bbox, "det_score": d.score, "det_cls": d.cls})
        schema = {
            "tile_id": tile_id,
            "tile_summary": {
                "scene_type": "string",
                "terrain": {
                    "relief": "flat|rolling|hilly|uneven|mixed|uncertain",
                    "elevation_changes_visible": True,
                    "natural_features": {
                        "forest": {
                            "present": True,
                            "density": "sparse|medium|dense|mixed|uncertain",
                            "clearings": True,
                            "logging_traces": True,
                        },
                        "water": {
                            "present": False,
                            "type": "river|lake|pond|ditch|uncertain",
                        },
                        "open_fields": {
                            "present": True,
                            "surface": "soil|grass|mixed|uncertain",
                            "disturbance_visible": True,
                        },
                    },
                },
                "activity_traces": {
                    "vehicle_tracks": {
                        "present": True,
                        "pattern": "linear|curved|repeated|chaotic|uncertain",
                    },
                    "excavation_or_earthworks": {
                        "present": False,
                        "type": "trench|pit|mound|crater|uncertain",
                    },
                    "construction_activity": {
                        "present": True,
                        "stage": "early|ongoing|abandoned|uncertain",
                    },
                },
                "overall_observations": "string",
            },
            "detections": [
                {
                    "det_id": 0,
                    "bbox": [0, 0, 0, 0],
                    "label": "string",
                    "confidence_visual": "high|medium|low|uncertain",
                    "object_description": {
                        "shape": "rectangular|circular|elongated|irregular|complex|uncertain",
                        "size_relative": "small|medium|large|uncertain",
                        "orientation": "north-south|east-west|diagonal|irregular|uncertain",
                        "levels": "single-story|multi-story|uncertain",
                        "material": "concrete|metal|wood|mixed|uncertain",
                        "roof": {
                            "type": "flat|gabled|pitched|irregular|uncertain",
                            "color": "string|uncertain",
                            "condition": "intact|damaged|partially damaged|patched|uncertain",
                            "visible_details": "string",
                        },
                        "surface_type": "paved|dirt|grass|mixed|uncertain",
                        "construction_state": "completed|under_construction|abandoned|ruined|uncertain",
                        "vehicle_type": "civilian|truck|heavy_vehicle|tracked|wheeled|uncertain|none",
                    },
                    "local_context": {
                        "adjacent_objects": ["road", "path", "fence", "vegetation", "other_structures"],
                        "ground_conditions": {
                            "disturbed": True,
                            "tracks_visible": True,
                            "soil_color_variation": True,
                        },
                        "accessibility": {
                            "direct_road_access": True,
                            "footpaths_visible": True,
                        },
                    },
                    "analyst_notes": {
                        "uncertainties": "string",
                        "why_label_was_chosen": "string",
                        "potential_alternative_interpretations": "string",
                    },
                }
            ],
        }
        size_note = f"Image size (px): {image_size[0]}x{image_size[1]}"
        if scaled:
            size_note += " (bboxes are scaled to this image)"
        data = {
            "tile_id": tile_id,
            "size_note": size_note,
            "class_counts": json.dumps(class_counts),
            "detections": json.dumps(det_list),
            "schema": json.dumps(schema),
        }
        return self.user_prompt_template.format_map(data)

    def _load_prompt_config(self) -> Dict[str, str]:
        path = os.environ.get("VLM_PROMPTS_PATH", "").strip()
        default_cfg = self._default_prompt_config()
        if not path:
            rel = os.path.join(os.path.dirname(__file__), "..", "..", "config", "vlm_prompts.json")
            path = os.path.abspath(rel)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                system = str(payload.get("system_prompt") or "").strip()
                user = str(payload.get("user_prompt_template") or "").strip()
                if system and user:
                    logger.info("VLM prompt config loaded path=%s", path)
                    return {"system_prompt": system, "user_prompt_template": user}
                logger.warning("VLM prompt config missing fields path=%s; using defaults", path)
            except Exception as exc:
                logger.warning("VLM prompt config load failed path=%s err=%s; using defaults", path, exc)
        else:
            logger.info("VLM prompt config not found path=%s; using defaults", path)
        return default_cfg

    def _default_prompt_config(self) -> Dict[str, str]:
        return {
            "system_prompt": (
                "You are a cautious OSINT satellite imagery analyst.\n"
                "You must NOT infer function, ownership, or intent.\n"
                "You must NOT use words like \"base\", \"military site\", \"warehouse\", or \"facility\" unless the function is visually undeniable.\n"
                "If something is unclear, mark it as \"uncertain\" instead of guessing.\n"
                "Your priority is descriptive accuracy, not classification.\n"
                "Assume that human analysts will rely on your descriptions to search, filter, and verify objects later.\n"
                "Return ONLY valid JSON. Do not change bbox. Do not invent new det_id. Use det_id mapping."
            ),
            "user_prompt_template": (
                "You analyze a satellite tile. You are given YOLO detections as weak context.\n"
                "Do NOT copy YOLO class names unless visually supported. Separate observation from interpretation.\n"
                "Split your description into three levels:\n"
                "1) Object-level: what is inside each bbox.\n"
                "2) Local context: what is immediately around each object.\n"
                "3) Tile-level environment: overall scene, terrain, activity traces.\n"
                "If uncertain, explicitly use \"uncertain\" in the relevant fields.\n"
                "{size_note}\n"
                "YOLO tile class counts:\n{class_counts}\n"
                "Detections (det_id mapping, do not change bbox):\n{detections}\n"
                "Return ONLY JSON matching schema:\n{schema}\n"
            ),
        }

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

    def _to_png_b64(self, rgb: np.ndarray) -> Tuple[str, float, float, Tuple[int, int]]:
        im = Image.fromarray(rgb, mode="RGB")
        w, h = im.size
        scale_x = 1.0
        scale_y = 1.0
        if self.max_image_side and max(w, h) > self.max_image_side:
            scale = float(self.max_image_side) / float(max(w, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            im = im.resize((new_w, new_h), resample=Image.BICUBIC)
            scale_x = new_w / max(w, 1)
            scale_y = new_h / max(h, 1)
            w, h = new_w, new_h
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii"), scale_x, scale_y, (w, h)
