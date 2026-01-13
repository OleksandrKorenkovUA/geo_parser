from typing import Dict, List, Tuple
import logging
import numpy as np
from .types import DetBox
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class Detector:
    def detect(self, rgb: np.ndarray) -> Tuple[List[DetBox], Dict[str, int]]:
        raise NotImplementedError


class YOLODetector(Detector):
    def __init__(self, model_path: str = "yolo12n.pt", conf: float = 0.25, max_det: int = 400,
                 device: str = "cpu"):
        self.model_path = model_path
        self.conf = conf
        self.max_det = max_det
        self.device = device
        self._model = None

    def _lazy(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info("YOLO model loaded path=%s", self.model_path)

    def detect(self, rgb: np.ndarray) -> Tuple[List[DetBox], Dict[str, int]]:
        with tracer.start_as_current_span("detector.yolo") as span:
            self._lazy()
            res = self._model.predict(rgb, conf=self.conf, max_det=self.max_det, device=self.device, verbose=False)[0]
            boxes: List[DetBox] = []
            counts: Dict[str, int] = {}
            if res.boxes is None:
                logger.info("YOLO detect no boxes")
                return boxes, counts
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.int32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32)
            cls = res.boxes.cls.cpu().numpy().astype(np.int32)
            names = res.names if hasattr(res, "names") else {}
            for i, (b, s, c) in enumerate(zip(xyxy, conf, cls)):
                label = str(names.get(int(c), int(c)))
                counts[label] = counts.get(label, 0) + 1
                x1, y1, x2, y2 = map(int, b.tolist())
                boxes.append(DetBox(det_id=i, bbox_px=(x1, y1, x2, y2), score=float(s), cls=label))
            span.set_attribute("detections.count", len(boxes))
            logger.info("YOLO detect boxes=%s classes=%s", len(boxes), counts)
            return boxes, counts
