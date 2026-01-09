from typing import Dict, List, Tuple
import logging
import numpy as np
import cv2
from PIL import Image
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class TileGate:
    def keep(self, rgb: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        raise NotImplementedError


class HeuristicGate(TileGate):
    def __init__(self, min_score: float = 0.24, min_edge_density: float = 0.05, min_entropy: float = 3.2):
        self.min_score, self.min_edge_density, self.min_entropy = min_score, min_edge_density, min_entropy

    def keep(self, rgb: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        with tracer.start_as_current_span("gate.heuristic") as span:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_density = float(edges.mean() / 255.0)
            hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
            p = hist / max(hist.sum(), 1.0)
            entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
            score = 0.6 * self._norm(edge_density, 0.01, 0.14) + 0.4 * self._norm(entropy, 2.0, 6.5)
            m = {"score": float(score), "edge_density": edge_density, "entropy": entropy}
            ok = (m["score"] >= self.min_score and m["edge_density"] >= self.min_edge_density and m["entropy"] >= self.min_entropy)
            span.set_attribute("gate.ok", ok)
            logger.info("Gate heuristic ok=%s metrics=%s", ok, m)
            return ok, m

    def _norm(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0, 1))


class CLIPGate(TileGate):
    def __init__(
        self,
        keep_prompts: List[str],
        drop_prompts: List[str],
        threshold: float = 0.15,
        device: str = "cpu",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ):
        self.keep_prompts = keep_prompts
        self.drop_prompts = drop_prompts
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._keep_text = None
        self._drop_text = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch, open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        with torch.no_grad():
            kt = tokenizer(self.keep_prompts).to(self.device)
            dt = tokenizer(self.drop_prompts).to(self.device)
            self._keep_text = self._model.encode_text(kt)
            self._keep_text /= self._keep_text.norm(dim=-1, keepdim=True)
            self._drop_text = self._model.encode_text(dt)
            self._drop_text /= self._drop_text.norm(dim=-1, keepdim=True)
        logger.info("CLIP gate model loaded name=%s pretrained=%s", self.model_name, self.pretrained)

    def keep(self, rgb: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        with tracer.start_as_current_span("gate.clip") as span:
            self._lazy()
            import torch
            img = Image.fromarray(rgb, mode="RGB")
            x = self._preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vi = self._model.encode_image(x)
                vi /= vi.norm(dim=-1, keepdim=True)
                keep_sim = float((vi @ self._keep_text.T).max().item())
                drop_sim = float((vi @ self._drop_text.T).max().item())
                score = keep_sim - drop_sim
            ok = score >= self.threshold
            span.set_attribute("gate.ok", ok)
            logger.info("Gate clip ok=%s score=%.4f keep=%.4f drop=%.4f", ok, score, keep_sim, drop_sim)
            return ok, {"clip_keep_sim": keep_sim, "clip_drop_sim": drop_sim, "clip_score": score}


class CNNGate(TileGate):
    def __init__(self, ckpt_path: str, threshold: float = 0.5, device: str = "cpu"):
        self.ckpt_path = ckpt_path
        self.threshold = threshold
        self.device = device
        self._model = None
        self._tfm = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from torchvision import models, transforms
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 1)
        sd = torch.load(self.ckpt_path, map_location=self.device)
        m.load_state_dict(sd)
        self._model = m.to(self.device).eval()
        self._tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("CNN gate model loaded ckpt=%s device=%s", self.ckpt_path, self.device)

    def keep(self, rgb: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        with tracer.start_as_current_span("gate.cnn") as span:
            self._lazy()
            import torch
            img = Image.fromarray(rgb, mode="RGB")
            x = self._tfm(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self._model(x).squeeze(1)
                prob = float(torch.sigmoid(logit).item())
            ok = prob >= self.threshold
            span.set_attribute("gate.ok", ok)
            logger.info("Gate cnn ok=%s prob=%.4f", ok, prob)
            return ok, {"cnn_prob": prob}
