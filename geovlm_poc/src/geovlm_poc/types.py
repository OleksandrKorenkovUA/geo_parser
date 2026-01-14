from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class TileRef:
    image_id: str
    tile_id: str
    row: int
    col: int
    window: Tuple[int, int, int, int]
    crs_wkt: str
    transform: Tuple[float, float, float, float, float, float]
    bounds: Tuple[float, float, float, float]


class DetBox(BaseModel):
    det_id: int
    bbox_px: Tuple[int, int, int, int]
    score: float = Field(ge=0.0, le=1.0)
    cls: Optional[str] = None


class VLMBoxAnnotation(BaseModel):
    det_id: int
    label: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None


class VLMTileSummary(BaseModel):
    scene_type: Optional[str] = None
    terrain: Dict[str, Any] = Field(default_factory=dict)
    activity_traces: Dict[str, Any] = Field(default_factory=dict)
    overall_observations: Optional[str] = None


class VLMDetection(BaseModel):
    det_id: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    label: Optional[str] = None
    confidence_visual: Optional[str] = None
    object_description: Dict[str, Any] = Field(default_factory=dict)
    local_context: Dict[str, Any] = Field(default_factory=dict)
    analyst_notes: Dict[str, Any] = Field(default_factory=dict)


class TileSemantics(BaseModel):
    tile_id: Optional[str] = None
    scene: Optional[str] = None
    annotations: List[VLMBoxAnnotation] = Field(default_factory=list)
    tile_summary: Optional[VLMTileSummary] = None
    detections: List[VLMDetection] = Field(default_factory=list)


class GeoObject(BaseModel):
    image_id: str
    obj_id: str
    label: str
    confidence: float
    geometry: Dict[str, Any]
    attributes: Dict[str, Any] = Field(default_factory=dict)
    tile_id: Optional[str] = None


class ChangeEvent(BaseModel):
    change_type: str
    label: str
    geometry: Dict[str, Any]
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class ChangeReport(BaseModel):
    image_a: str
    image_b: str
    crs_wkt: str
    generated_at_unix: int
    changes: List[ChangeEvent]
