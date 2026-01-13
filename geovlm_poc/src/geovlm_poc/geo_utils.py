import logging
from typing import List, Optional

import numpy as np

try:
    from rasterio.crs import CRS
except Exception:  # pragma: no cover - optional at runtime
    CRS = None


logger = logging.getLogger(__name__)


def strtree_query_geoms(tree, geoms: List, q) -> List:
    cand = tree.query(q)
    if cand is None:
        return []
    try:
        first = cand[0]
    except Exception:
        return []
    if isinstance(first, (int, np.integer)):
        return [geoms[int(i)] for i in cand]
    return list(cand)


def _to_crs_obj(crs) -> Optional["CRS"]:
    if crs is None or CRS is None:
        return None
    if hasattr(crs, "is_geographic"):
        return crs
    if isinstance(crs, str):
        try:
            return CRS.from_user_input(crs)
        except Exception:
            return None
    return None


def is_geographic_crs(crs) -> bool:
    crs_obj = _to_crs_obj(crs)
    if crs_obj is None:
        return True
    try:
        return bool(crs_obj.is_geographic)
    except Exception:
        return True


def require_projected_crs(crs, label: str) -> None:
    if crs is None:
        raise ValueError(f"{label} CRS is missing; distance buffers require projected CRS in meters.")
    if is_geographic_crs(crs):
        raise ValueError(f"{label} CRS is geographic; distance buffers require projected CRS in meters.")
