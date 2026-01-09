from typing import Iterable, Tuple
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from .types import TileRef
from .telemetry import get_tracer


logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ImageTiler:
    def __init__(self, tile_size: int = 1024, overlap: int = 256, bands: Tuple[int, int, int] = (1, 2, 3)):
        self.tile_size, self.overlap, self.bands = tile_size, overlap, bands

    def iter_tiles(self, path: str, image_id: str) -> Iterable[TileRef]:
        with tracer.start_as_current_span("tiler.iter_tiles") as span:
            span.set_attribute("image.path", path)
            with rasterio.open(path) as ds:
                w, h = ds.width, ds.height
                step = self.tile_size - self.overlap
                ncols = int(np.ceil((w - self.overlap) / step))
                nrows = int(np.ceil((h - self.overlap) / step))
                logger.info("Iter tiles image=%s size=%sx%s tile=%s overlap=%s grid=%sx%s",
                            image_id, w, h, self.tile_size, self.overlap, ncols, nrows)
                for r in range(nrows):
                    for c in range(ncols):
                        x0 = c * step
                        y0 = r * step
                        tw = min(self.tile_size, w - x0)
                        th = min(self.tile_size, h - y0)
                        if tw <= 0 or th <= 0:
                            continue
                        win = Window(x0, y0, tw, th)
                        b = window_bounds(win, ds.transform)
                        tr = ds.window_transform(win)
                        tile_id = f"r{r:05d}_c{c:05d}"
                        yield TileRef(
                            image_id=image_id,
                            tile_id=tile_id,
                            row=r,
                            col=c,
                            window=(int(x0), int(y0), int(tw), int(th)),
                            crs_wkt=(ds.crs.to_wkt() if ds.crs else ""),
                            transform=(tr.a, tr.b, tr.c, tr.d, tr.e, tr.f),
                            bounds=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        )

    def read_tile_rgb(self, path: str, tref: TileRef) -> np.ndarray:
        with tracer.start_as_current_span("tiler.read_tile") as span:
            span.set_attribute("tile.id", tref.tile_id)
            with rasterio.open(path) as ds:
                x0, y0, tw, th = tref.window
                win = Window(x0, y0, tw, th)
                arr = ds.read(list(self.bands), window=win, boundless=True, fill_value=0)
                arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = self._to_uint8(arr)
                logger.info("Read tile image=%s tile=%s shape=%s", tref.image_id, tref.tile_id, arr.shape)
                return arr

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        a = arr.astype(np.float32)
        lo = np.nanpercentile(a, 2)
        hi = np.nanpercentile(a, 98)
        if hi <= lo:
            return np.clip(a, 0, 255).astype(np.uint8)
        a = (a - lo) / (hi - lo)
        a = np.clip(a, 0, 1)
        return (a * 255.0).astype(np.uint8)
