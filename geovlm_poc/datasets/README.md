# Dataset Tools

Scripts in this folder help you build datasets for:
- CNN gate (binary: interesting vs boring)
- Retrieval (weak labels from OSM + optional blue-roof attribute)

Inputs:
- GeoTIFF images for tiling
- OSM features exported as GeoJSON (EPSG:4326 by default)

## CNN gate dataset

Creates an ImageFolder-style dataset:

```
data_cnn_gate/
  train/interesting, train/boring
  val/interesting, val/boring
  test/interesting, test/boring
```

Command:
```
python geovlm_poc/datasets/build_cnn_gate_dataset.py \
  --image /path/to/scene.tif \
  --osm /path/to/osm.geojson \
  --out data_cnn_gate
```

It uses OSM tags to mark "interesting" tiles and heuristics to keep "boring" tiles clean.

## Retrieval dataset

Produces a JSONL index with weak labels and optional CSV for CLIP fine-tune.

Command:
```
python geovlm_poc/datasets/build_retrieval_dataset.py \
  --image /path/to/scene.tif \
  --osm /path/to/osm.geojson \
  --out data_retrieval \
  --emit-csv
```

Outputs:
- `data_retrieval/tiles.jsonl` with labels and attributes
- `data_retrieval/images/` with tile crops
- `data_retrieval/train.csv` (image_path,text) if `--emit-csv` is set

## Notes

- OSM tags are weak labels; keep a strict intersection threshold to reduce noise.
- For negatives, exclude tiles near positives and keep low-texture tiles.
- For large areas, split by grid to avoid geographic leakage.
