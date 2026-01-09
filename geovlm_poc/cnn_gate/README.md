# CNN Gate Training (MobileNetV3-small)

This folder contains scripts and notes for training a lightweight CNN gate that
filters "interesting" vs "boring" tiles before expensive stages (YOLO/VLM/CLIP).

## 1) Define the labels

"Interesting" tiles are those worth passing to the next stage, for example:
- urban areas, roads, railways, industrial sites, ports, quarries
- dense infrastructure and structured patterns

"Boring" tiles are mostly uniform:
- fields, forests without structure, water, clouds/fog, empty plains

The goal is high recall for "interesting" tiles, not perfect accuracy.

## 2) Dataset layout (ImageFolder)

```
data_cnn_gate/
  train/
    interesting/
      *.jpg|png
    boring/
      *.jpg|png
  val/
    interesting/
    boring/
  test/
    interesting/
    boring/
```

Class names are important: `interesting` is treated as the positive class.

## 2.1) Build the dataset from GeoTIFF + OSM

If you have OSM GeoJSON, you can create weak labels automatically:
```
python geovlm_poc/datasets/build_cnn_gate_dataset.py \
  --image /path/to/scene.tif \
  --osm /path/to/osm.geojson \
  --out data_cnn_gate
```

## 3) Train the model

```
python geovlm_poc/cnn_gate/train_cnn_gate.py --data data_cnn_gate --out cnn_gate_mnv3s.pt --epochs 10 --device cuda
```

This saves a `state_dict` that matches `CNNGate` in the pipeline.

## 4) Pick a threshold (CNN_GATE_THR)

```
python geovlm_poc/cnn_gate/pick_thr.py --data data_cnn_gate --ckpt cnn_gate_mnv3s.pt --split val --target_recall 0.98
```

The script searches thresholds and reports:
- best F1
- lowest pass rate that still meets target recall

## 5) Wire it into the pipeline

In `.env`:
```
CNN_GATE_CKPT=/abs/path/to/cnn_gate_mnv3s.pt
CNN_GATE_THR=0.32
CNN_DEVICE=cuda:0
```

## Notes

- The gate uses MobileNetV3-small with a single-logit head.
- Inference in `CNNGate` uses resize to 224x224 + ImageNet normalization.
- Keep the training `imgsz` at 224 to match inference.
- If you change the architecture, the `state_dict` will not load.
