# Навчання CNN/CLIP gate та підготовка датасетів

Цей гайд описує, як підготувати тайли, розмітити їх як **keep/drop**, підібрати пороги для CLIP gate, а також навчити CNN gate під поточну архітектуру (`MobileNetV3-small`, 1 logit).

## 1) Підготовка датасетів (тайли keep/drop)

Gate — це бінарний фільтр. Тому вам потрібні два набори тайлів:
- **keep**: цікаві сцени (забудова, дороги, техніка, інфраструктура).
- **drop**: шум (ліс, поле, вода, хмари, однорідні області).

### Варіант A: через UI (найшвидше)

1) Запустіть UI:
```bash
streamlit run geovlm_poc/streamlit_app.py
```

2) У лівій панелі:
- `GATE_MODE=none` (щоб не відсіювати тайли)
- `Save tile PNG previews = true`

3) Запустіть аналіз. Після завершення тайли будуть у `runs/<run_id>/tiles/<image_id>/`.

4) Розмітьте вручну: розкладіть PNG у дві папки `keep/` і `drop/`.

### Варіант B: експорт тайлів напряму з Python

```bash
python - <<'PY'
from geovlm_poc.tiling import ImageTiler
from PIL import Image
import os

image_path = "/data/dateA.tif"
out_dir = "./data/tiles_raw"

tiler = ImageTiler(tile_size=1024, overlap=256)
os.makedirs(out_dir, exist_ok=True)

for tref in tiler.iter_tiles(image_path, image_id="dateA"):
    rgb = tiler.read_tile_rgb(image_path, tref)
    out_path = os.path.join(out_dir, f"{tref.tile_id}.png")
    Image.fromarray(rgb).save(out_path)
PY
```

Після цього вручну розмітьте тайли у `keep/` і `drop/`.

## 2) CLIP gate: підбір порога та (опційно) fine-tune

CLIP gate працює без навчання, але дуже залежить від:
- якості промптів,
- порога `CLIP_GATE_THR`.

### 2.1 Підбір `CLIP_GATE_THR` на розмічених тайлах

Використайте скрипт `geovlm_poc/scripts/pick_clip_thr.py`:

```bash
python geovlm_poc/scripts/pick_clip_thr.py \
  --keep-dir ./data/tiles/keep \
  --drop-dir ./data/tiles/drop \
  --device cuda \
  --model ViT-B-32 \
  --pretrained openai \
  --target-recall 0.98 \
  --out-csv ./out/clip_scores.csv
```

Скрипт надрукує рекомендовані пороги. Встановіть у `.env`:
```bash
CLIP_GATE_THR=0.18
```

### 2.2 Fine-tune OpenCLIP (опційно)

Формат CSV (два поля): `image_path,text`

Приклад `train.csv`:
```text
image_path,text
data/tiles/keep/0001.png,dense urban area from above
data/tiles/keep/0002.png,industrial site aerial view
data/tiles/drop/0101.png,forest from above
data/tiles/drop/0102.png,water surface aerial view
```

Запуск:
```bash
python geovlm_poc/scripts/finetune_openclip.py \
  --csv ./train.csv \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --epochs 3 \
  --batch-size 64 \
  --device cuda \
  --out ./out/openclip_finetuned.pt
```

Примітка: `CLIPGate` зараз використовує `open_clip.create_model_and_transforms(..., pretrained=...)` і не завантажує `state_dict` напряму.
Якщо ваша версія `open_clip` підтримує локальний шлях у `pretrained`, встановіть `CLIP_PRETRAINED=/path/to/openclip_finetuned.pt`.
Якщо ні — додайте `load_state_dict()` у `CLIPGate._lazy()` під ваші ваги.

## 3) CNN gate: навчання під MobileNetV3-small

`CNNGate` очікує чекпойнт `state_dict` для `mobilenet_v3_small` з одним виходом (1 logit).
Нормалізація та ресайз повинні бути такими самими, як у `CNNGate`:
- `Resize(224, 224)`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

### 3.1 Рекомендована структура даних

```text
data/cnn/
  train/
    keep/
    drop/
  val/
    keep/
    drop/
```

### 3.2 Мінімальний приклад тренування (PyTorch)

```python
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("data/cnn/train", transform=tfm)
val_ds = datasets.ImageFolder("data/cnn/val", transform=tfm)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
model = model.cuda()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
crit = torch.nn.BCEWithLogitsLoss()

for ep in range(5):
    model.train()
    for x, y in train_dl:
        x = x.cuda()
        y = y.float().cuda()
        logit = model(x).squeeze(1)
        loss = crit(logit, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(model.state_dict(), "cnn_gate.pt")
```

### 3.3 Підключення у пайплайн

```bash
GATE=cnn CNN_GATE_CKPT=/path/to/cnn_gate.pt CNN_GATE_THR=0.5 CNN_DEVICE=cuda \
geovlm-poc analyze --a /data/dateA.tif --b /data/dateB.tif --out ./out_run
```

## 4) Поради для якості датасету

- Балансуйте класи keep/drop (або використовуйте class weights).
- Включайте складні випадки: сніг, хмари, тінь, туман.
- Збирайте тайли з різних географічних зон та сезонів.
- Тримайте однаковий `TILE_SIZE` і `OVERLAP`, як у production.
