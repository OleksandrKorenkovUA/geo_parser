# geovlm_poc

PoC-проєкт для аналізу великих супутникових GeoTIFF (2+ ГБ): нарізає зображення на тайли, знаходить об’єкти через YOLO, підписує їх VLM за суворою JSON-схемою, переводить детекції у гео-координати, виконує порівняння двох дат і забезпечує семантичний пошук через embeddings + векторний індекс.

## З чого складається система

Пайплайн побудований так, щоб:
- не завантажувати весь GeoTIFF у памʼять,
- мінімізувати дорогі VLM-запити,
- тримати стабільну геометрію (bbox від YOLO),
- додати семантику та атрибути (VLM),
- забезпечити коректне співставлення обʼєктів між датами,
- дати вільний текстовий пошук по результатах.

**Архітектура:**

Tiler -> Gate -> YOLO -> VLM -> GeoAggregator -> ChangeDetector -> SemanticIndex

Коротко про ролі:
- **ImageTiler**: читає GeoTIFF вікнами (tiles), повертає метадані тайла і трансформацію для конвертації пікселів у CRS.
- **Gate (Heuristic/CLIP/CNN)**: відсікає «порожні»/нецікаві тайли, щоб зменшити кількість VLM-запитів.
- **YOLODetector**: дає стабільні bbox координати та підрахунок класів як контекст.
- **VLMAnnotator**: додає семантику та атрибути до детекцій, не змінює bbox і не вигадує нові det_id.
- **GeoAggregator**: переводить bbox у полігони CRS і прибирає дублікати, що виникли через overlap.
- **ChangeDetector**: порівнює обʼєкти між датами з толерансом до зсуву та визначає added/removed/modified.
- **SemanticIndex/Searcher**: будує embeddings, векторний індекс і дозволяє текстовий пошук з фільтрами.

## Встановлення

```bash
pip install -e .
```

Опційні залежності:

```bash
pip install -e ".[yolo]"
pip install -e ".[clip]"
pip install -e ".[cnn]"
pip install -e ".[faiss]"
```

## Налаштування оточення

Скопіюйте приклад та заповніть значення:

```bash
cp .env.example .env
```

Обов’язкові змінні:
- Для `analyze`: `VLM_BASE_URL`, `VLM_API_KEY`
- Для `build-index` і `search`: `EMB_BASE_URL`, `EMB_API_KEY`

### Приклад для ваших моделей (lightrag + qwen3-vl)

У вашій інфраструктурі:
- **VLM для аналізу зображень**: `qwen3-vl` на `http://gpu-test.silly.billy:8011/v1`
- **Embeddings**: `multilingual-embeddings` на `http://gpu-test.silly.billy:8022/v1`

Встановіть змінні так:

```bash
VLM_BASE_URL=http://gpu-test.silly.billy:8011/v1
VLM_API_KEY=DUMMY_KEY
VLM_MODEL=qwen3-vl

EMB_BASE_URL=http://gpu-test.silly.billy:8022/v1
EMB_API_KEY=DUMMY_KEY
EMB_MODEL=multilingual-embeddings
```

Примітка: цей PoC працює напряму з OpenAI-сумісними `/chat/completions` і `/embeddings` ендпоінтами, тому достатньо вказати `.../v1` як base URL.

Пояснення ключових параметрів:
- `TILE_SIZE`, `OVERLAP`: розмір тайла і перекриття.
- `GATE`: режим фільтрації (`heuristic|clip|cnn`).
- `CLIP_KEEP_PROMPTS`, `CLIP_DROP_PROMPTS`, `CLIP_GATE_THR`, `CLIP_MODEL`, `CLIP_PRETRAINED`, `CLIP_DEVICE`: промпти, поріг і модель CLIP для gate.
- `YOLO_MODEL`, `YOLO_CONF`: модель і поріг детекції.
- `MATCH_IOU`, `BUFFER_TOL`: пороги для change detection.
- `TOP_K`: кількість результатів пошуку.

### CLIP Gate: як писати промпти

- `CLIP_KEEP_PROMPTS` — що вважається "цікавим" тайлом, `CLIP_DROP_PROMPTS` — що відсікається.
- Формулювання: короткі конкретні фрази + вказівка "вид зверху/супутниковий знімок".
- Краще мати 10–30 варіацій і розділити їх `|` (CLIP gate бере максимум схожості серед keep/drop).
- Якщо працюєте українською, подумайте про multilingual CLIP або переклад промптів на англійську.
- Поріг `CLIP_GATE_THR`: чим вище — тим жорсткіша фільтрація (більше відсіву).

### CLIP (text → image) рекомендації, якщо будуєте індекс

Це опційно (в коді PoC CLIP використовується лише як gate), але для пошуку "текст → зображення" на супутникових тайлах:
- Тайли робіть на фіксованому MPP/zoom або на 2 масштабах (ближче + трохи далі).
- Використовуйте **ті самі** transforms від `open_clip` для всіх ембеддингів.
- Зберігайте метадані тайла: zoom, timestamp, source, gate_scores, cloud_mask тощо.
- Перед ембеддингами тримайте мʼякий `HeuristicGate`, щоб не витрачати ресурси.

Формули:
```text
img_vec = normalize(encode_image(tile))
text_vec = normalize(encode_text(prompt))
q = normalize(pos - 0.3 * neg)
```

Промпти:
- Додавайте "вид зверху / супутниковий знімок / дах / покрівля".
- Робіть ансамбль із 10–30 варіацій і усереднюйте їхні текстові ембеддинги.
- Негативні промпти (ліс, поле, вода, хмари) допомагають прибрати шум.

### Fine-tune OpenCLIP на своїх тайлах

Мінімальний скелет під CSV `image_path,text` є тут: `geovlm_poc/scripts/finetune_openclip.py`.
Старт:
```bash
python geovlm_poc/scripts/finetune_openclip.py --csv train.csv
```
Потрібні `open_clip_torch` і `torch`.

## Як користуватися (покроково)

### 1) Аналіз двох GeoTIFF

```bash
geovlm-poc analyze --a /path/dateA.tif --b /path/dateB.tif --out ./out_run
```

Що робить команда:
1. Нарізає обидва GeoTIFF на тайли.
2. Пропускає тайли через gate.
3. Запускає YOLO для bbox.
4. Запускає VLM для семантики (кешує результати).
5. Конвертує bbox у CRS-полігони та прибирає дублікати.
6. Порівнює дві дати й формує звіт змін.

Очікувані артефакти:
- `out_run/a.objects.jsonl`
- `out_run/b.objects.jsonl`
- `out_run/change_report.json`

### 1b) Аналіз однієї дати

```bash
geovlm-poc analyze-single --image /path/dateA.tif --out ./out_run
```

Очікувані артефакти:
- `out_run/objects.jsonl`

### 2) Побудова семантичного індексу

```bash
geovlm-poc build-index --objects ./out_run/a.objects.jsonl --out-index ./out_run/index_a --changes ./out_run/change_report.json --rail ./data/railways.geojson
```

Що робить команда:
- Формує текстові описи обʼєктів/змін.
- Рахує embeddings через OpenAI-сумісний endpoint.
- Будує індекс (FAISS, якщо встановлений; інакше numpy).
- За бажанням додає метадані `rail_dist_m` з шару залізниць.

### 3) Пошук

```bash
geovlm-poc search --index ./out_run/index_a --q "find warehouses with blue roofs near railway"
```

Виводить JSON-лінії з top-K результатами: score, тип (object/change), координати та короткий preview.

## Важливі примітки

- **CRS і відстані**: `rail_dist_m`, `BUFFER_TOL` та інші дистанції припускають метричну проекцію. Для EPSG:4326 дистанції не в метрах.
- **Вартість**: gate зменшує VLM-виклики; YOLO дає bbox; VLM лише додає семантику.
- **Кеш**: результати VLM кешуються за хешем тайла та моделі.

## Troubleshooting

- Якщо немає `ultralytics`, `open_clip_torch` або `faiss`, встановіть відповідні extras.
- Для `GATE=clip` потрібні `open_clip_torch` і `torch`.
- Для `GATE=cnn` потрібні `torch`, `torchvision` і чекпойнт `CNN_GATE_CKPT`.

## Рекомендовані налаштування для щільної міської забудови

- `TILE_SIZE=1024`, `OVERLAP=256`
- `MATCH_IOU=0.35`, `BUFFER_TOL=4.0`
- Gate: `GATE=heuristic|clip|cnn`
