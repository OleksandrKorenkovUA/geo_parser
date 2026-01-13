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
pip install -e ".[ui]"
```

## Налаштування оточення

Скопіюйте приклад та заповніть значення:

```bash
cp geovlm_poc/.env.example geovlm_poc/.env
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

### Сценарій встановлення та застосування (покроково)

1) **Створіть Python-оточення і встановіть залежності**

Мінімально:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

З опційними модулями (YOLO/CLIP/CNN/FAISS/UI):
```bash
pip install -e ".[yolo,clip,cnn,faiss,ui]"
```

2) **Налаштуйте змінні оточення**

```bash
cp geovlm_poc/.env.example geovlm_poc/.env
```

Відредагуйте `geovlm_poc/.env` і підвантажте:
```bash
set -a
source geovlm_poc/.env
set +a
```

Порада: якщо VLM/Embeddings не вимагають авторизації, задайте `VLM_API_KEY=DUMMY_KEY` і `EMB_API_KEY=DUMMY_KEY`.

3) **Запустіть аналіз (дві дати)**

```bash
geovlm-poc analyze --a /data/dateA.tif --b /data/dateB.tif --out ./out_run
```

Очікувані артефакти:
- `./out_run/a.objects.jsonl`
- `./out_run/b.objects.jsonl`
- `./out_run/change_report.json`

4) **Запустіть аналіз однієї дати (якщо треба)**

```bash
geovlm-poc analyze-single --image /data/dateA.tif --out ./out_run_single
```

Артефакт:
- `./out_run_single/objects.jsonl`

5) **Побудуйте семантичний індекс**

```bash
geovlm-poc build-index --objects ./out_run/a.objects.jsonl --out-index ./out_run/index_a --changes ./out_run/change_report.json
```

6) **Виконайте пошук**

```bash
geovlm-poc search --index ./out_run/index_a --q "find warehouses with blue roofs near railway"
```

### Варіанти використання (корисні комбінації)

**Gate**

- Вимкнути gate (для дебагу або максимальної повноти):
```bash
GATE=none geovlm-poc analyze --a /data/dateA.tif --b /data/dateB.tif --out ./out_run
```

- CLIP gate з власними промптами:
```bash
GATE=clip CLIP_DEVICE=cuda CLIP_GATE_THR=0.18 \
CLIP_KEEP_PROMPTS="dense urban area|industrial site|parking lot" \
CLIP_DROP_PROMPTS="forest|water|clouds" \
geovlm-poc analyze --a /data/dateA.tif --b /data/dateB.tif --out ./out_run
```

- CNN gate (вкажіть чекпойнт):
```bash
GATE=cnn CNN_GATE_CKPT=/models/cnn_gate.pt CNN_GATE_THR=0.6 CNN_DEVICE=cuda \
geovlm-poc analyze --a /data/dateA.tif --b /data/dateB.tif --out ./out_run
```

**YOLO**

- Інша модель + нижчий поріг:
```bash
YOLO_MODEL=yolo12s.pt YOLO_CONF=0.15 YOLO_DEVICE=cuda \
geovlm-poc analyze --a /data/dateA.tif --b /data/dateB.tif --out ./out_run
```

**Тайли**

- Змінити розмір/перекриття (впливає на баланс швидкість/точність):
```bash
TILE_SIZE=768 OVERLAP=192 \
geovlm-poc analyze --a /data/dateA.tif --b /data/dateB.tif --out ./out_run
```

**Індекс**

- Додати залізничний шар для фільтрів:
```bash
geovlm-poc build-index --objects ./out_run/a.objects.jsonl --out-index ./out_run/index_a \
  --changes ./out_run/change_report.json --rail ./data/railways.geojson
```

## Навчання CNN/CLIP gate та підготовка датасетів

Окремий гайд з прикладами команд знаходиться тут: `geovlm_poc/README_TRAINING.md`.

Пояснення ключових параметрів:
- `TILE_SIZE`, `OVERLAP`: розмір тайла і перекриття.
- `GATE`: режим фільтрації (`heuristic|clip|cnn`).
- `CLIP_KEEP_PROMPTS`, `CLIP_DROP_PROMPTS`, `CLIP_GATE_THR`, `CLIP_MODEL`, `CLIP_PRETRAINED`, `CLIP_DEVICE`: промпти, поріг і модель CLIP для gate.
- `YOLO_MODEL`, `YOLO_CONF`: модель і поріг детекції.
- `DETECTOR_WORKERS`, `VLM_WORKERS`: паралельність стадій детектора/VLM у пайплайні.
- `MATCH_IOU`, `BUFFER_TOL`: пороги для change detection.
- `TOP_K`: кількість результатів пошуку.
- `VLM_MODE`: режим structured output для VLM (`free_text|json|choice`).
- `VLM_JSON_SCHEMA`: JSON-схема для `VLM_MODE=json` (якщо не задано — використовується вбудована).
- `VLM_CHOICE_LIST`: список дозволених відповідей для `VLM_MODE=choice` (розділювач `|` або `,`).
- `VLM_LOG_PAYLOAD`: `1/true` щоб логувати фактичний payload (картинка редагується).
- `VLM_BACKEND`: формат multimodal payload (`openai|vllm_qwen`).
- `VLM_MAX_IMAGE_SIDE`: максимальна сторона зображення для VLM (0 = без зменшення).
- `VLM_TIMEOUT_READ`: read timeout для VLM запиту (секунди).
- `VLM_TIMEOUT_CONNECT`, `VLM_TIMEOUT_WRITE`, `VLM_TIMEOUT_POOL`: розширені таймаути httpx (секунди).
- `VLM_IMAGE_MODE`: спосіб передачі зображень у VLM (`base64|url|path`).
- `VLM_IMAGE_URL_PREFIX`, `VLM_IMAGE_URL_ROOT`: префікс/корінь для побудови URL, якщо `VLM_IMAGE_MODE=url`.
- `CACHE_KEY_MODE`: режим ключа кешу (`tile|pixels|source`).
- `ALLOW_EMPTY_VLM`: якщо `1`, VLM запускається навіть коли `YOLO` не дав детекцій.
- Примітка: якщо `ALLOW_EMPTY_VLM=1` і є ризик зміни детекцій між запуском, встановіть `CACHE_KEY_MODE=source`, щоб кеш враховував dets.

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

## UI (Streamlit)

Швидкий запуск UI:

```bash
pip install -e ".[ui]"
streamlit run geovlm_poc/streamlit_app.py
```

UI підтримує три сценарії: Analyze (запуск аналізу з тайлінгом, gate/YOLO/VLM, кешем, артефактами), Search (український пошук із фільтрами та мапою), Compare (change detection між двома запуском).

## Важливі примітки

- **CRS і відстані**: `rail_dist_m`, `BUFFER_TOL` та інші дистанції припускають метричну проекцію. Для EPSG:4326 дистанції не в метрах.
- **Вартість**: gate зменшує VLM-виклики; YOLO дає bbox; VLM лише додає семантику.
- **Кеш**: результати VLM кешуються за хешем тайла та моделі.

## Troubleshooting

- Якщо немає `ultralytics`, `open_clip_torch` або `faiss`, встановіть відповідні extras.
- Для `GATE=clip` потрібні `open_clip_torch` і `torch`.
- Для `GATE=cnn` потрібні `torch`, `torchvision` і чекпойнт `CNN_GATE_CKPT`.
- `400 Bad Request` із повідомленням про `structured outputs` означає, що одночасно передано кілька взаємовиключних режимів (`json|regex|choice`). Встановіть **один** режим через `VLM_MODE` або вимкніть structured output (`VLM_MODE=free_text`).
- Для діагностики увімкніть логування payload: `VLM_LOG_PAYLOAD=1` і перевірте, що у запиті є лише один constraint.
- Для vLLM + Qwen3‑VL використовуйте `VLM_BACKEND=vllm_qwen` — зображення передається через `mm_processor_kwargs.images`, а не як `image_url` у `messages`.
- `httpx.ReadError` під навантаженням означає backpressure/overload. Рекомендації: `VLM_CONCURRENCY=1`, `VLM_MAX_IMAGE_SIDE=512|768`, `VLM_TIMEOUT_READ=120`.
- Якщо `VLM_IMAGE_MODE=url|path`, пайплайн автоматично вмикає збереження PNG тайлів (потрібно, щоб VLM мав доступ до файлів/URL).

## Рекомендовані налаштування для щільної міської забудови

- `TILE_SIZE=1024`, `OVERLAP=256`
- `MATCH_IOU=0.35`, `BUFFER_TOL=4.0`
- Gate: `GATE=heuristic|clip|cnn`
