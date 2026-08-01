# D&D PodTrak Tool v3

Локальный пайплайн для многодорожечной записи D&D-сессии с PodTrak.

Цель: положить WAV-дорожки в папку, запустить одну команду и получить материал, с которым дальше можно работать: clean transcript, SRT, JSONL, чанки и готовые отчёты по AI-анализу.

## Что делает

```text
PodTrak WAV
→ ffmpeg preprocess без сдвига таймкодов
  → highpass
  → denoise
  → noise gate
  → mono 16 kHz
→ transcription backend
  → mlx-whisper на Apple Silicon, по умолчанию
  → faster-whisper как fallback
→ raw JSONL
→ дедупликация по времени + тексту + громкости
→ clean JSONL/TXT/SRT
→ chunks/*.json
→ AI-анализ через API (ai.enabled), провайдер anthropic или openai_compatible
  → ручной режим (prompts/*.md) как fallback, если API недоступен
→ reports/*.md
```

## Почему v3 быстрее на Mac M3 Max

По умолчанию теперь используется:

```json
"transcription_backend": "mlx",
"model_size": "mlx-community/whisper-large-v3-turbo"
```

`mlx-whisper` использует MLX, который предназначен для Apple Silicon. Для M3 Max это обычно практичнее, чем гонять `large-v3` через `faster-whisper` на CPU.

`faster-whisper` оставлен как fallback:

```json
"transcription_backend": "faster_whisper",
"model_size": "medium",
"device": "cpu",
"compute_type": "int8"
```

> **MLX (Apple Silicon):** `device` и `compute_type` применяются только к
> бэкенду faster-whisper. MLX их игнорирует и всегда работает fp16 на
> Metal GPU / Neural Engine. Качество декодирования на MLX контролируется
> блоком `decode` (`initial_prompt`, `condition_on_previous_text`,
> `compression_ratio_threshold`, `logprob_threshold`, `no_speech_threshold`,
> `hallucination_silence_threshold`), который задаются профилями качества по умолчанию.

## Установка

```bash
brew install ffmpeg

cd dnd_podtrack_tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Настройка

Скопируй конфиг в папку с WAV:

```bash
cp config.example.json /path/to/session/config.json
```

Проверь блок `tracks`:

```json
"tracks": [
  {"file": "dnd_2-Данжен Мастер.wav", "speaker": "Данжен Мастер", "character": "ДМ", "priority": 100},
  {"file": "dnd_2-Дима. Ангрон.wav", "speaker": "Дима", "character": "Ангрон", "priority": 50}
]
```

`priority` нужен при спорных дублях. У ДМа обычно можно поставить выше.

> **Авто-обнаружение треков:** если `tracks` в конфиге не задан, пайплайн сам
> находит все аудиофайлы в папке сессии (расширения из `audio_extensions`,
> по умолчанию `.wav`) — сколько бы их ни было. Имя файла парсится как
> `Спикер. Персонаж` (например `dnd_2-Дима. Ангрон.wav` → спикер «Дима»,
> персонаж «Ангрон»); имя без `. ` даёт `спикер == персонаж`. Трек, чей спикер
> равен `dm_speaker`, получает приоритет 100. Файл `master_mix` и записи из
> `exclude` пропускаются. Явный `tracks` (если задан) всегда побеждает.

## Быстрый тест на первых 10 минутах

Перед полным прогоном лучше не жечь время:

```bash
python3 dnd_pipeline.py run /path/to/session \
  --config /path/to/session/config.json \
  --limit-minutes 10
```

Тестовые файлы пишутся с суффиксом `_test`, чтобы не затирать полный результат:

```text
raw/dnd_2_test_raw.jsonl
clean/dnd_2_test_clean.jsonl
clean/dnd_2_test_clean.txt
clean/dnd_2_test_clean.srt
chunks/test_chunk_000.json
```

## Полный запуск

```bash
python3 dnd_pipeline.py run /path/to/session \
  --config /path/to/session/config.json
```

## Быстрая смена backend/model без редактирования config.json

MLX turbo:

```bash
python3 dnd_pipeline.py run /path/to/session \
  --config /path/to/session/config.json \
  --backend mlx \
  --model mlx-community/whisper-large-v3-turbo
```

MLX full large-v3:

```bash
python3 dnd_pipeline.py run /path/to/session \
  --config /path/to/session/config.json \
  --backend mlx \
  --model mlx-community/whisper-large-v3
```

Fallback через faster-whisper:

```bash
python3 dnd_pipeline.py run /path/to/session \
  --config /path/to/session/config.json \
  --backend faster_whisper \
  --model medium
```

## Важный момент про VAD

Для `faster-whisper` работает:

```json
"use_vad": true,
"vad_min_silence_ms": 800
```

Для `mlx-whisper` этот VAD не применяется. Поэтому в MLX-режиме основная очистка делается через `preprocess`: highpass, denoise и noise gate.

## Noise gate

Базовый мягкий вариант:

```json
"noise_gate_threshold_db": -45,
"noise_gate_ratio": 6
```

Если съедает тихие реплики:

```json
"noise_gate_threshold_db": -50,
"noise_gate_ratio": 4
```

Если чужие голоса слишком пролезают:

```json
"noise_gate_threshold_db": -40,
"noise_gate_ratio": 8
```

Можно переопределить для конкретной дорожки:

```json
{
  "file": "dnd_2-Дима. Ангрон.wav",
  "speaker": "Дима",
  "character": "Ангрон",
  "priority": 50,
  "preprocess": {
    "noise_gate_threshold_db": -42
  }
}
```

## Выходные файлы

```text
_dnd_pipeline_out/dnd_2/
  preprocessed/
  work/
  cache/
  raw/dnd_2_raw.jsonl
  clean/dnd_2_clean.jsonl
  clean/dnd_2_clean.txt
  clean/dnd_2_clean.srt
  chunks/chunk_000.json
  prompts/chunk_000_prompt.md
  manual_ai_results/
  reports/
```

Главный файл для дальнейшей работы:

```text
clean/dnd_2_clean.jsonl
```

Именно его лучше использовать для анализа действий, бросков и MVP.

## AI-этап через API

Если в `config.json` включено `ai.enabled`, `run` сам прогоняет чанки через LLM и сразу собирает отчёты — руками ничего копировать не нужно.

Два провайдера:

- `anthropic` — облачный Claude. По умолчанию работает через Batch API (`ai.mode: "batch"`, дешевле и не жмёт rate limit; можно переключить на `"direct"` для мгновенного прохода без батча). Ключ берётся из переменной окружения `ANTHROPIC_API_KEY` (или другой, если задать `ai.api_key_env`) — в `config.json` ключ никогда не хранится.
- `openai_compatible` — любой сервер с OpenAI-совместимым `/chat/completions`, например локальный Ollama или LM Studio. Обязательно нужен `ai.base_url` (например `http://localhost:11434/v1`).

Если API недоступен (ключа нет, `ai.enabled` выключен или провайдер вернул ошибку), пайплайн сам откатывается в ручной режим — см. ниже.

Повторный запуск или докачка недостающих чанков (без пересчёта уже готовых — они пропускаются по хэшу):

```bash
python3 dnd_pipeline.py ai-analyze /path/to/session \
  --config /path/to/session/config.json
```

Пересчитать всё заново (например, другой моделью после правки `ai.model`):

```bash
python3 dnd_pipeline.py ai-analyze /path/to/session \
  --config /path/to/session/config.json \
  --force
```

Результаты по каждому чанку складываются в `manual_ai_results/*_events.json` — в том же формате, что и при ручном режиме, — так что `build-report` работает одинаково для обоих путей.

### Конфиг-поля `ai.*`

| Поле | По умолчанию | Что делает |
| --- | --- | --- |
| `ai.enabled` | `false` | Включает автоматический AI-этап через API. Если `false` — сразу ручной режим. |
| `ai.provider` | `"anthropic"` | `"anthropic"` (облако) или `"openai_compatible"` (любой OpenAI-совместимый эндпоинт, локальный или удалённый). |
| `ai.model` | `"claude-sonnet-5"` | Имя модели у выбранного провайдера. |
| `ai.mode` | `"batch"` | Только для `anthropic`: `"batch"` (Batch API, дешевле, с задержкой) или `"direct"` (обычные запросы). |
| `ai.base_url` | `null` | Обязателен для `openai_compatible` — URL сервера, например `http://localhost:11434/v1`. |
| `ai.api_key_env` | `null` | Имя переменной окружения с ключом. Для `anthropic` по умолчанию `ANTHROPIC_API_KEY`; для `openai_compatible` по умолчанию ключ не требуется. |
| `ai.max_output_tokens` | `8000` | Лимит токенов на ответ модели по одному чанку. |
| `ai.concurrency` | `2` | Сколько чанков обрабатывать параллельно (для `direct`-режима/`openai_compatible`). |

## Ручной AI-этап без API (fallback)

Если `ai.enabled` выключен или API недоступен, `run` (и `ai-analyze`) сам откатывается в ручной режим: открой `prompts/*.md`, копируй промпт в ChatGPT/локальную LLM, а JSON-ответ сохраняй в:

```text
manual_ai_results/chunk_000_events.json
manual_ai_results/chunk_001_events.json
```

Потом собери отчёты:

```bash
python3 dnd_pipeline.py build-report /path/to/session \
  --config /path/to/session/config.json
```

Получишь:

```text
reports/actions_timeline.md
reports/dice_stats.md
reports/mvp_candidates.md
reports/session_report.md
```

## PDF-хроника

Поверх `manual_ai_results/*.json` можно собрать иллюстрированный PDF-отчёт сессии — обложка, оглавление, MVP, кости, таймлайн действий, состав партии и AI-сгенерированные сцены. Оформление в June-стиле: тёмная обложка, кремовые внутренние страницы, бордо/золото.

Воркфлоу:

1. `run` уже сам вызывает сборку PDF в конце (после AI-этапа, если он прошёл через API) — если `manual_ai_results/` не пуст, PDF соберётся автоматически. Заодно на этом шаге считается «синтез сессии» (recap, quest hooks, промпты для картинок) и пишется `_dnd_pipeline_out/<session_name>/image_prompts.md`. Обрати внимание: `ai-analyze` (ручной перезапуск AI-этапа) PDF не собирает — только `run` и `build-pdf`.
2. Открой `_dnd_pipeline_out/<session_name>/image_prompts.md`, скорми промпты Midjourney (или любому image-генератору).
3. Сохрани картинки в `report_assets/` рядом с сессией под именами `scene1.png`, `scene2.jpg`, … (номер = порядок сцены в `image_prompts.md`; расширение `.png`/`.jpg`/`.jpeg` любое).
4. Пересобери PDF, чтобы вставить картинки:

```bash
python3 dnd_pipeline.py build-pdf /path/to/session \
  --config /path/to/session/config.json
```

Если сцен ещё нет — PDF всё равно соберётся, просто без картинок; в логе будет подсказка, куда их положить.

### `party.json` — состав партии

Необязательный файл `report_assets/party.json` со списком персонажей — попадает на страницу «Партия» и используется как контекст для AI-синтеза (описание персонажей на английском помогает генерировать более точные промпты для сцен):

```json
[
  {
    "name": "Ангрон",
    "class_ru": "Варвар-полуорк",
    "player": "Дима",
    "ref": "angron_ref.jpg",
    "appearance_en": "hulking scarred half-orc barbarian, tusks, war paint"
  }
]
```

- `name` — обязателен, записи без него отбрасываются;
- `class_ru` и `player` — опциональны, выводятся под портретом;
- `ref` — имя файла-референса (портрет/арт) в той же папке `report_assets/`, вставляется в PDF, если файл существует;
- `appearance_en` — описание внешности на английском, идёт в промпт AI-синтеза, чтобы сцены рисовались с узнаваемыми персонажами.

### Конфиг-поля `pdf.*`

| Поле | По умолчанию | Что делает |
| --- | --- | --- |
| `pdf.enabled` | `true` | Собирать ли PDF (после `run` и в `build-pdf`; `ai-analyze` PDF не собирает). `false` — пропустить (в т.ч. в `build-pdf` — команда сразу выйдет с предупреждением). |
| `pdf.assets_dir` | `null` | Папка со сценами/`party.json`. По умолчанию — `report_assets/` рядом с сессией. |
| `pdf.campaign_title` | `"Хроники кампании"` | Заголовок кампании в подвале страниц. |
| `pdf.subtitle` | `"D&D · Forgotten Realms"` | Подзаголовок на обложке. |

Отдельно: `build-pdf` поддерживает `--force-synthesis` (пересчитать recap/сцены заново, даже если есть кэш) и `--quality-profile` (как у остальных команд).

Результат: `_dnd_pipeline_out/<session_name>/Session_<session_name>_Report.pdf`.

> **Переносимость.** Рендер PDF идёт через пакет `typst` (`pip install typst`) — это чистый pip-wheel без системных зависимостей (никаких Chromium/WeasyPrint/LaTeX). Если пакет не установлен, `build-pdf` не падает, а пишет предупреждение с подсказкой `pip install typst` и завершает работу с кодом 0.

## Кэш

Кэшируется каждая дорожка отдельно. Если ты меняешь дедупликацию, промпты или отчёты, транскрипция заново не запускается. Если меняешь модель, backend, preprocess или limit — создаётся новый кэш.


## v4: повышение качества

Добавлены профили качества:

```bash
python3 dnd_pipeline.py run /path/to/session --config /path/to/config.json --limit-minutes 10 --quality-profile gentle
python3 dnd_pipeline.py run /path/to/session --config /path/to/config.json --limit-minutes 10 --quality-profile balanced
python3 dnd_pipeline.py run /path/to/session --config /path/to/config.json --limit-minutes 10 --quality-profile aggressive
```

Как выбирать:

- `gentle` — если пропадают тихие реплики. Меньше давит noise gate, осторожнее дедуплит.
- `balanced` — основной режим.
- `aggressive` — если слишком много дублей и чужих голосов в микрофонах. Может съесть тихую речь.

После каждого `run` создаётся диагностический отчёт:

```text
_dnd_pipeline_out/<session>/reports/quality_report_test.md
_dnd_pipeline_out/<session>/reports/quality_report.md
```

Смотри там:

- raw/clean количество реплик;
- сколько дублей склеено;
- сколько реплик на каждого персонажа;
- подозрительные сегменты по метрикам Whisper/MLX.

Также добавлен postprocess `merge_adjacent_same_speaker`, который склеивает короткие соседние куски одного персонажа, чтобы clean.txt и AI-промпты были читабельнее.

Ручные AI-ответы теперь читаются терпимее: если ChatGPT вернул JSON с typographic quotes `”...”` или в markdown-блоке, `build-report` попробует это исправить автоматически.

## Подробность вывода

Доступны на всех подкомандах (`run`, `rebuild`, `prepare-ai`, `ai-analyze`, `build-report`):

- `-v`, `--verbose` — показывает построчный транскрипт и debug-детали
- `-q`, `--quiet` — только предупреждения и ошибки (тихий режим для регулярных прогонов)
- по умолчанию — прогресс по стадиям, но БЕЗ построчного транскрипта (он теперь только под `--verbose`)

Флаги взаимоисключающие.


## v5: фильтр галлюцинаций Whisper/MLX

После транскрибации инструмент дополнительно отбрасывает типичный мусор на тишине: `Спасибо.`, `Продолжение следует...`, `DimaTorzok`, субтитровые фразы, повтор одного слова и сегменты с аномальным `compression_ratio`/очень низким `rms_db`.

Отброшенные сегменты не теряются: они сохраняются в `raw/*_rejected_hallucinations.jsonl`. Если фильтр оказался слишком жёстким, можно ослабить настройки в блоке `hallucination_filter` в `config.json`.

Рекомендуемый цикл после грязного прогона v3/v4:

```bash
rm -rf ./_dnd_pipeline_out/dnd_2/cache
python3 dnd_pipeline.py run /path/to/session --config /path/to/session/config.json --limit-minutes 10 --quality-profile balanced
```

Проверить `clean/*_test_clean.txt` и `raw/*_test_rejected_hallucinations.jsonl`, затем запускать полный прогон.
