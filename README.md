# D&D PodTrak Tool v3

Локальный пайплайн для многодорожечной записи D&D-сессии с PodTrak.

Цель: положить WAV-дорожки в папку, запустить одну команду и получить материал, с которым дальше можно работать: clean transcript, SRT, JSONL, чанки и промпты для ручного AI-анализа без API-токенов.

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
→ prompts/*.md для ручного AI-этапа
→ reports/*.md после ручных AI-ответов
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

## Ручной AI-этап без API

После `run` открой `prompts/*.md`, копируй промпт в ChatGPT/локальную LLM, а JSON-ответ сохраняй в:

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

Ручные AI-ответы теперь читаются терпимее: если ChatGPT вернул JSON с typographic quotes `“...”` или в markdown-блоке, `build-report` попробует это исправить автоматически.


## v5: фильтр галлюцинаций Whisper/MLX

После транскрибации инструмент дополнительно отбрасывает типичный мусор на тишине: `Спасибо.`, `Продолжение следует...`, `DimaTorzok`, субтитровые фразы, повтор одного слова и сегменты с аномальным `compression_ratio`/очень низким `rms_db`.

Отброшенные сегменты не теряются: они сохраняются в `raw/*_rejected_hallucinations.jsonl`. Если фильтр оказался слишком жёстким, можно ослабить настройки в блоке `hallucination_filter` в `config.json`.

Рекомендуемый цикл после грязного прогона v3/v4:

```bash
rm -rf ./_dnd_pipeline_out/dnd_2/cache
python3 dnd_pipeline.py run /path/to/session --config /path/to/session/config.json --limit-minutes 10 --quality-profile balanced
```

Проверить `clean/*_test_clean.txt` и `raw/*_test_rejected_hallucinations.jsonl`, затем запускать полный прогон.
