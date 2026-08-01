# AI-этап через API: дизайн

Дата: 2026-08-01. Статус: утверждён (brainstorming с Кареном).

## Цель

Убрать ручную копипасту prompts/*.md → чат → manual_ai_results/*.json.
Пайплайн сам прогоняет чанки через LLM-провайдера и складывает результаты
туда, где их ждёт `build_reports()`. Ручной режим остаётся запасным.

## Решения (зафиксированы с пользователем)

- Модель по умолчанию: `claude-sonnet-5`.
- Режим по умолчанию: Anthropic Batch API (−50% цены, результат ~час; ок,
  т.к. пайплайн гоняется после сессии, не интерактивно).
- Модель и провайдер переключаются конфигом; поддержка локальных провайдеров
  (Ollama, LM Studio, vLLM, OpenRouter) через OpenAI-compatible протокол.
- Ручной режим сохраняется: промпты генерируются всегда; при выключенном AI
  или отсутствии ключа поведение как сейчас.

## Место в пайплайне

```
make_chunks → make_prompts → run_ai_analysis (NEW) → build_reports
```

- `cmd_run`: после `make_prompts()` вызывается `run_ai_analysis()`, если
  `ai.enabled` и провайдер доступен. Дальше сразу `build_reports()` — полный
  прогон одной командой.
- Новая подкоманда `ai-analyze <session_dir>`: только AI-этап (+ отчёты),
  для повторных запусков/докачки упавших чанков.
- Результаты пишутся в `manual_ai_results/chunk_NNN_events.json` — формат и
  место не меняются, `read_manual_results()`/`build_reports()` не трогаем.
- Файл, положенный руками, не перезаписывается (см. идемпотентность).

## Конфиг

```json
"ai": {
  "enabled": true,
  "provider": "anthropic",
  "model": "claude-sonnet-5",
  "mode": "batch",
  "base_url": null,
  "api_key_env": null,
  "max_output_tokens": 8000,
  "concurrency": 2
}
```

- `provider`: `"anthropic"` | `"openai_compatible"`.
- `model`: строка, уходит провайдеру как есть.
- `mode` (только anthropic): `"batch"` | `"direct"`. Для `openai_compatible`
  игнорируется — всегда обычные запросы.
- `base_url` (только openai_compatible): `http://localhost:11434/v1` (Ollama),
  `http://localhost:1234/v1` (LM Studio), `https://openrouter.ai/api/v1`.
- `api_key_env`: имя env-переменной с ключом. Default: `ANTHROPIC_API_KEY`
  для anthropic; для openai_compatible ключ опционален (локальным не нужен).
  Ключи никогда не пишутся в конфиг/git/логи.
- `concurrency`: параллелизм для direct/openai_compatible режимов.

## Архитектура

```
run_ai_analysis(chunk_paths, cfg, paths)
  │  общий код: skip по кэшу, сборка промптов (prompt_for_chunk — тот же,
  │  что для ручного режима), раскладка результатов, лог прогресса
  ├─ AnthropicProvider
  │    SDK `anthropic`; Batch API (create → poll 30s → results)
  │    или direct-запросы; structured outputs (output_config.format
  │    c json_schema) → гарантированно валидный JSON
  └─ OpenAICompatProvider
       stdlib urllib (без новых зависимостей); POST /v1/chat/completions;
       response_format json_object где поддерживается; ответ через
       normalize_json_text() + проверка обязательных полей
```

Интерфейс провайдера: `analyze(jobs: list[ChunkJob]) -> dict[str, dict]`
(имя чанка → распарсенный результат). Всё вокруг — провайдеро-независимо.

JSON-схема результата повторяет формат из `prompt_for_chunk`: `session`,
`chunk_index`, `scene_type`, `actions[]`, `dice_rolls[]`, `mvp_signals[]`,
`summary`.

## Идемпотентность и кэш

- Результат чанка сопровождается записью в `out/cache/ai_state.json`:
  `{chunk_name: {chunk_hash, model, provider, status}}` (hash — `stable_hash`
  содержимого чанка, как в кэше транскрипции).
- Повторный запуск: чанк с существующим результатом и совпадающим хэшем —
  пропуск. Файл без записи в кэше (положен руками) — тоже пропуск.
- `--force-ai`: пересчитать всё (например, другой моделью).
- Batch-режим: `batch_id` сохраняется в `ai_state.json`. Прерванный скрипт →
  повторный запуск продолжает поллить существующий batch, не платит заново.
- Частичный успех: succeeded-чанки сохраняются сразу; errored/expired
  логируются и пересоздаются при следующем запуске.

## Ошибки

- Нет ключа / `enabled: false` → warning + прежнее сообщение про ручной режим;
  выход без ошибки.
- Ошибки API: ретраи силами SDK (anthropic) / 3 ретрая с бэкоффом (urllib);
  затем понятная ошибка с именем чанка.
- Битый JSON от локальной модели → warning, чанк остаётся без результата
  (виден в логе, можно добить повторным запуском или руками).
- Anthropic `stop_reason == "refusal"` → warning, чанк в список неудачных.

## Зависимости

- `anthropic` в requirements.txt (нужен только для provider=anthropic;
  импорт ленивый — openai_compatible работает без него).

## Тесты

Мок-тесты, без сетевых вызовов:
- сборка jobs: skip по хэшу, skip ручных файлов, --force-ai;
- разбор batch-результатов: succeeded/errored/расклад по файлам;
- OpenAICompatProvider: парсинг ответа, битый JSON → warning;
- fallback: нет ключа → warning, файлы не тронуты;
- возобновление поллинга по сохранённому batch_id.

## Вне скоупа

- Автогенерация financial/итоговых отчётов сверх текущего build_reports.
- Смена формата промпта/схемы событий.
- Параллельные batch'и нескольких сессий.
