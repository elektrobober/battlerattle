"""LLM providers for the AI analysis stage.

Pure provider classes: no imports from dnd_pipeline (the orchestrator
imports us, not the other way around). Network code lives here; all
callers mock it in tests.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("dnd_pipeline")

REQUIRED_RESULT_FIELDS = ("actions", "dice_rolls", "mvp_signals", "summary")


@dataclass
class ChunkJob:
    name: str          # chunk file stem, e.g. "chunk_000"
    prompt: str
    chunk_hash: str


@dataclass
class AIResult:
    name: str
    data: dict[str, Any] | None = None
    error: str | None = None


# JSON schema mirrors the format demanded by prompt_for_chunk().
# Structured outputs require additionalProperties: false on every object.
EVENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["session", "chunk_index", "scene_type", "actions", "dice_rolls", "mvp_signals", "summary"],
    "properties": {
        "session": {"type": "string"},
        "chunk_index": {"type": "integer"},
        "scene_type": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time", "character", "action", "outcome", "importance"],
                "properties": {
                    "time": {"type": "string"},
                    "character": {"type": "string"},
                    "action": {"type": "string"},
                    "outcome": {"type": "string"},
                    "importance": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        },
        "dice_rolls": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time", "character", "roll_type", "die", "natural",
                             "modifier", "total", "context", "confidence", "raw_text"],
                "properties": {
                    "time": {"type": "string"},
                    "character": {"type": "string"},
                    "roll_type": {"type": "string"},
                    "die": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "natural": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "modifier": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "total": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "context": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "raw_text": {"type": "string"},
                },
            },
        },
        "mvp_signals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time", "character", "category", "reason", "weight"],
                "properties": {
                    "time": {"type": "string"},
                    "character": {"type": "string"},
                    "category": {"type": "string"},
                    "reason": {"type": "string"},
                    "weight": {"type": "integer"},
                },
            },
        },
        "summary": {"type": "string"},
    },
}

SYNTHESIS_REQUIRED_FIELDS = ("recap", "quest_hooks", "scenes")

# Схема сессионного синтеза: рекап, зацепки, ключевые сцены с промптами.
SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["recap", "quest_hooks", "scenes"],
    "properties": {
        "recap": {"type": "string"},
        "quest_hooks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "description"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "chunk_index", "time", "image_prompt"],
                "properties": {
                    "title": {"type": "string"},
                    "chunk_index": {"type": "integer"},
                    "time": {"type": "string"},
                    "image_prompt": {"type": "string"},
                },
            },
        },
    },
}


def parse_model_json(name: str, text: str, normalize: Callable[[str], str], required_fields: tuple[str, ...] | None = None) -> AIResult:
    """Parse a model's text answer into a result dict; soft-fail on garbage."""
    if required_fields is None:
        required_fields = REQUIRED_RESULT_FIELDS
    try:
        data = json.loads(normalize(text))
    except ValueError as e:
        return AIResult(name=name, error=f"битый JSON от модели: {e}")
    if not isinstance(data, dict):
        return AIResult(name=name, error="ответ модели — не JSON-объект")
    missing = [k for k in required_fields if k not in data]
    if missing:
        return AIResult(name=name, error=f"в ответе нет полей: {', '.join(missing)}")
    return AIResult(name=name, data=data)


# Потолок ожидания одного 429: провайдер может назвать окно в минутах,
# но висеть на одном чанке дольше трёх минут смысла нет — проще перепроверить.
RATE_LIMIT_WAIT_CAP = 180
RATE_LIMIT_WAIT_BASE = 25
# Подсказка провайдера короче секунды — не подсказка, а busy-loop.
RATE_LIMIT_WAIT_FLOOR = 1

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_reset_duration(value: str | None) -> float | None:
    """Go-style длительность из заголовков сброса лимита: '6m0s', '1.5s', '20ms'."""
    if not value:
        return None
    total = 0.0
    matched = False
    for amount, unit in _DURATION_RE.findall(value.strip()):
        matched = True
        total += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if matched else None


class OpenAICompatProvider:
    """Any OpenAI-compatible /v1/chat/completions endpoint: Ollama, LM Studio, vLLM, OpenRouter, OpenAI."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        max_output_tokens: int = 8000,
        concurrency: int = 2,
        normalize: Callable[[str], str] | None = None,
        timeout: int = 600,
        retries: int = 3,
        rate_limit_retries: int = 8,
        required_fields: tuple[str, ...] | None = None,
        schema: dict[str, Any] | None = None,
        schema_name: str = "result",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.concurrency = max(1, concurrency)
        self.normalize = normalize or (lambda t: t)
        self.timeout = timeout
        self.retries = retries
        self.rate_limit_retries = max(1, rate_limit_retries)
        self.schema = schema
        self.schema_name = schema_name
        self.required_fields = required_fields or REQUIRED_RESULT_FIELDS

    def analyze(
        self,
        jobs: list[ChunkJob],
        on_result: Callable[[AIResult], None] | None = None,
    ) -> list[AIResult]:
        results: list[AIResult] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for res in pool.map(self._analyze_one, jobs):
                if on_result:
                    on_result(res)
                results.append(res)
        return results

    def _response_format(self) -> dict[str, Any]:
        """strict json_schema там, где он есть; иначе — просто «верни валидный JSON».

        Без схемы провайдер гарантирует только синтаксис: поле recap так
        приезжало массивом абзацев вместо строки, и Typst печатал repr.
        """
        if self.schema is None:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {"name": self.schema_name, "strict": True, "schema": self.schema},
        }

    def _analyze_one(self, job: ChunkJob) -> AIResult:
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "response_format": self._response_format(),
            "messages": [{"role": "user", "content": job.prompt}],
        }
        try:
            raw = self._post_with_retries(payload)
        except Exception as e:  # noqa: BLE001 — per-chunk soft fail, summary logged by caller
            return AIResult(name=job.name, error=str(e))
        return parse_model_json(job.name, raw, self.normalize, required_fields=self.required_fields)

    def _post_with_retries(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err: Exception | None = None
        # Бюджеты раздельные: 429 — это не сбой, а ожидание окна TPM,
        # и он не должен съедать попытки, отведённые под 5xx и обрывы сети.
        net_attempt = 0
        rate_limit_attempt = 0
        while True:
            req = urllib.request.Request(url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    last_err = e
                    rate_limit_attempt += 1
                    if rate_limit_attempt >= self.rate_limit_retries:
                        raise RuntimeError(
                            f"лимит запросов (429): окно не открылось за "
                            f"{rate_limit_attempt} попыток"
                        ) from e
                    delay = self._rate_limit_delay(e.headers, rate_limit_attempt)
                    logger.info(
                        f"Rate limit (429), жду {delay:.0f}с и повторяю "
                        f"({rate_limit_attempt}/{self.rate_limit_retries})…"
                    )
                    time.sleep(delay)
                    continue
                if e.code >= 500:
                    last_err = e
                    net_attempt += 1
                    if net_attempt >= self.retries:
                        break
                    time.sleep(2 ** (net_attempt - 1))
                    continue
                raise RuntimeError(f"HTTP {e.code} от {url}: {e.reason}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                net_attempt += 1
                if net_attempt >= self.retries:
                    break
                time.sleep(2 ** (net_attempt - 1))
        raise RuntimeError(f"API не ответил после {self.retries} попыток: {last_err}")

    def _rate_limit_delay(self, headers: Any, attempt: int) -> float:
        """Сколько ждать до следующей попытки: слово провайдера важнее нашей догадки."""
        lookup = {k.lower(): v for k, v in dict(headers or {}).items()}
        hinted: float | None = None
        retry_after = lookup.get("retry-after")
        if retry_after:
            try:
                hinted = float(retry_after)
            except (TypeError, ValueError):
                hinted = None
        if hinted is None:
            for header in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
                hinted = parse_reset_duration(lookup.get(header))
                if hinted is not None:
                    break
        # Подсказку меньше секунды игнорируем: раз 429 всё-таки прилетел, окно
        # закрыто дольше, чем говорит заголовок (или запрос крупнее всего лимита).
        if hinted is not None and hinted >= RATE_LIMIT_WAIT_FLOOR:
            return min(hinted, RATE_LIMIT_WAIT_CAP)
        # Без подсказок — растущая пауза: первое окно минутное, дальше провайдер
        # мог зажать сильнее, чем на минуту.
        return min(RATE_LIMIT_WAIT_BASE * 1.5 ** (attempt - 1), RATE_LIMIT_WAIT_CAP)


# Очередь Batch API считается отдельно от TPM. У tier-1 потолок enqueued-токенов
# 900k — режем пачки с запасом, иначе создание батча падает целиком.
BATCH_TOKEN_BUDGET = 700_000
# Грубая оценка для русского текста: ~3 символа на токен.
CHARS_PER_TOKEN = 3


class OpenAIBatchProvider:
    """OpenAI Batch API: отдельная очередь мимо TPM, -50% цены, окно 24 часа.

    Спасает там, где синхронный вызов бесполезен: чанк на 50k токенов
    физически не влезает в минутный лимит tier-1 в 30k TPM.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        max_output_tokens: int = 8000,
        normalize: Callable[[str], str] | None = None,
        timeout: int = 600,
        poll_interval: int = 60,
        token_budget: int = BATCH_TOKEN_BUDGET,
        required_fields: tuple[str, ...] | None = None,
        schema: dict[str, Any] | None = None,
        schema_name: str = "result",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.normalize = normalize or (lambda t: t)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.token_budget = max(1, token_budget)
        self.schema = schema
        self.schema_name = schema_name
        self.required_fields = required_fields or REQUIRED_RESULT_FIELDS

    # ── публичный интерфейс, одинаковый с остальными провайдерами ──
    def analyze(
        self,
        jobs: list[ChunkJob],
        on_result: Callable[[AIResult], None] | None = None,
        resume_batch_id: str | None = None,
        on_batch_created: Callable[[str], None] | None = None,
    ) -> list[AIResult]:
        if resume_batch_id:
            logger.info(f"Batch {resume_batch_id}: продолжаю ожидание.")
            return self._collect(resume_batch_id, on_result, names=None)

        results: list[AIResult] = []
        groups = self._split_by_budget(jobs)
        for i, group in enumerate(groups, 1):
            file_id = self._upload_jsonl(group)
            batch_id = self._create_batch(file_id)
            if on_batch_created:
                on_batch_created(batch_id)
            suffix = f" (пачка {i} из {len(groups)})" if len(groups) > 1 else ""
            logger.info(
                f"Batch создан: {batch_id}, чанков: {len(group)}{suffix}. "
                f"Окно 24 часа, обычно быстрее; прервёшь — следующий запуск дождётся его же."
            )
            results.extend(self._collect(batch_id, on_result, names=[j.name for j in group]))
        return results

    # ── шаги ──
    def _split_by_budget(self, jobs: list[ChunkJob]) -> list[list[ChunkJob]]:
        groups: list[list[ChunkJob]] = []
        current: list[ChunkJob] = []
        spent = 0
        for job in jobs:
            cost = len(job.prompt) // CHARS_PER_TOKEN + self.max_output_tokens
            if current and spent + cost > self.token_budget:
                groups.append(current)
                current, spent = [], 0
            current.append(job)
            spent += cost
        if current:
            groups.append(current)
        return groups

    def _request_body(self, job: ChunkJob) -> dict[str, Any]:
        response_format: dict[str, Any] = {"type": "json_object"}
        if self.schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": self.schema_name, "strict": True, "schema": self.schema},
            }
        return {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "response_format": response_format,
            "messages": [{"role": "user", "content": job.prompt}],
        }

    def _upload_jsonl(self, jobs: list[ChunkJob]) -> str:
        lines = [
            json.dumps({
                "custom_id": j.name,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": self._request_body(j),
            }, ensure_ascii=False)
            for j in jobs
        ]
        payload = "\n".join(lines).encode("utf-8")
        boundary = "----dndpipeline-batch-boundary"
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="batch.jsonl"\r\n'
            f"Content-Type: application/jsonl\r\n\r\n"
        ).encode("utf-8")
        body = head + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
        data = self._api("POST", "/files", body=body,
                         content_type=f"multipart/form-data; boundary={boundary}")
        return data["id"]

    def _create_batch(self, file_id: str) -> str:
        data = self._api("POST", "/batches", body=json.dumps({
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }).encode("utf-8"))
        return data["id"]

    def _collect(
        self,
        batch_id: str,
        on_result: Callable[[AIResult], None] | None,
        names: list[str] | None,
    ) -> list[AIResult]:
        status = self._wait(batch_id)
        if status.get("status") != "completed":
            state = status.get("status")
            error = f"batch {batch_id}: статус {state}"
            logger.warning(f"Batch {batch_id} завершился со статусом {state}.")
            results = [AIResult(name=n, error=error) for n in (names or [])]
            for res in results:
                if on_result:
                    on_result(res)
            return results

        results = []
        for file_id in (status.get("output_file_id"), status.get("error_file_id")):
            if not file_id:
                continue
            for line in self._download(file_id).splitlines():
                if not line.strip():
                    continue
                res = self._result_from_line(json.loads(line))
                if on_result:
                    on_result(res)
                results.append(res)
        return results

    def _wait(self, batch_id: str) -> dict[str, Any]:
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            status = self._api("GET", f"/batches/{batch_id}")
            if status.get("status") in terminal:
                return status
            counts = status.get("request_counts") or {}
            logger.info(
                f"Batch {batch_id}: {status.get('status')}, готово "
                f"{counts.get('completed', 0)}/{counts.get('total', 0)}…"
            )
            time.sleep(self.poll_interval)

    def _result_from_line(self, line: dict[str, Any]) -> AIResult:
        name = line.get("custom_id") or "?"
        error = line.get("error")
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            return AIResult(name=name, error=f"batch: {message}")
        response = line.get("response") or {}
        if response.get("status_code") != 200:
            return AIResult(name=name, error=f"batch: HTTP {response.get('status_code')}")
        try:
            content = response["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            return AIResult(name=name, error=f"batch: неожиданный ответ ({e})")
        return parse_model_json(name, content, self.normalize, required_fields=self.required_fields)

    # ── HTTP ──
    def _api(self, method: str, path: str, body: bytes | None = None,
             content_type: str = "application/json") -> dict[str, Any]:
        return json.loads(self._http(method, path, body, content_type))

    def _download(self, file_id: str) -> str:
        return self._http("GET", f"/files/{file_id}/content", None, "application/json")

    def _http(self, method: str, path: str, body: bytes | None, content_type: str) -> str:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    # Опрос статуса идёт часами — единичный сбой не повод ронять ожидание.
                    last_err = e
                    time.sleep(5 * (attempt + 1))
                    continue
                detail = e.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"HTTP {e.code} от {url}: {detail or e.reason}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Batch API не ответил: {last_err}")


class AnthropicProvider:
    """Anthropic Messages API: Batch mode (default, -50% cost) or direct requests."""

    def __init__(
        self,
        model: str,
        api_key: str,
        mode: str = "batch",
        max_output_tokens: int = 8000,
        concurrency: int = 2,
        poll_interval: int = 30,
        client: Any = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.mode = mode
        self.max_output_tokens = max_output_tokens
        self.concurrency = max(1, concurrency)
        self.poll_interval = poll_interval
        self.client = client  # tests inject a stub; real client is created lazily
        self.schema = schema or EVENTS_SCHEMA

    def _get_client(self) -> Any:
        if self.client is None:
            import anthropic  # lazy: openai_compatible works without the package

            self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client

    def analyze(
        self,
        jobs: list[ChunkJob],
        on_result: Callable[[AIResult], None] | None = None,
        resume_batch_id: str | None = None,
        on_batch_created: Callable[[str], None] | None = None,
    ) -> list[AIResult]:
        if self.mode == "batch":
            return self._analyze_batch(jobs, on_result, resume_batch_id, on_batch_created)
        return self._analyze_direct(jobs, on_result)

    def _request_params(self, job: ChunkJob) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "output_config": {"format": {"type": "json_schema", "schema": self.schema}},
            "messages": [{"role": "user", "content": job.prompt}],
        }

    def _result_from_message(self, name: str, msg: Any) -> AIResult:
        if msg.stop_reason == "refusal":
            return AIResult(name=name, error="модель отказалась отвечать (refusal)")
        if msg.stop_reason == "max_tokens":
            return AIResult(name=name, error="ответ обрезан (max_tokens); подними ai.max_output_tokens")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            return AIResult(name=name, data=json.loads(text))
        except ValueError as e:
            return AIResult(name=name, error=f"битый JSON: {e}")

    # ── direct ──
    def _analyze_direct(
        self, jobs: list[ChunkJob], on_result: Callable[[AIResult], None] | None
    ) -> list[AIResult]:
        results: list[AIResult] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for res in pool.map(self._analyze_one_direct, jobs):
                if on_result:
                    on_result(res)
                results.append(res)
        return results

    def _analyze_one_direct(self, job: ChunkJob) -> AIResult:
        try:
            msg = self._get_client().messages.create(**self._request_params(job))
        except Exception as e:  # noqa: BLE001 — per-chunk soft fail, summary logged by caller
            return AIResult(name=job.name, error=str(e))
        return self._result_from_message(job.name, msg)

    # ── batch ──
    def _analyze_batch(
        self,
        jobs: list[ChunkJob],
        on_result: Callable[[AIResult], None] | None,
        resume_batch_id: str | None,
        on_batch_created: Callable[[str], None] | None,
    ) -> list[AIResult]:
        client = self._get_client()
        batch_id = resume_batch_id
        if batch_id is None:
            batch = client.messages.batches.create(
                requests=[{"custom_id": j.name, "params": self._request_params(j)} for j in jobs]
            )
            batch_id = batch.id
            if on_batch_created:
                on_batch_created(batch_id)
            logger.info(f"Batch создан: {batch_id} ({len(jobs)} чанков). Обычно готов в течение часа.")
        while True:
            status = client.messages.batches.retrieve(batch_id)
            if status.processing_status == "ended":
                break
            logger.info(f"Batch {batch_id}: ждём, в обработке {status.request_counts.processing}…")
            time.sleep(self.poll_interval)
        results: list[AIResult] = []
        for item in client.messages.batches.results(batch_id):
            if item.result.type == "succeeded":
                res = self._result_from_message(item.custom_id, item.result.message)
            else:
                res = AIResult(name=item.custom_id, error=f"batch result: {item.result.type}")
            if on_result:
                on_result(res)
            results.append(res)
        return results
