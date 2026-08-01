"""LLM providers for the AI analysis stage.

Pure provider classes: no imports from dnd_pipeline (the orchestrator
imports us, not the other way around). Network code lives here; all
callers mock it in tests.
"""
from __future__ import annotations

import json
import logging
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
        required_fields: tuple[str, ...] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.concurrency = max(1, concurrency)
        self.normalize = normalize or (lambda t: t)
        self.timeout = timeout
        self.retries = retries
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

    def _analyze_one(self, job: ChunkJob) -> AIResult:
        payload = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
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
        for attempt in range(self.retries):
            req = urllib.request.Request(url, data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    last_err = e
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code} от {url}: {e.reason}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API не ответил после {self.retries} попыток: {last_err}")


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
