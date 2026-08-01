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
                    "die": {"type": ["string", "null"]},
                    "natural": {"type": ["integer", "null"]},
                    "modifier": {"type": ["integer", "null"]},
                    "total": {"type": ["integer", "null"]},
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


def parse_model_json(name: str, text: str, normalize: Callable[[str], str]) -> AIResult:
    """Parse a model's text answer into a result dict; soft-fail on garbage."""
    try:
        data = json.loads(normalize(text))
    except ValueError as e:
        return AIResult(name=name, error=f"битый JSON от модели: {e}")
    if not isinstance(data, dict):
        return AIResult(name=name, error="ответ модели — не JSON-объект")
    missing = [k for k in REQUIRED_RESULT_FIELDS if k not in data]
    if missing:
        return AIResult(name=name, error=f"в ответе нет полей: {', '.join(missing)}")
    return AIResult(name=name, data=data)
