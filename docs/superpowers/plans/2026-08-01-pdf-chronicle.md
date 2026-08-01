# PDF Session Chronicle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automated June-style PDF chronicle: AI synthesis pass (recap, quest hooks, key scenes + image prompts), manual image generation by the user, one-command PDF assembly via Typst.

**Architecture:** New session-level synthesis call reuses `ai_providers` (schema/required-fields become parameters). `build_reports` computation is extracted into `compute_report_data()` shared by markdown reports and the PDF. PDF assembly stages everything into `out/pdf_build/` (template, fonts, images, `data.json`) and compiles with the `typst` pip package — zero system deps on macOS/Linux/Windows.

**Tech Stack:** Python stdlib, existing `ai_providers`, `typst` (pip wheel), Typst markup template, pytest.

**Spec:** `docs/superpowers/specs/2026-08-01-pdf-chronicle-design.md`
**Reference PDF:** `june/Session_June_Report.pdf` (style etalon)

## Global Constraints

- Existing markdown reports (`reports/*.md`) must stay byte-identical after the refactor — `tests/test_reports.py` is the regression gate.
- `ai_providers.py` must not import `dnd_pipeline`; `import typst` only lazily inside the render function.
- Tests: zero network, `typst`/`anthropic` packages not required (smoke test skips if `typst` missing, marked `slow`).
- Hand-placed files in `report_assets/` are read-only inputs — never modified.
- Synthesis goes to API only when `ai.enabled`; PDF assembly itself is local and free.
- Russian user-facing messages; image prompts inside synthesis output are English.
- Fonts: template uses `("PT Serif", "Libertinus Serif")` — PT Serif files optional in `pdf_template/fonts/` (graceful fallback to Typst's embedded Libertinus).
- Config defaults: `pdf: {enabled: true, assets_dir: null, campaign_title: "Хроники кампании", subtitle: "D&D · Forgotten Realms"}`.

---

### Task 1: Extract compute_report_data from build_reports

**Files:**
- Modify: `dnd_pipeline.py` (`build_reports`, line ~1565)
- Test: `tests/test_reports.py` (existing — must stay green), `tests/test_pdf_chronicle.py` (new)

**Interfaces:**
- Consumes: `read_manual_results(paths)` output rows.
- Produces: `compute_report_data(results: list[dict], cfg: dict) -> dict` with keys:
  - `actions: list[dict]` — sorted by time, raw action dicts
  - `dice: list[dict]` — raw roll dicts in input order
  - `dice_stats: dict[str, dict]` — per character: `{"avg": float, "count": int, "nat20": int, "nat1": int}` (only rolls with int `natural`)
  - `mvp_events: list[dict]` — raw mvp signals with normalized int `weight`
  - `mvp_scores: dict[str, int]` — per character total
  - `mvp_categories: dict[str, dict[str, int]]` — per character per category points
  - `summaries: list[dict]` — `{"chunk_index": int|None, "summary": str}` in chunk order
  Used by Tasks 5 and by rewritten `build_reports`.

- [ ] **Step 1: Write failing tests** (new file `tests/test_pdf_chronicle.py`)

```python
# tests/test_pdf_chronicle.py
"""Tests for the PDF chronicle stage: report data, synthesis, assets, staging."""
import pytest

import dnd_pipeline as dp


def _results_fixture():
    return [
        {
            "chunk_index": 0, "summary": "Драка в таверне",
            "actions": [
                {"time": "00:10:00.000", "character": "Гай", "action": "напал", "outcome": "успех", "importance": "high"},
                {"time": "00:05:00.000", "character": "Ангрон", "action": "вошёл", "outcome": "эффектно", "importance": "low"},
            ],
            "dice_rolls": [
                {"time": "00:11:00.000", "character": "Гай", "roll_type": "attack", "die": "d20",
                 "natural": 20, "modifier": 5, "total": 25, "context": "атака", "confidence": "high", "raw_text": "нат 20"},
                {"time": "00:12:00.000", "character": "Гай", "roll_type": "save", "die": "d20",
                 "natural": 1, "modifier": 2, "total": 3, "context": "спас", "confidence": "high", "raw_text": "единица"},
                {"time": "00:13:00.000", "character": "Ангрон", "roll_type": "attack", "die": "d20",
                 "natural": None, "modifier": None, "total": 18, "context": "без ната", "confidence": "low", "raw_text": "18"},
            ],
            "mvp_signals": [
                {"time": "00:10:30.000", "character": "Гай", "category": "combat", "reason": "яркая атака", "weight": 2},
                {"time": "00:14:00.000", "character": "Ангрон", "category": "fun", "reason": "шутка", "weight": "1"},
            ],
        },
        {
            "chunk_index": 1, "summary": "Допрос пленника",
            "actions": [], "dice_rolls": [],
            "mvp_signals": [
                {"time": "00:40:00.000", "character": "Гай", "category": "social", "reason": "жёсткий ход", "weight": 1},
            ],
        },
    ]


class TestComputeReportData:
    def test_actions_sorted_by_time(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        assert [a["character"] for a in data["actions"]] == ["Ангрон", "Гай"]

    def test_dice_stats_avg_and_crits(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        gai = data["dice_stats"]["Гай"]
        assert gai["avg"] == pytest.approx(10.5)
        assert gai["count"] == 2
        assert gai["nat20"] == 1
        assert gai["nat1"] == 1
        # Ангрон: natural=None не считается
        assert "Ангрон" not in data["dice_stats"]

    def test_mvp_scores_and_categories(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        assert data["mvp_scores"] == {"Гай": 3, "Ангрон": 1}
        assert data["mvp_categories"]["Гай"] == {"combat": 2, "social": 1}

    def test_string_weight_normalized(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        angron = [e for e in data["mvp_events"] if e["character"] == "Ангрон"]
        assert angron[0]["weight"] == 1

    def test_summaries_in_chunk_order(self):
        data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        assert data["summaries"] == [
            {"chunk_index": 0, "summary": "Драка в таверне"},
            {"chunk_index": 1, "summary": "Допрос пленника"},
        ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pdf_chronicle.py -v`
Expected: FAIL with `AttributeError: ... 'compute_report_data'`

- [ ] **Step 3: Implement** — add `compute_report_data` before `build_reports` and rewrite `build_reports` to consume it. Markdown output must remain byte-identical.

```python
def compute_report_data(results: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Общие расчёты для markdown-отчётов и PDF-хроники."""
    actions: list[dict[str, Any]] = []
    dice: list[dict[str, Any]] = []
    mvp_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for res in results:
        actions.extend(res.get("actions", []) or [])
        dice.extend(res.get("dice_rolls", []) or [])
        for item in res.get("mvp_signals", []) or []:
            item = dict(item)
            try:
                item["weight"] = int(item.get("weight", 1))
            except (ValueError, TypeError):
                item["weight"] = 1
            mvp_events.append(item)
        if res.get("summary"):
            summaries.append({"chunk_index": res.get("chunk_index"), "summary": res["summary"]})

    dice_stats: dict[str, dict[str, Any]] = {}
    for d in dice:
        natural = d.get("natural")
        if not isinstance(natural, int):
            continue
        char = d.get("character") or "?"
        st = dice_stats.setdefault(char, {"values": [], "nat20": 0, "nat1": 0})
        st["values"].append(natural)
        if natural == 20:
            st["nat20"] += 1
        if natural == 1:
            st["nat1"] += 1
    for char, st in dice_stats.items():
        values = st.pop("values")
        st["avg"] = sum(values) / len(values)
        st["count"] = len(values)

    mvp_scores: dict[str, int] = {}
    mvp_categories: dict[str, dict[str, int]] = {}
    for item in mvp_events:
        char = item.get("character") or "?"
        weight = item["weight"]
        mvp_scores[char] = mvp_scores.get(char, 0) + weight
        cat = item.get("category") or "?"
        mvp_categories.setdefault(char, {})
        mvp_categories[char][cat] = mvp_categories[char].get(cat, 0) + weight

    return {
        "actions": sorted(actions, key=lambda x: x.get("time", "")),
        "dice": dice,
        "dice_stats": dice_stats,
        "mvp_events": mvp_events,
        "mvp_scores": mvp_scores,
        "mvp_categories": mvp_categories,
        "summaries": summaries,
    }
```

Rewrite `build_reports` to use `data = compute_report_data(results, cfg)`:
- actions timeline iterates `data["actions"]` (already sorted — drop the local `sorted()`)
- dice section: per-roll lines still iterate `data["dice"]` in input order; the averages block iterates `sorted(data["dice_stats"].items())` printing `f"- **{char}**: {st['avg']:.2f} по {st['count']} броскам"`
- MVP: per-event lines iterate `data["mvp_events"]` (weight already int); totals iterate `sorted(data["mvp_scores"].items(), key=lambda x: x[1], reverse=True)`
- summaries: `[f"- chunk {s['chunk_index']}: {s['summary']}" for s in data["summaries"]]`

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pdf_chronicle.py tests/test_reports.py -v` then `python -m pytest -q`
Expected: all PASS, `test_reports.py` untouched and green (byte-identical markdown)

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_pdf_chronicle.py
git commit -m "refactor: extract compute_report_data shared by markdown reports and PDF"
```

---

### Task 2: Parameterize ai_providers schema and validation; add SYNTHESIS_SCHEMA

**Files:**
- Modify: `ai_providers.py`
- Test: `tests/test_ai_providers.py` (append)

**Interfaces:**
- Produces:
  - `parse_model_json(name, text, normalize, required_fields=REQUIRED_RESULT_FIELDS)` — new optional arg
  - `OpenAICompatProvider(..., required_fields: tuple[str, ...] | None = None)` — passed to `parse_model_json`
  - `AnthropicProvider(..., schema: dict | None = None)` — used in `_request_params` instead of hardcoded `EVENTS_SCHEMA` (default stays `EVENTS_SCHEMA`)
  - `SYNTHESIS_SCHEMA: dict` — top-level required: `recap` (string), `quest_hooks` (array of `{title, description}`), `scenes` (array of `{title, chunk_index, time, image_prompt}`); every object `additionalProperties: false`
  - `SYNTHESIS_REQUIRED_FIELDS = ("recap", "quest_hooks", "scenes")`
  Used by Task 3.

- [ ] **Step 1: Write failing tests** (append to `tests/test_ai_providers.py`)

```python
class TestSchemaParameterization:
    def test_parse_custom_required_fields(self):
        text = json.dumps({"recap": "r", "quest_hooks": [], "scenes": []})
        res = ap.parse_model_json("synthesis", text, lambda t: t,
                                  required_fields=("recap", "quest_hooks", "scenes"))
        assert res.error is None

    def test_parse_custom_required_fields_missing(self):
        res = ap.parse_model_json("synthesis", '{"recap": "r"}', lambda t: t,
                                  required_fields=("recap", "quest_hooks", "scenes"))
        assert "quest_hooks" in res.error

    def test_anthropic_custom_schema_in_params(self):
        client = FakeAnthropicClient()
        p = ap.AnthropicProvider(model="m", api_key="k", mode="direct", concurrency=1,
                                 client=client, schema=ap.SYNTHESIS_SCHEMA)
        p.analyze([ap.ChunkJob("synthesis", "p", "h")])
        assert client.direct_params["output_config"]["format"]["schema"] is ap.SYNTHESIS_SCHEMA

    def test_openai_compat_custom_required_fields(self, monkeypatch):
        payload = json.dumps({"recap": "r", "quest_hooks": [], "scenes": []})
        monkeypatch.setattr(ap.urllib.request, "urlopen",
                            lambda req, timeout=None: _fake_response(payload))
        p = ap.OpenAICompatProvider(model="m", base_url="http://x/v1", concurrency=1,
                                    required_fields=("recap", "quest_hooks", "scenes"))
        results = p.analyze([ap.ChunkJob("synthesis", "p", "h")])
        assert results[0].error is None


class TestSynthesisSchema:
    def test_structure(self):
        props = ap.SYNTHESIS_SCHEMA["properties"]
        assert set(ap.SYNTHESIS_SCHEMA["required"]) == {"recap", "quest_hooks", "scenes"}
        assert props["scenes"]["items"]["required"] == ["title", "chunk_index", "time", "image_prompt"]

    def test_all_objects_forbid_additional_properties(self):
        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(ap.SYNTHESIS_SCHEMA)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_ai_providers.py -v -k "SchemaParam or SynthesisSchema"`
Expected: FAIL (`TypeError: parse_model_json() got an unexpected keyword argument` / missing `SYNTHESIS_SCHEMA`)

- [ ] **Step 3: Implement in `ai_providers.py`**

- `parse_model_json(name, text, normalize, required_fields=REQUIRED_RESULT_FIELDS)`: replace the hardcoded loop over `REQUIRED_RESULT_FIELDS` with `required_fields`.
- `OpenAICompatProvider.__init__`: add `required_fields: tuple[str, ...] | None = None`; store `self.required_fields = required_fields or REQUIRED_RESULT_FIELDS`; `_analyze_one` passes it to `parse_model_json`.
- `AnthropicProvider.__init__`: add `schema: dict[str, Any] | None = None`; store `self.schema = schema or EVENTS_SCHEMA`; `_request_params` uses `self.schema`.
- Add after `EVENTS_SCHEMA`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_ai_providers.py -v` then `python -m pytest -q`
Expected: all PASS (old defaults untouched)

- [ ] **Step 5: Commit**

```bash
git add ai_providers.py tests/test_ai_providers.py
git commit -m "feat: parameterize provider schema/validation, add session synthesis schema"
```

---

### Task 3: Session synthesis stage

**Files:**
- Modify: `dnd_pipeline.py` (after `run_ai_analysis`)
- Test: `tests/test_pdf_chronicle.py` (append)

**Interfaces:**
- Consumes: `compute_report_data` (Task 1), `SYNTHESIS_SCHEMA`/`SYNTHESIS_REQUIRED_FIELDS` (Task 2), `resolve_ai_config`, `resolve_ai_api_key`, `load_ai_state`, `save_ai_state`, `read_manual_results`, `stable_hash`, `write_json`, `write_md`.
- Produces:
  - `build_synthesis_input(results: list[dict], cfg: dict) -> dict` — compact: all summaries, top-30 mvp events by weight, high-importance actions
  - `synthesis_prompt(payload: dict, party: list[dict]) -> str`
  - `make_synthesis_provider(ai: dict, api_key: str | None) -> Any` — like `make_ai_provider` but `mode="direct"`, synthesis schema/required fields
  - `run_session_synthesis(cfg, paths, party: list[dict], force: bool = False) -> dict | None` — None → synthesis unavailable (disabled/no key/failed); dict → synthesis result; writes `out_dir/session_synthesis.json` and `out_dir/image_prompts.md`; caches by input hash in `state["synthesis"]`
  Used by Tasks 5/7. (`party` comes from Task 4's loader; pass `[]` when absent.)

- [ ] **Step 1: Write failing tests** (append to `tests/test_pdf_chronicle.py`)

```python
import json

from ai_providers import AIResult


class FakeSynthProvider:
    def __init__(self, result):
        self.result = result
        self.seen_prompt = None

    def analyze(self, jobs, on_result=None):
        self.seen_prompt = jobs[0].prompt
        return [self.result]


def _synthesis_ok():
    return {
        "recap": "Партия ворвалась в таверну.",
        "quest_hooks": [{"title": "Щит Морбрина", "description": "Разведать поселение."}],
        "scenes": [{"title": "Драка в таверне", "chunk_index": 0, "time": "00:10:00.000",
                    "image_prompt": "dark fantasy tavern brawl, hulking warrior"}],
    }


def _session_with_results(tmp_path, monkeypatch, provider):
    paths = dp.build_paths(tmp_path, "test")
    dp.ensure_dirs(paths)
    for i, res in enumerate(_results_fixture()):
        dp.write_json(paths.manual_ai_dir / f"chunk_{i:03d}_events.json", res)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(dp, "make_synthesis_provider", lambda ai, key: provider)
    return paths


class TestBuildSynthesisInput:
    def test_compact_payload(self):
        payload = dp.build_synthesis_input(_results_fixture(), {"session_name": "t"})
        assert payload["session"] == "t"
        assert len(payload["summaries"]) == 2
        assert all("reason" in e for e in payload["mvp_top"])
        assert all(a["importance"] == "high" for a in payload["key_actions"])


class TestSynthesisPrompt:
    def test_mentions_party_appearance(self):
        party = [{"name": "Ангрон", "appearance_en": "hulking scarred warrior"}]
        prompt = dp.synthesis_prompt({"session": "t", "summaries": [], "mvp_top": [], "key_actions": []}, party)
        assert "hulking scarred warrior" in prompt
        assert "recap" in prompt


class TestRunSessionSynthesis:
    def test_disabled_returns_none(self, tmp_path):
        paths = dp.build_paths(tmp_path, "test")
        dp.ensure_dirs(paths)
        assert dp.run_session_synthesis({"session_name": "test"}, paths, []) is None

    def test_writes_result_and_prompts(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", data=_synthesis_ok()))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        result = dp.run_session_synthesis(cfg, paths, [])
        assert result["recap"].startswith("Партия")
        saved = dp.load_json(paths.out_dir / "session_synthesis.json")
        assert saved == result
        md = (paths.out_dir / "image_prompts.md").read_text(encoding="utf-8")
        assert "dark fantasy tavern brawl" in md
        assert "scene1" in md  # инструкция об именах файлов

    def test_cache_skips_provider(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", data=_synthesis_ok()))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        dp.run_session_synthesis(cfg, paths, [])
        provider.seen_prompt = None
        result2 = dp.run_session_synthesis(cfg, paths, [])
        assert result2 is not None
        assert provider.seen_prompt is None  # второй раз в провайдера не ходили

    def test_force_recomputes(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", data=_synthesis_ok()))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        dp.run_session_synthesis(cfg, paths, [])
        provider.seen_prompt = None
        dp.run_session_synthesis(cfg, paths, [], force=True)
        assert provider.seen_prompt is not None

    def test_provider_error_returns_none(self, tmp_path, monkeypatch):
        provider = FakeSynthProvider(AIResult(name="synthesis", error="boom"))
        paths = _session_with_results(tmp_path, monkeypatch, provider)
        cfg = {"session_name": "test", "ai": {"enabled": True}}
        assert dp.run_session_synthesis(cfg, paths, []) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pdf_chronicle.py -v -k "Synthesis"`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement in `dnd_pipeline.py`** (after `run_ai_analysis`)

```python
# ──────────────────────────────────────────────────────────────
# Session synthesis (recap, quest hooks, key scenes for the PDF)
# ──────────────────────────────────────────────────────────────


def build_synthesis_input(results: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    data = compute_report_data(results, cfg)
    mvp_top = sorted(data["mvp_events"], key=lambda e: e["weight"], reverse=True)[:30]
    key_actions = [a for a in data["actions"] if a.get("importance") == "high"]
    return {
        "session": cfg["session_name"],
        "summaries": data["summaries"],
        "mvp_top": mvp_top,
        "key_actions": key_actions,
    }


def synthesis_prompt(payload: dict[str, Any], party: list[dict[str, Any]]) -> str:
    party_desc = "\n".join(
        f"- {p.get('name')}: {p.get('appearance_en', '')}" for p in party
    ) or "(нет описаний)"
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""# Синтез D&D-сессии для иллюстрированной хроники

Ты собираешь материалы для PDF-хроники сессии. Верни СТРОГО JSON.

Задачи:
1. recap — художественный пересказ сессии, 4-6 абзацев на русском,
   «что было в прошлый раз» для игроков перед следующей игрой.
2. quest_hooks — нерешённые нити, добытые сведения, открытые направления
   (title + description, по-русски).
3. scenes — 3-5 самых кинематографичных сцен сессии. Для каждой:
   title (рус.), chunk_index, time (HH:MM:SS.mmm из данных),
   image_prompt — промпт для генерации картинки НА АНГЛИЙСКОМ:
   стиль dark fantasy oil painting, детали места и действия, внешность
   участников из списка ниже. Первая сцена — самая эффектная (обложка).

Внешность персонажей для image_prompt:
{party_desc}

Правила: не выдумывай события, опирайся только на данные. Имена
персонажей в русских текстах не переводи.

Данные сессии:

{data}
"""


def make_synthesis_provider(ai: dict[str, Any], api_key: str | None) -> Any:
    from ai_providers import (SYNTHESIS_REQUIRED_FIELDS, SYNTHESIS_SCHEMA,
                              AnthropicProvider, OpenAICompatProvider)

    if ai["provider"] == "anthropic":
        return AnthropicProvider(
            model=ai["model"], api_key=api_key, mode="direct",
            max_output_tokens=ai["max_output_tokens"], concurrency=1,
            schema=SYNTHESIS_SCHEMA,
        )
    return OpenAICompatProvider(
        model=ai["model"], base_url=ai["base_url"], api_key=api_key,
        max_output_tokens=ai["max_output_tokens"], concurrency=1,
        normalize=normalize_json_text, required_fields=SYNTHESIS_REQUIRED_FIELDS,
    )


def write_image_prompts_md(synthesis: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Промпты для картинок сцен",
        "",
        "Сгенерируй картинки и положи файлы в report_assets/ с именами",
        "scene1.png, scene2.png, ... (номер = номер сцены ниже; scene1 — обложка).",
        "",
    ]
    for i, scene in enumerate(synthesis.get("scenes", []), start=1):
        lines += [
            f"## scene{i} — {scene.get('title', '')} (`{scene.get('time', '')}`)",
            "",
            "```",
            scene.get("image_prompt", ""),
            "```",
            "",
        ]
    write_md(out_path, lines)


def run_session_synthesis(
    cfg: dict[str, Any], paths: Paths, party: list[dict[str, Any]], force: bool = False
) -> dict[str, Any] | None:
    """Сессионный синтез для PDF. None → недоступен (выключен/нет ключа/ошибка)."""
    from ai_providers import ChunkJob

    ai = resolve_ai_config(cfg)
    if not ai["enabled"]:
        logger.info("Синтез сессии пропущен: ai.enabled=false.")
        return None
    api_key = resolve_ai_api_key(ai)
    if ai["provider"] == "anthropic" and not api_key:
        logger.warning("Синтез сессии пропущен: нет API-ключа.")
        return None

    results = read_manual_results(paths)
    if not results:
        logger.warning("Синтез сессии пропущен: нет manual_ai_results/. Запусти ai-analyze.")
        return None

    payload = build_synthesis_input(results, cfg)
    prompt = synthesis_prompt(payload, party)
    input_hash = stable_hash({"prompt": prompt, "model": ai["model"], "provider": ai["provider"]})

    state = load_ai_state(paths)
    synth_path = paths.out_dir / "session_synthesis.json"
    cached = state.get("synthesis")
    if not force and cached and cached.get("input_hash") == input_hash and synth_path.exists():
        logger.info("Синтез сессии: беру из кэша.")
        return load_json(synth_path)

    provider = make_synthesis_provider(ai, api_key)
    logger.info(f"Синтез сессии: provider={ai['provider']}, model={ai['model']}…")
    result = provider.analyze([ChunkJob(name="synthesis", prompt=prompt, chunk_hash=input_hash)])[0]
    if result.data is None:
        logger.warning(f"Синтез сессии не удался: {result.error}")
        return None

    write_json(synth_path, result.data)
    write_image_prompts_md(result.data, paths.out_dir / "image_prompts.md")
    state["synthesis"] = {"input_hash": input_hash, "model": ai["model"], "provider": ai["provider"]}
    save_ai_state(paths, state)
    logger.info(f"Синтез готов: {synth_path.name}, промпты картинок: image_prompts.md")
    return result.data
```

Check `load_ai_state` — no change needed (`state.get("synthesis")` works without setdefault).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pdf_chronicle.py tests/test_ai_analysis.py -v` then `python -m pytest -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_pdf_chronicle.py
git commit -m "feat: session synthesis stage - recap, quest hooks, scene image prompts"
```

---

### Task 4: Assets loading (party.json, scene images, pdf config)

**Files:**
- Modify: `dnd_pipeline.py` (after synthesis block)
- Test: `tests/test_pdf_chronicle.py` (append)

**Interfaces:**
- Produces:
  - `PDF_DEFAULTS: dict` — `{"enabled": True, "assets_dir": None, "campaign_title": "Хроники кампании", "subtitle": "D&D · Forgotten Realms"}`
  - `resolve_pdf_config(cfg: dict) -> dict` — deep_merge with defaults
  - `pdf_assets_dir(session_dir: Path, pdf_cfg: dict) -> Path` — `pdf_cfg["assets_dir"]` resolved, else `session_dir / "report_assets"`
  - `load_party(assets_dir: Path) -> list[dict]` — `party.json` or `[]`; entries validated to have `name`
  - `find_scene_images(assets_dir: Path) -> dict[int, Path]` — files matching `scene<N>*.png|jpg|jpeg` → `{N: path}`
  Used by Tasks 5/7.

- [ ] **Step 1: Write failing tests** (append to `tests/test_pdf_chronicle.py`)

```python
class TestPdfAssets:
    def test_resolve_pdf_config_defaults(self):
        pdf = dp.resolve_pdf_config({})
        assert pdf["enabled"] is True
        assert pdf["campaign_title"] == "Хроники кампании"

    def test_assets_dir_default_and_override(self, tmp_path):
        assert dp.pdf_assets_dir(tmp_path, dp.resolve_pdf_config({})) == tmp_path / "report_assets"
        custom = tmp_path / "campaign_assets"
        pdf = dp.resolve_pdf_config({"pdf": {"assets_dir": str(custom)}})
        assert dp.pdf_assets_dir(tmp_path, pdf) == custom

    def test_load_party_missing_returns_empty(self, tmp_path):
        assert dp.load_party(tmp_path) == []

    def test_load_party_reads_and_filters(self, tmp_path):
        dp.write_json(tmp_path / "party.json", [
            {"name": "Ангрон", "class_ru": "Воин", "player": "Дима", "ref": "ref.jpg"},
            {"class_ru": "без имени — отбрасываем"},
        ])
        party = dp.load_party(tmp_path)
        assert len(party) == 1
        assert party[0]["name"] == "Ангрон"

    def test_find_scene_images(self, tmp_path):
        (tmp_path / "scene1_tavern.png").write_bytes(b"x")
        (tmp_path / "scene3.jpg").write_bytes(b"x")
        (tmp_path / "ref.1 Гай.jpg").write_bytes(b"x")
        (tmp_path / "scene_bad.png").write_bytes(b"x")
        scenes = dp.find_scene_images(tmp_path)
        assert sorted(scenes) == [1, 3]
        assert scenes[1].name == "scene1_tavern.png"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pdf_chronicle.py -v -k PdfAssets`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement**

```python
# ──────────────────────────────────────────────────────────────
# PDF chronicle: config and assets
# ──────────────────────────────────────────────────────────────

PDF_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "assets_dir": None,
    "campaign_title": "Хроники кампании",
    "subtitle": "D&D · Forgotten Realms",
}


def resolve_pdf_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(PDF_DEFAULTS, cfg.get("pdf") or {})


def pdf_assets_dir(session_dir: Path, pdf_cfg: dict[str, Any]) -> Path:
    if pdf_cfg.get("assets_dir"):
        return Path(pdf_cfg["assets_dir"]).expanduser().resolve()
    return session_dir / "report_assets"


def load_party(assets_dir: Path) -> list[dict[str, Any]]:
    p = assets_dir / "party.json"
    if not p.exists():
        return []
    try:
        raw = load_json(p)
    except ValueError as e:
        logger.warning(f"Битый party.json: {e}")
        return []
    return [m for m in raw if isinstance(m, dict) and m.get("name")]


def find_scene_images(assets_dir: Path) -> dict[int, Path]:
    scenes: dict[int, Path] = {}
    if not assets_dir.exists():
        return scenes
    for p in sorted(assets_dir.iterdir()):
        m = re.match(r"scene(\d+)", p.name)
        if m and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            scenes.setdefault(int(m.group(1)), p)
    return scenes
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pdf_chronicle.py -v` then `python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_pdf_chronicle.py
git commit -m "feat: PDF config and report_assets loading (party.json, scene images)"
```

---

### Task 5: PDF data assembly and build staging

**Files:**
- Modify: `dnd_pipeline.py`
- Test: `tests/test_pdf_chronicle.py` (append)

**Interfaces:**
- Consumes: Tasks 1-4 outputs.
- Produces:
  - `build_pdf_data(cfg, pdf_cfg, report_data, synthesis, party, scene_images) -> dict` — the exact dict serialized to `data.json`; scene entries get `"file": "images/scene<N>.<ext>"` or `None`; party refs get `"ref_file": "images/<ref name>"` or `None`; `synthesis` may be `None` → `recap`/`quest_hooks`/`scenes` empty
  - `stage_pdf_build(paths, data, party, scene_images, assets_dir, template_dir) -> Path` — creates `paths.out_dir / "pdf_build"`: copies `report.typ`, `fonts/` (if present) from `template_dir`, copies scene/ref images into `pdf_build/images/`, writes `data.json`; returns build dir. Re-runnable: dir wiped first.
  Used by Tasks 6/7. `template_dir` default: `Path(__file__).parent / "pdf_template"`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_pdf_chronicle.py`)

```python
class TestBuildPdfData:
    def _base_args(self, tmp_path):
        report_data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        synthesis = _synthesis_ok()
        party = [{"name": "Ангрон", "class_ru": "Воин", "player": "Дима", "ref": "ref.jpg",
                  "appearance_en": "warrior"}]
        (tmp_path / "scene1.png").write_bytes(b"png")
        scene_images = {1: tmp_path / "scene1.png"}
        return report_data, synthesis, party, scene_images

    def test_full_data(self, tmp_path):
        report_data, synthesis, party, scenes = self._base_args(tmp_path)
        data = dp.build_pdf_data({"session_name": "t"}, dp.resolve_pdf_config({}),
                                 report_data, synthesis, party, scenes)
        assert data["session"] == "t"
        assert data["campaign_title"] == "Хроники кампании"
        assert data["recap"].startswith("Партия")
        assert data["scenes"][0]["file"] == "images/scene1.png"
        assert data["party"][0]["ref_file"] is None  # ref.jpg не существует на диске
        assert data["mvp_scores"][0] == {"character": "Гай", "score": 3}

    def test_no_synthesis_degrades(self, tmp_path):
        report_data, _, party, scenes = self._base_args(tmp_path)
        data = dp.build_pdf_data({"session_name": "t"}, dp.resolve_pdf_config({}),
                                 report_data, None, party, scenes)
        assert data["recap"] == ""
        assert data["quest_hooks"] == []
        assert data["scenes"] == []


class TestStagePdfBuild:
    def test_stages_everything(self, tmp_path):
        paths = dp.build_paths(tmp_path / "session", "t")
        dp.ensure_dirs(paths)
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "scene1.png").write_bytes(b"png")
        (assets / "ref.jpg").write_bytes(b"jpg")
        template_dir = tmp_path / "tpl"
        template_dir.mkdir()
        (template_dir / "report.typ").write_text("#let data = json(\"data.json\")")

        party = [{"name": "Ангрон", "ref": "ref.jpg"}]
        scene_images = {1: assets / "scene1.png"}
        report_data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
        data = dp.build_pdf_data({"session_name": "t"}, dp.resolve_pdf_config({}),
                                 report_data, _synthesis_ok(), party, scene_images)

        build_dir = dp.stage_pdf_build(paths, data, party, scene_images, assets, template_dir)
        assert (build_dir / "report.typ").exists()
        assert (build_dir / "data.json").exists()
        assert (build_dir / "images" / "scene1.png").exists()
        assert (build_dir / "images" / "ref.jpg").exists()
        saved = dp.load_json(build_dir / "data.json")
        assert saved["party"][0]["ref_file"] == "images/ref.jpg"
```

Note: `build_pdf_data` decides `ref_file` by checking existence in `assets_dir`? No — keep it pure: `build_pdf_data` sets `ref_file` from `scene_images`/party `ref` names only if the file exists under the passed `assets_dir`. To keep it testable, `build_pdf_data` takes no filesystem args; instead `stage_pdf_build` fixes up `ref_file` for refs it successfully copies. Resolve: `build_pdf_data` sets `ref_file: None`; `stage_pdf_build` copies existing refs and rewrites `data["party"][i]["ref_file"]` before writing `data.json`. The first test asserts `ref_file is None` from `build_pdf_data` (file missing), the staging test asserts the rewritten value.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pdf_chronicle.py -v -k "BuildPdfData or StagePdfBuild"`
Expected: FAIL

- [ ] **Step 3: Implement**

```python
def build_pdf_data(
    cfg: dict[str, Any],
    pdf_cfg: dict[str, Any],
    report_data: dict[str, Any],
    synthesis: dict[str, Any] | None,
    party: list[dict[str, Any]],
    scene_images: dict[int, Path],
) -> dict[str, Any]:
    scenes = []
    for i, scene in enumerate((synthesis or {}).get("scenes", []), start=1):
        img = scene_images.get(i)
        scenes.append({
            **scene,
            "file": f"images/scene{i}{img.suffix.lower()}" if img else None,
        })
    mvp_scores = [
        {"character": char, "score": score}
        for char, score in sorted(report_data["mvp_scores"].items(), key=lambda x: x[1], reverse=True)
    ]
    dice_stats = [
        {"character": char, **st}
        for char, st in sorted(report_data["dice_stats"].items(), key=lambda x: x[1]["avg"], reverse=True)
    ]
    return {
        "session": cfg["session_name"],
        "campaign_title": pdf_cfg["campaign_title"],
        "subtitle": pdf_cfg["subtitle"],
        "recap": (synthesis or {}).get("recap", ""),
        "quest_hooks": (synthesis or {}).get("quest_hooks", []),
        "scenes": scenes,
        "party": [{**m, "ref_file": None} for m in party],
        "mvp_scores": mvp_scores,
        "mvp_categories": report_data["mvp_categories"],
        "mvp_events": report_data["mvp_events"],
        "dice_stats": dice_stats,
        "dice": report_data["dice"],
        "actions": report_data["actions"],
        "summaries": report_data["summaries"],
    }


def stage_pdf_build(
    paths: Paths,
    data: dict[str, Any],
    party: list[dict[str, Any]],
    scene_images: dict[int, Path],
    assets_dir: Path,
    template_dir: Path,
) -> Path:
    import shutil

    build_dir = paths.out_dir / "pdf_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    (build_dir / "images").mkdir(parents=True)

    shutil.copy2(template_dir / "report.typ", build_dir / "report.typ")
    fonts_src = template_dir / "fonts"
    if fonts_src.is_dir():
        shutil.copytree(fonts_src, build_dir / "fonts")

    for i, img in scene_images.items():
        shutil.copy2(img, build_dir / "images" / f"scene{i}{img.suffix.lower()}")

    for idx, member in enumerate(party):
        ref_name = member.get("ref")
        if ref_name and (assets_dir / ref_name).exists():
            shutil.copy2(assets_dir / ref_name, build_dir / "images" / ref_name)
            data["party"][idx]["ref_file"] = f"images/{ref_name}"

    write_json(build_dir / "data.json", data)
    return build_dir
```

Check `import shutil` placement — local import matches codebase style for rarely-used modules, or hoist to top; either fine, be consistent.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pdf_chronicle.py -v` then `python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dnd_pipeline.py tests/test_pdf_chronicle.py
git commit -m "feat: PDF data assembly and build directory staging"
```

---

### Task 6: Typst template and renderer

**Files:**
- Create: `pdf_template/report.typ`
- Create: `pdf_template/fonts/` (PT Serif, best-effort download)
- Modify: `dnd_pipeline.py` (`render_pdf`)
- Modify: `conftest.py` (register `slow` marker if not present)
- Test: `tests/test_pdf_chronicle.py` (append)

**Interfaces:**
- Produces: `render_pdf(build_dir: Path, out_path: Path) -> None` — lazy `import typst`; `typst.compile(str(build_dir / "report.typ"), output=str(out_path), root=str(build_dir), font_paths=[str(build_dir / "fonts")] if fonts exist else [])`. Missing package → `RuntimeError("Для сборки PDF нужен пакет typst: pip install typst")`.

- [ ] **Step 1: Write the template** — `pdf_template/report.typ`:

```typst
// Хроника сессии — June-стиль: тёмная обложка, кремовые страницы,
// бордо/золото. Данные приходят из data.json (см. build_pdf_data).
#let data = json("data.json")

#let dark = rgb("#241812")
#let cream = rgb("#f4edd8")
#let gold = rgb("#b9974e")
#let bordo = rgb("#7a1f1f")
#let ink = rgb("#3a2a1a")
#let muted = rgb("#7a5c3a")

#set text(font: ("PT Serif", "Libertinus Serif"), size: 10.5pt, fill: ink, lang: "ru")

// ── Обложка ──
#set page(paper: "a4", fill: dark, margin: (x: 2.2cm, top: 3.5cm, bottom: 2.5cm))
#align(center)[
  #text(fill: gold, style: "italic", size: 13pt)[#data.subtitle]
  #v(0.6em)
  #text(fill: cream, size: 30pt, weight: "bold")[ХРОНИКА СЕССИИ]
  #v(0.2em)
  #text(fill: gold, size: 26pt, weight: "bold")[#data.session]
  #v(1.2em)
  #if data.scenes.len() > 0 and data.scenes.at(0).file != none [
    #image(data.scenes.at(0).file, width: 92%)
  ]
  #v(1em)
  #if data.scenes.len() > 0 [
    #text(fill: bordo.lighten(35%), style: "italic", size: 12pt)[#data.scenes.at(0).title]
  ]
]

// ── Внутренние страницы ──
#set page(
  fill: cream,
  margin: (x: 2.2cm, top: 2.6cm, bottom: 2.4cm),
  background: pad(0.9cm, rect(width: 100%, height: 100%,
    stroke: 1pt + gold.darken(20%), radius: 1pt,
    inset: 3pt, rect(width: 100%, height: 100%, stroke: 0.5pt + gold))),
  footer: context align(center,
    text(size: 8pt, style: "italic", fill: muted)[
      #data.campaign_title · Сессия #data.session · стр. #counter(page).display()
    ]),
)
#counter(page).update(1)
#set heading(numbering: none)
#show heading.where(level: 1): it => [
  #text(fill: bordo, size: 18pt, weight: "bold")[#it.body]
  #v(-0.4em)
  #line(length: 100%, stroke: 0.7pt + gold.darken(10%))
  #v(0.4em)
]
#show heading.where(level: 2): it => text(fill: bordo, size: 13pt, weight: "bold")[#it.body]

#pagebreak(weak: true)
= Оглавление
#outline(title: none, depth: 1, indent: auto)

#pagebreak()
= Сводка
#text(style: "italic")[Сессия в цифрах: #data.summaries.len() эпизодов · #data.mvp_events.len() MVP-сигналов · #data.dice.len() бросков.]
#v(0.6em)
== MVP — кто затащил сессию
#{
  let max_score = if data.mvp_scores.len() > 0 { calc.max(..data.mvp_scores.map(s => s.score)) } else { 1 }
  table(
    columns: (auto, 1fr, auto), stroke: none, row-gutter: 0.45em,
    ..data.mvp_scores.enumerate().map(((i, s)) => (
      [#text(fill: if i == 0 { gold.darken(20%) } else { bordo })[#(i + 1). #s.character]],
      [#rect(width: 100% * s.score / max_score, height: 7pt,
             fill: if i == 0 { gold.darken(10%) } else { bordo }, radius: 3pt)],
      [*#s.score*],
    )).flatten()
  )
}
#v(0.6em)
== Кости — удача за столом
#table(
  columns: (1fr, auto, auto, auto, auto),
  stroke: 0.4pt + gold.darken(10%), inset: 6pt,
  table.header([*Персонаж*], [*Средний d20*], [*Бросков*], [*нат-20*], [*нат-1*]),
  ..data.dice_stats.map(s => (
    [#s.character],
    [#calc.round(s.avg, digits: 2)],
    [#s.count],
    [#if s.nat20 > 0 [#text(fill: gold.darken(20%))[*#s.nat20*]] else [·]],
    [#if s.nat1 > 0 [#text(fill: bordo)[*#s.nat1*]] else [·]],
  )).flatten()
)

#if data.recap != "" [
  #pagebreak()
  = Рекап — что было в прошлый раз
  #text(style: "italic")[Краткий пересказ ключевых событий сессии.]
  #v(0.5em)
  #for para in data.recap.split("\n\n") [
    #par(justify: true)[#para]
  ]
]

#if data.quest_hooks.len() > 0 [
  #pagebreak()
  = Зацепки и квесты
  #text(style: "italic")[Самое важное для следующей игры.]
  #v(0.5em)
  #for hook in data.quest_hooks [
    #par[— *#hook.title.* #hook.description]
    #v(0.3em)
  ]
]

#if data.party.len() > 0 [
  #pagebreak()
  = Партия
  #grid(
    columns: (1fr, 1fr), gutter: 1.2em,
    ..data.party.map(m => align(center)[
      #if m.ref_file != none [#image(m.ref_file, width: 100%)]
      #text(fill: bordo, weight: "bold")[#m.name]
      #if "class_ru" in m [ \ #text(style: "italic", size: 9pt)[#m.class_ru] ]
      #if "player" in m [ \ #text(style: "italic", size: 9pt)[игрок: #m.player] ]
    ])
  )
]

#if data.scenes.len() > 1 [
  #pagebreak()
  = Ключевые сцены
  #for scene in data.scenes.slice(1) [
    #if scene.file != none [#image(scene.file, width: 100%)]
    #align(center)[#text(fill: bordo, weight: "bold")[#scene.title] #text(size: 9pt, fill: muted)[`#scene.time`]]
    #v(0.8em)
  ]
]

#pagebreak()
= Хроника сессии — полный ход
#for s in data.summaries [
  #par[*Эпизод #s.chunk_index.* #s.summary]
  #v(0.35em)
]

#pagebreak()
= MVP — полный разбор
#for e in data.mvp_events [
  #par(hanging-indent: 1em)[`#e.time` *#e.character* +#e.weight [#e.category] — #e.reason]
]

#pagebreak()
= Кости — полная выкладка
#for d in data.dice [
  #par(hanging-indent: 1em)[`#d.time` *#d.character* #d.roll_type: нат=#repr(d.natural) мод=#repr(d.modifier) итог=#repr(d.total) — #d.context]
]

#pagebreak()
= Тайм-лайн ключевых действий
#for a in data.actions [
  #par(hanging-indent: 1em)[`#a.time` *#a.character* — #a.action → #a.outcome]
]
```

- [ ] **Step 2: Fonts (best-effort).** Try downloading PT Serif (OFL) into `pdf_template/fonts/`:

```bash
mkdir -p pdf_template/fonts
for f in PT_Serif-Web-Regular PT_Serif-Web-Bold PT_Serif-Web-Italic PT_Serif-Web-BoldItalic; do
  curl -fsSL -o "pdf_template/fonts/$f.ttf" \
    "https://github.com/google/fonts/raw/main/ofl/ptserif/$f.ttf" || true
done
ls -la pdf_template/fonts/
```

If download fails (offline/moved), remove the empty dir and note it in the report — template falls back to Libertinus Serif bundled inside typst. Do not block the task on fonts.

- [ ] **Step 3: Write failing tests** (append to `tests/test_pdf_chronicle.py`)

```python
import shutil


def _staged_build(tmp_path):
    paths = dp.build_paths(tmp_path / "session", "t")
    dp.ensure_dirs(paths)
    assets = tmp_path / "assets"
    assets.mkdir()
    report_data = dp.compute_report_data(_results_fixture(), {"session_name": "t"})
    data = dp.build_pdf_data({"session_name": "t"}, dp.resolve_pdf_config({}),
                             report_data, _synthesis_ok(), [], {})
    template_dir = dp.Path(dp.__file__).parent / "pdf_template"
    return dp.stage_pdf_build(paths, data, [], {}, assets, template_dir)


class TestRenderPdf:
    def test_missing_typst_raises_helpful_error(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def no_typst(name, *a, **k):
            if name == "typst":
                raise ImportError("no module")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_typst)
        with pytest.raises(RuntimeError, match="pip install typst"):
            dp.render_pdf(tmp_path, tmp_path / "out.pdf")

    @pytest.mark.slow
    def test_compiles_fixture_to_pdf(self, tmp_path):
        pytest.importorskip("typst")
        build_dir = _staged_build(tmp_path)
        out = tmp_path / "report.pdf"
        dp.render_pdf(build_dir, out)
        assert out.exists()
        assert out.read_bytes()[:5] == b"%PDF-"
```

- [ ] **Step 4: Implement `render_pdf`** in `dnd_pipeline.py`:

```python
def render_pdf(build_dir: Path, out_path: Path) -> None:
    try:
        import typst
    except ImportError as e:
        raise RuntimeError("Для сборки PDF нужен пакет typst: pip install typst") from e

    fonts_dir = build_dir / "fonts"
    font_paths = [str(fonts_dir)] if fonts_dir.is_dir() else []
    typst.compile(
        str(build_dir / "report.typ"),
        output=str(out_path),
        root=str(build_dir),
        font_paths=font_paths,
    )
    logger.info(f"PDF собран: {out_path}")
```

Register marker in `conftest.py` (if missing): add

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: медленные тесты (компиляция PDF)")
```

- [ ] **Step 5: Install typst and iterate the template**

```bash
pip install typst
python -m pytest tests/test_pdf_chronicle.py -v -k RenderPdf
```

Typst syntax errors are exact (line/column). Iterate on `pdf_template/report.typ` until the slow test compiles. Then eyeball the output: generate a fixture PDF to /tmp and open it — cover, frame, footer, TOC with page numbers must be present.

- [ ] **Step 6: Full suite**

Run: `python -m pytest -q`
Expected: PASS (slow test runs if typst installed; skips otherwise)

- [ ] **Step 7: Commit**

```bash
git add pdf_template/ dnd_pipeline.py conftest.py tests/test_pdf_chronicle.py
git commit -m "feat: Typst template and PDF renderer (June style, portable)"
```

---

### Task 7: CLI wiring, deps, docs

**Files:**
- Modify: `dnd_pipeline.py` (`cmd_run`, new `cmd_build_pdf`, `main()`)
- Modify: `requirements.txt`, `config.example.json`, `README.md`
- Test: `tests/test_pdf_chronicle.py` (append)

**Interfaces:**
- Produces:
  - `build_pdf_pipeline(session_dir, cfg, paths, force_synthesis=False) -> Path | None` — orchestrates: pdf_cfg → assets → party → synthesis → report_data → build_pdf_data → stage → render; returns PDF path or None (disabled / no results / typst missing → warning)
  - Subcommand `build-pdf <session_dir> [--config] [--force-synthesis] [--quality-profile]`
  - `cmd_run` tail: after `build_reports`, call `build_pdf_pipeline`; if scenes are missing it still runs (PDF without scene images) and logs: «Сцены: сгенерируй по out/image_prompts.md, положи в report_assets/ и запусти build-pdf».

- [ ] **Step 1: Write failing tests** (append to `tests/test_pdf_chronicle.py`)

```python
class TestBuildPdfCli:
    def _setup(self, tmp_path, monkeypatch):
        session = tmp_path / "session"
        session.mkdir()
        dp.write_json(session / "config.json", {"session_name": "test"})
        paths = dp.build_paths(session, "test")
        dp.ensure_dirs(paths)
        for i, res in enumerate(_results_fixture()):
            dp.write_json(paths.manual_ai_dir / f"chunk_{i:03d}_events.json", res)
        monkeypatch.setattr(dp, "run_session_synthesis",
                            lambda cfg, paths, party, force=False: _synthesis_ok())
        rendered = {}
        monkeypatch.setattr(dp, "render_pdf",
                            lambda build_dir, out: rendered.update({"build": build_dir, "out": out}) or out.write_bytes(b"%PDF-"))
        return session, paths, rendered

    def test_build_pdf_subcommand(self, tmp_path, monkeypatch):
        session, paths, rendered = self._setup(tmp_path, monkeypatch)
        rc = dp.main(["build-pdf", str(session)])
        assert rc == 0
        assert rendered["out"].name == "Session_test_Report.pdf"
        assert (rendered["build"] / "data.json").exists()

    def test_pdf_disabled_skips(self, tmp_path, monkeypatch):
        session, paths, rendered = self._setup(tmp_path, monkeypatch)
        dp.write_json(session / "config.json",
                      {"session_name": "test", "pdf": {"enabled": False}})
        rc = dp.main(["build-pdf", str(session)])
        assert rc == 0
        assert "out" not in rendered

    def test_typst_missing_warns_not_crashes(self, tmp_path, monkeypatch):
        session, paths, _ = self._setup(tmp_path, monkeypatch)
        def boom(build_dir, out):
            raise RuntimeError("Для сборки PDF нужен пакет typst: pip install typst")
        monkeypatch.setattr(dp, "render_pdf", boom)
        rc = dp.main(["build-pdf", str(session)])
        assert rc == 0  # warning, не падение
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pdf_chronicle.py -v -k BuildPdfCli`
Expected: FAIL (no `build-pdf` subcommand → SystemExit(2) from argparse)

- [ ] **Step 3: Implement**

```python
def build_pdf_pipeline(
    session_dir: Path, cfg: dict[str, Any], paths: Paths, force_synthesis: bool = False
) -> Path | None:
    pdf_cfg = resolve_pdf_config(cfg)
    if not pdf_cfg["enabled"]:
        logger.info("PDF выключен (pdf.enabled=false).")
        return None
    results = read_manual_results(paths)
    if not results:
        logger.warning("PDF пропущен: нет manual_ai_results/. Сначала ai-analyze.")
        return None

    assets_dir = pdf_assets_dir(session_dir, pdf_cfg)
    party = load_party(assets_dir)
    synthesis = run_session_synthesis(cfg, paths, party, force=force_synthesis)
    scene_images = find_scene_images(assets_dir)
    if synthesis and not scene_images:
        logger.info("Сцены: сгенерируй по out/image_prompts.md, положи в report_assets/ и запусти build-pdf.")

    report_data = compute_report_data(results, cfg)
    data = build_pdf_data(cfg, pdf_cfg, report_data, synthesis, party, scene_images)
    template_dir = Path(__file__).parent / "pdf_template"
    build_dir = stage_pdf_build(paths, data, party, scene_images, assets_dir, template_dir)
    out_path = paths.out_dir / f"Session_{cfg['session_name']}_Report.pdf"
    try:
        render_pdf(build_dir, out_path)
    except RuntimeError as e:
        logger.warning(str(e))
        return None
    return out_path


def cmd_build_pdf(args: argparse.Namespace) -> None:
    session_dir, cfg, paths = load_cfg(args)
    build_pdf_pipeline(session_dir, cfg, paths,
                       force_synthesis=getattr(args, "force_synthesis", False))
```

`cmd_run` tail — after `build_reports(paths, cfg)` add:

```python
        build_pdf_pipeline(session_dir, cfg, paths)
```

Parser (after `ai-analyze` block):

```python
    p_pdf = sub.add_parser("build-pdf", parents=[verbosity], help="assemble the illustrated PDF chronicle")
    p_pdf.add_argument("session_dir", help="folder with session files")
    p_pdf.add_argument("--config", help="path to config.json")
    p_pdf.add_argument("--force-synthesis", action="store_true", help="recompute session synthesis")
    p_pdf.add_argument("--quality-profile", choices=["gentle", "balanced", "aggressive"], help="override quality_profile")
    p_pdf.set_defaults(func=cmd_build_pdf)
```

- [ ] **Step 4: Deps and docs**

`requirements.txt` — append:

```
typst>=0.13  # PDF-хроника; pip-wheel, без системных зависимостей
```

`config.example.json` — add after the `ai` section:

```json
  "pdf": {
    "enabled": true,
    "assets_dir": null,
    "campaign_title": "Хроники кампании",
    "subtitle": "D&D · Forgotten Realms"
  },
```

`README.md` — new section: PDF-хроника (структура June-стиля), воркфлоу (run → image_prompts.md → Midjourney → report_assets/scene1.png… → build-pdf), `party.json` format with `appearance_en`, `pdf.*` config table, note про переносимость (pip-only).

- [ ] **Step 5: Run everything**

Run: `python -m pytest -q`, then smoke: `python dnd_pipeline.py build-pdf --help`
Expected: PASS; subcommand listed

- [ ] **Step 6: Commit**

```bash
git add dnd_pipeline.py requirements.txt config.example.json README.md tests/test_pdf_chronicle.py
git commit -m "feat: build-pdf command and run-stage integration for the PDF chronicle"
```

---

## Self-Review Notes

- Spec coverage: synthesis (T2-T3), assets/party (T4), data+staging (T5), Typst template/render/fonts (T6), CLI+deps+docs (T7), report-data refactor with byte-identical markdown (T1). No gaps.
- Type consistency: `compute_report_data` dict keys used identically in T3 (`build_synthesis_input`) and T5 (`build_pdf_data`); `run_session_synthesis(cfg, paths, party, force)` matches T7 call; `stage_pdf_build(paths, data, party, scene_images, assets_dir, template_dir)` consistent between T5 tests and T7 pipeline.
- Known risk: Typst syntax in the template is written blind — T6 explicitly budgets an iterate-on-compiler-errors loop with the slow test as the gate.
