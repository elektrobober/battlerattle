# Design: Wire decode options into transcription (MLX + faster-whisper)

Date: 2026-06-30
Status: Approved

## Problem

The MLX backend (`run_mlx_transcribe`) is the one actually used on Apple Silicon
(M3 Max). It forwards only `path_or_hf_repo`, `language`, `verbose` to
`mlx_whisper.transcribe`. Every other quality knob in `config.json` is silently
dropped — including `condition_on_previous_text: false`, which the user sets to
suppress hallucination loop-propagation. On MLX it stays at the library default
(`True`), so the user's intent is silently defeated. This is a bug, not just a
missing feature.

In addition, neither backend passes Whisper's built-in input-side
anti-hallucination controls (`compression_ratio_threshold`, `logprob_threshold`,
`no_speech_threshold`, `hallucination_silence_threshold`) nor `initial_prompt`.
For the `turbo` model the user defaults to, a missing `initial_prompt` means
poor punctuation and worse proper-noun recognition (party names). MLX has no
Silero VAD, so `hallucination_silence_threshold` partly substitutes for it.

Garbage at this first stage propagates through dedupe → chunks → prompts →
reports (timeline / dice stats / MVP), where it cannot be repaired. Quality must
be invested at the input.

## Goals

1. Make `config.json` decode settings actually reach both backends.
2. Tune sensible per-profile anti-hallucination defaults.
3. Support a user-authored `initial_prompt`.

## Non-goals

- Auto-generating `initial_prompt` from track names (user writes it manually).
- Changing the post-hoc `hallucination_filter` (complementary, untouched).
- Switching ASR engines (Breeze/GigaAM/cloud) — out of scope.
- `language="auto"` default — the content is near-pure Russian; stays `ru`.
- Layered config (campaign base + tiny per-session file) — deferred to a separate
  follow-up sub-project. It is the main reuse improvement but kept out of this
  spec to keep scope tight. The manual `initial_prompt` added here is
  campaign-stable, so it lands cleanly in that future base config.

## Design

### Unit 1 — `resolve_decode_options(cfg, backend) -> dict`

Pure function. The core unit; fully unit-testable.

- Reads `cfg["decode"]` (profile defaults already merged with user overrides by
  the existing `apply_quality_profile` / `deep_merge` path).
- Emits backend-correct parameter names:
  - faster-whisper → `log_prob_threshold`
  - mlx → `logprob_threshold`
- Drops any key whose value is `None` so the library default applies (e.g.
  `initial_prompt: null` is not passed at all).
- Legacy fallback: if `decode.condition_on_previous_text` is absent, fall back to
  the top-level `cfg["condition_on_previous_text"]` so existing configs keep
  working.
- Returns a dict ready to splat: `transcribe(..., **opts)`.

Parameters handled: `condition_on_previous_text`, `initial_prompt`,
`compression_ratio_threshold`, `logprob_threshold`/`log_prob_threshold`,
`no_speech_threshold`, `hallucination_silence_threshold`.

`temperature` is left at the library default (its fallback schedule is wanted).

### Unit 2 — per-profile decode defaults (`apply_quality_profile`)

Each profile gains a `"decode"` block:

| param                            | gentle | balanced | aggressive |
|----------------------------------|-------:|---------:|-----------:|
| condition_on_previous_text       | false  | false    | false      |
| compression_ratio_threshold      | 2.6    | 2.4      | 2.2        |
| logprob_threshold                | -1.2   | -1.0     | -0.8       |
| no_speech_threshold              | 0.7    | 0.6      | 0.5        |
| hallucination_silence_threshold  | 5.0    | 2.0      | 1.0        |
| initial_prompt                   | null   | null     | null       |

Direction: `aggressive` rejects suspicious/silent output harder (risk: drops
quiet lines); `gentle` preserves quiet speech. Mirrors the existing
preprocess/dedupe profile philosophy. User can override any value in config.

### Unit 3 — wire into both transcribe functions

- `run_mlx_transcribe`: `mlx_whisper.transcribe(..., **resolve_decode_options(cfg, "mlx"))`.
- `run_faster_whisper_transcribe`: merge `resolve_decode_options(cfg, "faster_whisper")`
  into the existing kwargs; remove the now-duplicated hardcoded
  `condition_on_previous_text`.

### Unit 4 — cache invalidation

`transcribe_track` builds `transcription_cfg_hash` from a dict of settings. Add
the resolved decode options to that dict so changing a threshold invalidates the
per-track cache and forces re-transcription. Without this, tuning would silently
reuse stale results.

Consequence (expected): the first run after this change re-transcribes, since
existing caches no longer match.

### Unit 5 — config.example.json + docs

- Add a `decode` block (showing the `balanced` values) and an `initial_prompt`
  example containing the party/character names and a punctuation sample.
- Note in comments/README that `device` and `compute_type` apply to
  faster-whisper only; MLX ignores them (runs fp16 on Metal/ANE).

## Testing (TDD)

Unit-tested (pure logic, no model/audio needed):

- `resolve_decode_options`:
  - profile defaults present in output
  - user override beats profile default
  - backend name mapping (`log_prob_threshold` vs `logprob_threshold`)
  - `None` values dropped (e.g. absent `initial_prompt` not emitted)
  - legacy top-level `condition_on_previous_text` fallback honored
- `apply_quality_profile`:
  - each profile injects a `decode` block
  - user `decode` override merges over profile defaults

Real transcription is not tested (requires a model + hours of audio); all logic
lives in the resolver, which is fully covered.

## Risks

- Cache invalidation forces one re-transcription run. Expected and acceptable.
- Aggressive thresholds may drop genuine quiet lines — mitigated by `gentle`
  profile and per-value config override.
