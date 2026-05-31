# Natural Delivery + Instant Thinking-Bridge — Design

**Date:** 2026-05-31
**Status:** Approved (pending spec review)

## Problem

During live AI interviews (e.g. micro1's "Zara"), two things make the copilot feel unnatural:

1. **Robotic answers.** The system prompt (`core/llm.py:130-304`) pushes bullet "answer cards",
   `**bold**`, visible STAR labels, and rigid "must include" structures. Read aloud, these sound
   listy and recited — not like a human speaking.

2. **Silent waiting gap.** The natural opener is the *first token of the LLM stream*, so it only
   appears after: turn-decision → `classify_question` (LLM) → RAG fetch → first token. That is 1–3s
   of silent "Copilot Thinking…" (`main.py:589`) during which the interviewer has stopped and the
   candidate has nothing to say. The candidate sits in silence, then starts late — visibly waiting
   on the tool.

## Goals

- Answers sound like a sharp human speaking in first person, not a documentation page.
- The candidate can **start talking within ~100ms** of the interviewer stopping, by surfacing an
  instant pre-defined thinking-bridge line while the real answer generates underneath.
- Answers are short enough to deliver without waiting for a long generation (3–4 sentences).

## Non-Goals

- No change to turn/pause detection (handled separately in `core/turn_detection.py`).
- No new LLM provider work, no UI framework changes.
- No disfluencies ("um", "like") — natural but still polished.

## Decisions (from brainstorming)

| Decision | Choice |
|----------|--------|
| Voice | Natural professional — contractions, flowing sentences, first person, no bullets when spoken, no disfluencies |
| Stall line | Instant pre-defined thinking-bridge, randomized by question type, **no evaluative phrasing** ("good question" is banned — a candidate doesn't praise the question) |
| On-screen format | Spoken-first prose; minimal/no bullets in spoken mode |
| Length | Tight: 3–4 sentences |
| Bridge display | Bridge line stays as part of the answer (one continuous spoken turn) |
| LLM opener | Removed in spoken mode — the app provides the opener, so the LLM goes straight into substance (no double-bridge) |

## Design

### Component 1 — Bridge-line picker (`core/bridge_lines.py`, new)

A pure, dependency-free module so it is instant (<100ms, no API call) and unit-testable.

```python
def pick_bridge_line(text: str, is_followup: bool = False) -> str: ...
```

- Classifies the question into `behavioral | technical | coding | followup | generic` using a
  **local keyword heuristic** (reuse the spirit of the existing `_route_model` / classification
  keyword logic — coding/technical/behavioral cues). No network call.
- `is_followup=True` always selects the follow-up pool.
- Returns a randomized line from the matching pool. Randomization avoids the same opener twice in a
  row where practical (track last-used line to skip immediate repeats).

Starting pools (approved):

- **behavioral:** "Yeah, I've actually got a good example of that." · "So one that comes to mind…" ·
  "Sure, let me give you a specific example."
- **technical:** "Right, so the way I'd approach this…" · "Okay, so at a high level…" ·
  "Yeah, so the core of it is…"
- **coding:** "Okay, let me think through the approach first." · "Right, so my first instinct here…"
- **followup:** "Yeah, so to go a bit deeper on that…" · "Right, building on what I just said…"
- **generic:** "Right, so…" · "Yeah, so for me…" · "Okay, so…"

**Invariant:** no line in any pool contains evaluative praise of the question.

### Component 2 — Spoken prompt mode (`core/llm.py`)

- When `ANSWER_STYLE=spoken` (already the default per `llm.py:86`), the system prompt:
  - Drops the "NATURAL BRIDGE LINE" section and the bullet "answer card" / STAR-label scaffolding.
  - Instructs: first person, contractions, flowing spoken sentences, **3–4 sentences max**, no
    markdown bullets or bold, no visible STAR/section labels. Lead technical answers with the
    concrete answer/command phrased as speech.
  - Still allows the existing `SKIP` behavior for filler.
- `ANSWER_STYLE=standard` keeps the current structured prompt unchanged (existing tests depend on
  this — see `TestAnswerStyle`-style cases in the graph).

### Component 3 — Wiring (`main.py`)

At the answer trigger (currently `main.py:589`, the `"Copilot Thinking..."` line):

1. Compute `bridge = pick_bridge_line(q_text, is_followup)`.
2. Show `bridge` in the overlay immediately (before `classify_question` / RAG / LLM).
3. Seed the streamed-answer buffer so the streamed tokens append after the bridge, rendering as one
   continuous turn: `answer_clean = bridge + " "` then stream as today.
4. The LLM (spoken mode) produces only substance, so there is no duplicate opener.

Edge cases:
- If the LLM returns `__SKIP__` after a bridge was shown (rare borderline), clear the bridge from the
  overlay and fall back to the listening state. Acceptable because bridges are content-light.
- Manual-trigger and interruption paths reuse the same `pick_bridge_line` call.

## Data Flow

```
interviewer stops
  → decide_turn_action → should_answer
  → pick_bridge_line(q_text)          [local, <100ms]
  → overlay shows bridge              [candidate starts speaking]
  → classify_question + RAG           [~0.5-1.5s]
  → generate_answer_stream (spoken)   [streams substance under bridge]
```

## Testing

- `tests/test_bridge_lines.py` (new):
  - returns a line from the correct pool for representative behavioral/technical/coding inputs
  - `is_followup=True` → follow-up pool
  - empty/short text → generic pool, never crashes
  - **no pool line contains banned evaluative phrases** (e.g. "good question", "great question")
  - repeated calls don't return the identical line twice in a row (anti-repeat)
- `tests/test_llm.py` (extend): spoken prompt has no bullet/STAR scaffolding and states the 3–4
  sentence spoken constraint; `ANSWER_STYLE=standard` prompt unchanged.

## Risks

- **Bridge/answer seam:** a bridge ending in "…" must flow into the streamed first words. Mitigated
  by spoken mode going straight into substance and prose (no leading label).
- **Wrong category bridge:** local heuristic may misclassify; impact is low because generic lines are
  safe for any question. The real answer still uses the full `classify_question`.
- **SKIP after bridge:** handled by clearing the bridge on `__SKIP__`.
```
