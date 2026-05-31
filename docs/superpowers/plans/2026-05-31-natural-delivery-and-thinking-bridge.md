# Natural Delivery + Instant Thinking-Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live answers sound like a human speaking and surface an instant thinking-bridge line the moment the interviewer stops, so the candidate never sits in silence waiting on the tool.

**Architecture:** A new pure module `core/bridge_lines.py` picks a randomized, category-appropriate opener with a local keyword heuristic (no API call). `core/llm.py`'s spoken prompt is rewritten to produce 3–4 flowing spoken sentences with no bullets/labels and no opener (the app supplies it). `main.py` shows the bridge instantly at the answer-trigger point and seeds the streamed answer so bridge + answer read as one continuous turn.

**Tech Stack:** Python 3, asyncio, PyQt overlay, pytest/unittest.

---

## File Structure

- **Create** `core/bridge_lines.py` — bridge-line pools + `pick_bridge_line()` heuristic. Pure, no I/O.
- **Create** `tests/test_bridge_lines.py` — unit tests for the picker.
- **Modify** `core/llm.py:369-406` — rewrite `_answer_style_prompt()` to flowing-prose spoken mode.
- **Modify** `tests/test_llm.py` — extend with spoken-prompt shape assertions (only if importable in CI; see Task 2 note).
- **Modify** `main.py` — call `pick_bridge_line`, show it instantly, seed the answer buffer, clear on SKIP.

---

## Task 1: Bridge-line picker module

**Files:**
- Create: `core/bridge_lines.py`
- Test: `tests/test_bridge_lines.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bridge_lines.py`:

```python
import unittest

from core.bridge_lines import pick_bridge_line, BANNED_SUBSTRINGS, BRIDGE_POOLS


class TestBridgeLines(unittest.TestCase):
    def test_behavioral_question_uses_behavioral_pool(self):
        line = pick_bridge_line("Tell me about a time you handled conflict on your team.")
        self.assertIn(line, BRIDGE_POOLS["behavioral"])

    def test_technical_question_uses_technical_pool(self):
        line = pick_bridge_line("How would you design a scalable Kubernetes deployment on AWS?")
        self.assertIn(line, BRIDGE_POOLS["technical"])

    def test_coding_question_uses_coding_pool(self):
        line = pick_bridge_line("Write a function to reverse a linked list.")
        self.assertIn(line, BRIDGE_POOLS["coding"])

    def test_followup_flag_uses_followup_pool(self):
        line = pick_bridge_line("And why is that?", is_followup=True)
        self.assertIn(line, BRIDGE_POOLS["followup"])

    def test_empty_text_returns_generic_line_without_crashing(self):
        line = pick_bridge_line("")
        self.assertIn(line, BRIDGE_POOLS["generic"])

    def test_no_pool_line_contains_evaluative_phrasing(self):
        for pool in BRIDGE_POOLS.values():
            for line in pool:
                lowered = line.lower()
                for banned in BANNED_SUBSTRINGS:
                    self.assertNotIn(banned, lowered, f"{line!r} contains banned phrase {banned!r}")

    def test_consecutive_calls_avoid_immediate_repeat(self):
        # With anti-repeat, 20 calls on the same input should not return the same
        # line twice in a row (pools have >= 2 lines each).
        prev = None
        for _ in range(20):
            line = pick_bridge_line("How would you scale this API?")
            self.assertNotEqual(line, prev)
            prev = line


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bridge_lines.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.bridge_lines'`

- [ ] **Step 3: Write minimal implementation**

Create `core/bridge_lines.py`:

```python
"""Instant, pre-defined thinking-bridge openers for live interview answers.

Pure and dependency-free so it runs in <1ms with no API call. The app shows the
returned line the moment the interviewer stops, then streams the real answer
underneath it as one continuous spoken turn.

Lines are deliberately NON-evaluative — a candidate in the hot seat does not
praise the question ("good question"). They are thinking-bridges that flow into
substance.
"""

import random

# A candidate never praises the interviewer's question; these must never appear.
BANNED_SUBSTRINGS = ("good question", "great question", "interesting question", "nice question")

BRIDGE_POOLS = {
    "behavioral": [
        "Yeah, I've actually got a good example of that.",
        "So one that comes to mind…",
        "Sure, let me give you a specific example.",
    ],
    "technical": [
        "Right, so the way I'd approach this…",
        "Okay, so at a high level…",
        "Yeah, so the core of it is…",
    ],
    "coding": [
        "Okay, let me think through the approach first.",
        "Right, so my first instinct here…",
    ],
    "followup": [
        "Yeah, so to go a bit deeper on that…",
        "Right, building on what I just said…",
    ],
    "generic": [
        "Right, so…",
        "Yeah, so for me…",
        "Okay, so…",
    ],
}

_BEHAVIORAL_CUES = (
    "tell me about a time", "give me an example", "describe a situation",
    "how do you handle", "how do you deal", "conflict", "disagree",
    "strength", "weakness", "challenge you faced", "motivat", "why this role",
    "why do you want", "proud of", "failure", "mistake",
)

_CODING_CUES = (
    "write a function", "write code", "implement", "reverse", "algorithm",
    "given an array", "given a string", "leetcode", "time complexity",
    "big o", "data structure", "solve this", "code this",
)

_TECHNICAL_CUES = (
    "design", "architecture", "scalable", "scale", "kubernetes", "docker",
    "aws", "cloud", "database", "sql", "api", "latency", "throughput",
    "deploy", "pipeline", "terraform", "network", "cache", "security",
    "system design", "trade-off", "tradeoff", "how would you", "how does",
    "what is", "explain", "compare", "why use",
)

_last_line = None


def _classify(text: str) -> str:
    lowered = (text or "").lower()
    if not lowered.strip():
        return "generic"
    if any(cue in lowered for cue in _BEHAVIORAL_CUES):
        return "behavioral"
    if any(cue in lowered for cue in _CODING_CUES):
        return "coding"
    if any(cue in lowered for cue in _TECHNICAL_CUES):
        return "technical"
    return "generic"


def pick_bridge_line(text: str, is_followup: bool = False) -> str:
    """Return an instant opener line for the given interviewer question.

    `is_followup` forces the follow-up pool. Avoids returning the same line as
    the immediately previous call so openers do not sound scripted.
    """
    global _last_line
    category = "followup" if is_followup else _classify(text)
    pool = BRIDGE_POOLS[category]
    choices = [line for line in pool if line != _last_line] or pool
    line = random.choice(choices)
    _last_line = line
    return line
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bridge_lines.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add core/bridge_lines.py tests/test_bridge_lines.py
git commit -m "feat: add instant thinking-bridge line picker"
```

---

## Task 2: Rewrite spoken prompt to flowing prose

**Files:**
- Modify: `core/llm.py:369-406` (`_answer_style_prompt`)
- Test: `tests/test_llm.py`

**Note on tests:** `tests/test_llm.py` imports `chromadb` (via `core.rag`) which may be absent in
some environments. If `python -m pytest tests/test_llm.py` errors on collection with
`ModuleNotFoundError: No module named 'chromadb'`, run `pip install chromadb` first. The assertions
below only touch `LLMClient._answer_style_prompt`, no network.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_llm.py` (inside the existing test class that constructs an `LLMClient`, or a new
`TestSpokenPrompt` class). Use this self-contained class:

```python
import os
import unittest

from core.llm import LLMClient


class TestSpokenPrompt(unittest.TestCase):
    def test_spoken_mode_is_flowing_prose_with_no_bullet_scaffold(self):
        os.environ["ANSWER_STYLE"] = "spoken"
        try:
            prompt = LLMClient()._answer_style_prompt()
        finally:
            os.environ.pop("ANSWER_STYLE", None)
        lowered = prompt.lower()
        # No bullet / card scaffolding in spoken mode.
        self.assertNotIn("▸", prompt)          # ▸ bullet glyph
        self.assertNotIn("say:", lowered)
        self.assertNotIn("close:", lowered)
        self.assertNotIn("bullet", lowered)
        # States the spoken sentence cap and prose intent.
        self.assertIn("sentence", lowered)
        self.assertIn("do not use bullet", lowered)

    def test_standard_mode_returns_no_extra_shaping(self):
        os.environ["ANSWER_STYLE"] = "standard"
        try:
            prompt = LLMClient()._answer_style_prompt()
        finally:
            os.environ.pop("ANSWER_STYLE", None)
        self.assertEqual(prompt, "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py::TestSpokenPrompt -q`
Expected: FAIL — `test_spoken_mode_is_flowing_prose_with_no_bullet_scaffold` fails because the current
prompt still contains `▸`, `Say:`, and `Close:`.

- [ ] **Step 3: Replace `_answer_style_prompt`**

In `core/llm.py`, replace the whole method body (lines 369-406) with:

```python
    def _answer_style_prompt(self) -> str:
        """Spoken-delivery shaping for live interviews — flowing prose, no bullets.

        The app shows an instant thinking-bridge opener (see core/bridge_lines.py),
        so the model goes straight into substance with no opener of its own.
        """
        if self.answer_style not in {"spoken", "natural", "live"}:
            return ""

        return """

---

## SPOKEN LIVE INTERVIEW MODE
I am speaking these words out loud in a live interview, so they must sound like natural speech, not a written document.

- Speak in first person with contractions ("I'd", "I've", "that's"). Confident, specific, practical.
- Keep it to 3-4 flowing sentences. Lead with the actual answer, then one concrete proof point from my experience.
- Do NOT use bullet points, numbered lists, markdown bold, headings, or STAR/section labels.
- Do NOT open with a bridge or filler line — start directly on the substance (the app already supplied my opening words).
- For technical questions, say the concrete answer or exact command first, then briefly why.
- For behavioral questions, tell it as a short spoken story: the situation, what I did, and the result — woven together, not labelled.
- Leave room for a follow-up; do not cram in every detail.
- If the interviewer only said filler with no question, reply exactly with "SKIP".
"""
```

> NOTE TO IMPLEMENTER: type the string cleanly as plain ASCII — the line above must read
> `Do NOT use bullet points, numbered lists, markdown bold, headings, or STAR/section labels.`
> (no stray characters). The test asserts the substring `do not use bullet`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_llm.py::TestSpokenPrompt -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the existing llm suite for regressions**

Run: `python -m pytest tests/test_llm.py -q`
Expected: PASS (or unchanged pre-existing pass count). If it errors on `chromadb` import, install it
first as noted above, then re-run.

- [ ] **Step 6: Commit**

```bash
git add core/llm.py tests/test_llm.py
git commit -m "feat: spoken prompt produces flowing prose, no bullets or opener"
```

---

## Task 3: Wire the instant bridge into the live loop

**Files:**
- Modify: `main.py` (import; trigger point at `main.py:589`; SKIP handling)

**Note:** This is async-loop integration glue around the unit-tested `pick_bridge_line`. There is no
unit test for the loop itself; verification is a compile check plus a manual run. Keep edits surgical.

- [ ] **Step 1: Add the import**

In `main.py`, find the existing turn-detection import (line ~22):

```python
from core.turn_detection import decide_turn_action, recommended_wait_timeout
```

Add directly beneath it:

```python
from core.bridge_lines import pick_bridge_line
```

- [ ] **Step 2: Show the bridge instantly and seed the answer buffer**

In `main.py`, locate this block (around line 589):

```python
                    last_advice = "<i style='color:#888888'>Copilot Thinking...</i>"
                    update_ui()

                    # Fetch RAG with conversation history
                    context = rag_manager.retrieve_context(q_text, conversation_history=transcript_history)

                    # --- STREAMING GENERATION ---
                    answer_clean = ""
```

Replace it with:

```python
                    # Instant thinking-bridge: show a natural opener the moment the
                    # interviewer stops, so the candidate can start talking while the
                    # real answer generates. The spoken prompt emits no opener of its
                    # own, so the streamed text continues this line as one turn.
                    bridge_line = pick_bridge_line(q_text, is_followup)
                    ui_overlay.update_text(
                        bridge_line,
                        question_type=q_type if q_type else "generic",
                        is_streaming=True,
                    )

                    # Fetch RAG with conversation history
                    context = rag_manager.retrieve_context(q_text, conversation_history=transcript_history)

                    # --- STREAMING GENERATION ---
                    # Seed with the bridge so streamed tokens append after it.
                    answer_clean = bridge_line + " "
```

- [ ] **Step 3: Verify the streaming render starts from the seeded buffer**

Confirm the streaming loop just below still appends to `answer_clean` and renders it (it does today at
`main.py:645-654`). No change needed — seeding `answer_clean` means the first repaint shows
`bridge_line + streamed text`. If `BATCH_CHARS` gating delays the first repaint, the explicit
`ui_overlay.update_text(bridge_line, ...)` in Step 2 already painted the bridge, so the opener is
visible immediately regardless.

- [ ] **Step 4: Clear the bridge if the model returns SKIP**

In `main.py`, find the post-stream SKIP handling (search for `elif skipped:` near `main.py:666`).
Inspect the existing branch; ensure that when `skipped` is true the overlay is reset to the listening
state rather than leaving the bridge on screen. The existing block sets a listening/advice message —
if it does not call `update_ui()` after clearing, add it. Concretely, the `elif skipped:` branch
should end with:

```python
                        last_advice = "<i style='color:#888888'>(Listening)</i>"
                        update_ui()
                        continue
```

(Only add the lines that are missing; do not duplicate an existing `continue`.)

- [ ] **Step 5: Compile check**

Run: `python -m py_compile main.py`
Expected: no output (exit 0).

- [ ] **Step 6: Run the focused unit suites**

Run: `python -m pytest tests/test_bridge_lines.py tests/test_turn_detection.py -q`
Expected: PASS (all green).

- [ ] **Step 7: Manual smoke verification**

Launch the app and ask a sample question (or feed a transcript). Confirm:
- The bridge line appears effectively instantly when the interviewer stops.
- The streamed answer continues from the bridge with no duplicate opener and no bullets.
- A pure-filler interviewer utterance does not leave a dangling bridge on screen.

Document the observed result (pass/fail) — do not claim success without running it.

- [ ] **Step 8: Commit**

```bash
git add main.py
git commit -m "feat: show instant thinking-bridge at answer trigger"
```

---

## Self-Review Notes

- **Spec coverage:** Voice (Task 2), instant bridge (Tasks 1+3), spoken-first prose (Task 2),
  3–4 sentence cap (Task 2), bridge stays as part of answer (Task 3 seeding), no evaluative phrasing
  (Task 1 `BANNED_SUBSTRINGS` test), SKIP-after-bridge edge (Task 3 Step 4). All covered.
- **Type consistency:** `pick_bridge_line(text, is_followup=False)` and `BRIDGE_POOLS` /
  `BANNED_SUBSTRINGS` names are identical across Task 1 impl, Task 1 tests, and Task 3 call site.
- **Standard mode preserved:** `_answer_style_prompt` still returns `""` for non-spoken styles
  (Task 2 second test), so `ANSWER_STYLE=standard` keeps the existing structured system prompt.
