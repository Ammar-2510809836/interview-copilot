from dataclasses import dataclass


DANGLING_WORDS = {
    "the", "a", "an", "and", "or", "but", "so", "if", "when", "while",
    "because", "since", "although", "with", "without", "using", "by",
    "for", "from", "to", "in", "on", "at", "of", "about", "into",
    "through", "during", "before", "after", "between", "under", "over",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "might", "may", "can",
    "that", "which", "who", "whom", "whose", "where", "what",
    "we", "they", "i", "you", "he", "she", "it", "our", "their", "my",
    "this", "these", "those", "then", "than", "also", "very",
    "not", "just", "only", "even", "still", "already",
    "like", "such", "each", "every", "both", "either", "neither",
    "regarding", "concerning", "including", "especially", "specifically",
}

SHORT_FOLLOWUP_WORDS = {
    "why", "how", "how so", "what about", "what if", "and why", "can you elaborate",
}

# Pause thresholds are measured as SILENCE since the interviewer's last word,
# not as total turn duration. TTS interviewers (e.g. micro1's Zara) leave
# ~1-2s gaps between sentences of the same question, so the "complete" pause
# is deliberately patient to avoid cutting them off mid-question.
COMPLETE_PAUSE_SECONDS = 2.5
AMBIGUOUS_PAUSE_SECONDS = 3.5
# Safety valve: only force an answer on an unfinished sentence after a long
# silence (never based on how long they have been talking).
INCOMPLETE_FORCE_PAUSE_SECONDS = 6.0


@dataclass(frozen=True)
class TurnDecision:
    should_answer: bool
    force: bool
    reason: str


def _words(text: str) -> list[str]:
    return (text or "").strip().split()


def _last_word(text: str) -> str:
    words = _words(text.lower())
    return words[-1].rstrip(".,!?;:") if words else ""


def is_short_followup(text: str, previous_question: str = "") -> bool:
    if not previous_question:
        return False

    normalized = " ".join((text or "").lower().strip().rstrip("?!.,").split())
    if normalized in SHORT_FOLLOWUP_WORDS:
        return True

    return len(_words(text)) <= 4 and normalized.startswith(("why", "how", "what if", "what about"))


def looks_incomplete(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return True
    if clean.endswith(","):
        return True
    return _last_word(clean) in DANGLING_WORDS


def looks_complete(text: str) -> bool:
    clean = (text or "").strip()
    if len(_words(clean)) < 4:
        return False
    if looks_incomplete(clean):
        return False
    if clean.endswith(("?", ".", "!")):
        return True
    lowered = clean.lower()
    return lowered.startswith((
        "tell me about",
        "describe",
        "give me an example",
        "walk me through",
        "explain",
        "how would",
        "what would",
        "why would",
        "can you",
        "could you",
    ))


def recommended_wait_timeout(text: str, previous_question: str = "") -> float:
    """How long to wait for the next word before re-evaluating the turn.

    Returned as the polling interval for silence detection; the actual pause is
    tracked from the interviewer's last word, so an idle loop accumulates pause
    across successive polls.
    """
    if not text:
        return AMBIGUOUS_PAUSE_SECONDS
    if is_short_followup(text, previous_question):
        return 0.4
    if looks_complete(text):
        return COMPLETE_PAUSE_SECONDS
    return AMBIGUOUS_PAUSE_SECONDS


def decide_turn_action(text: str, pause: float, previous_question: str = "") -> TurnDecision:
    """Decide whether to answer given the SILENCE gap since the last word.

    `pause` is the wall-clock silence since the interviewer's last transcript
    chunk — not the total time they have been speaking. This keeps the copilot
    from jumping in during natural mid-question pauses.
    """
    if is_short_followup(text, previous_question):
        return TurnDecision(True, False, "short_followup")

    if looks_incomplete(text):
        if pause >= INCOMPLETE_FORCE_PAUSE_SECONDS:
            return TurnDecision(True, True, "max_wait")
        return TurnDecision(False, False, "incomplete")

    if looks_complete(text) and pause >= COMPLETE_PAUSE_SECONDS:
        return TurnDecision(True, False, "complete_pause")

    if pause >= AMBIGUOUS_PAUSE_SECONDS:
        return TurnDecision(True, False, "ambiguous_pause")

    return TurnDecision(False, False, "waiting")
