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
        "Right, so the way I’d approach this…",
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
