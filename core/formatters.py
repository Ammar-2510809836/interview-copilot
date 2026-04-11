"""
Interview answer formatters module.

Provides formatting functions for different interview answer types including
STAR method, technical responses, coding answers, and follow-up responses.
"""

import re
from typing import List, Optional, Dict, Any


def format_star_response(situation: str, task: str, action: str, result: str) -> str:
    """
    Format a STAR method interview response.

    Args:
        situation: The context or background of the scenario
        task: The specific responsibility or challenge faced
        action: The steps taken to address the task
        result: The outcome or impact of the actions

    Returns:
        A properly formatted STAR answer with bold headers
    """
    parts = []

    if situation and situation.strip():
        parts.append(f"**Situation:** {situation.strip()}")

    if task and task.strip():
        parts.append(f"**Task:** {task.strip()}")

    if action and action.strip():
        parts.append(f"**Action:** {action.strip()}")

    if result and result.strip():
        parts.append(f"**Result:** {result.strip()}")

    if not parts:
        return ""

    return "\n\n".join(parts)


def format_technical_response(
    definition: str,
    key_points: List[str],
    experience_example: Optional[str] = None
) -> str:
    """
    Format a technical concept explanation response.

    Args:
        definition: One sentence direct answer defining the concept
        key_points: List of bullet point strings highlighting important aspects
        experience_example: Optional "In my experience..." sentence

    Returns:
        Formatted technical response with bold headers and bullet points
    """
    parts = []

    # Add definition
    if definition and definition.strip():
        parts.append(f"**Definition:** {definition.strip()}")

    # Add key points with bullet symbols
    if key_points:
        filtered_points = [p.strip() for p in key_points if p and p.strip()]
        if filtered_points:
            parts.append("**Key Points:**")
            for point in filtered_points:
                parts.append(f"▸ {point}")

    # Add experience example if provided
    if experience_example and experience_example.strip():
        parts.append(f"**Experience:** {experience_example.strip()}")

    if not parts:
        return ""

    return "\n\n".join(parts)


def format_coding_response(
    approach: str,
    code: str,
    complexity: str,
    edge_cases: List[str]
) -> str:
    """
    Format a coding interview response.

    Args:
        approach: Explanation of the solution approach and algorithm
        code: The code solution as a string
        complexity: Time/space complexity analysis
        edge_cases: List of edge cases to consider

    Returns:
        Formatted coding answer with approach, code block, complexity, and edge cases
    """
    parts = []

    # Add approach explanation
    if approach and approach.strip():
        parts.append(f"**Approach:**\n{approach.strip()}")

    # Add code block
    if code and code.strip():
        # Ensure code is properly wrapped in markdown code block
        code_clean = code.strip()
        if not code_clean.startswith("```"):
            code_clean = f"```python\n{code_clean}\n```"
        parts.append(f"**Solution:**\n{code_clean}")

    # Add complexity analysis
    if complexity and complexity.strip():
        parts.append(f"**Complexity:** {complexity.strip()}")

    # Add edge cases
    if edge_cases:
        filtered_cases = [c.strip() for c in edge_cases if c and c.strip()]
        if filtered_cases:
            parts.append("**Edge Cases:**")
            for case in filtered_cases:
                parts.append(f"▸ {case}")

    if not parts:
        return ""

    return "\n\n".join(parts)


def format_followup_response(
    previous_topic: str,
    additional_detail: str,
    deeper_insight: str
) -> str:
    """
    Format a follow-up response that builds on previous answers.

    Args:
        previous_topic: The topic/answer being acknowledged
        additional_detail: New information or detail to add
        deeper_insight: Deeper analysis or insight showing depth of thinking

    Returns:
        Formatted follow-up response acknowledging previous topic and adding depth
    """
    parts = []

    # Acknowledge previous topic
    if previous_topic and previous_topic.strip():
        parts.append(
            f"Building on what I mentioned about **{previous_topic.strip()}**..."
        )

    # Add additional detail
    if additional_detail and additional_detail.strip():
        parts.append(additional_detail.strip())

    # Add deeper insight
    if deeper_insight and deeper_insight.strip():
        parts.append(f"**Going deeper:** {deeper_insight.strip()}")

    if not parts:
        return ""

    return "\n\n".join(parts)


def parse_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse raw LLM output and detect if it follows STAR or other formats.

    Args:
        raw_text: The raw text output from an LLM

    Returns:
        Dictionary with:
            - type: "star", "technical", "coding", or "generic"
            - sections: Dictionary of extracted sections based on type
    """
    if not raw_text or not isinstance(raw_text, str):
        return {"type": "generic", "sections": {"full_text": "", "raw": raw_text}}

    text = raw_text.strip()

    # Detect STAR format
    star_pattern = re.compile(
        r'(?:\*\*)?(?:Situation|S):(?:\*\*)?\s*(.+?)(?=\n\s*(?:\*\*)?(?:Task|T):)',
        re.IGNORECASE | re.DOTALL
    )
    task_pattern = re.compile(
        r'(?:\*\*)?(?:Task|T):(?:\*\*)?\s*(.+?)(?=\n\s*(?:\*\*)?(?:Action|A):)',
        re.IGNORECASE | re.DOTALL
    )
    action_pattern = re.compile(
        r'(?:\*\*)?(?:Action|A):(?:\*\*)?\s*(.+?)(?=\n\s*(?:\*\*)?(?:Result|R):)',
        re.IGNORECASE | re.DOTALL
    )
    result_pattern = re.compile(
        r'(?:\*\*)?(?:Result|R):(?:\*\*)?\s*(.+?)(?=\n|$)',
        re.IGNORECASE | re.DOTALL
    )

    situation_match = star_pattern.search(text)
    task_match = task_pattern.search(text)
    action_match = action_pattern.search(text)
    result_match = result_pattern.search(text)

    # Check if STAR format is detected (need at least 3 components)
    star_components = sum([
        bool(situation_match),
        bool(task_match),
        bool(action_match),
        bool(result_match)
    ])

    if star_components >= 3:
        return {
            "type": "star",
            "sections": {
                "situation": situation_match.group(1).strip() if situation_match else "",
                "task": task_match.group(1).strip() if task_match else "",
                "action": action_match.group(1).strip() if action_match else "",
                "result": result_match.group(1).strip() if result_match else "",
                "full_text": text
            }
        }

    # Detect coding format (has code blocks)
    code_pattern = re.compile(r'```[\s\S]*?```')
    complexity_pattern = re.compile(
        r'(?:\*\*)?(?:Complexity|Time Complexity|Space Complexity):(?:\*\*)?',
        re.IGNORECASE
    )

    has_code_block = bool(code_pattern.search(text))
    has_complexity = bool(complexity_pattern.search(text))

    if has_code_block or has_complexity:
        # Try to extract code
        code_match = code_pattern.search(text)
        code_content = code_match.group(0) if code_match else ""

        return {
            "type": "coding",
            "sections": {
                "code": code_content,
                "full_text": text
            }
        }

    # Detect technical format (has definition/key points pattern)
    technical_pattern = re.compile(
        r'(?:\*\*)?(?:Definition|Key Points?|Technical):(?:\*\*)?',
        re.IGNORECASE
    )

    if technical_pattern.search(text):
        # Try to extract definition
        def_pattern = re.compile(
            r'(?:\*\*)?Definition:(?:\*\*)?\s*(.+?)(?=\n\s*(?:\*\*)|(?:▸)|\n\n|$)',
            re.IGNORECASE | re.DOTALL
        )
        def_match = def_pattern.search(text)

        return {
            "type": "technical",
            "sections": {
                "definition": def_match.group(1).strip() if def_match else "",
                "full_text": text
            }
        }

    # Default to generic
    return {
        "type": "generic",
        "sections": {
            "full_text": text,
            "raw": text
        }
    }


def strip_thinking_tokens(text: str) -> str:
    """
    Remove thinking tokens that some models add to their output.

    Removes patterns like:
    - <thinking>...</thinking>
    -  thinking.../thinking
    - [thinking]...[/thinking]
    -  thinking: ...

    Args:
        text: The text potentially containing thinking tokens

    Returns:
        Clean text with thinking tokens removed
    """
    if not text or not isinstance(text, str):
        return ""

    # Pattern for XML-style thinking tags
    patterns = [
        # XML-style tags: <thinking>...</thinking>
        (r'<thinking>\s*[\s\S]*?\s*</thinking>', ''),
        # Markdown code block style:  thinking.../thinking
        (r'```\s*thinking\s*\n[\s\S]*?\n```', ''),
        # BBCode style: [thinking]...[/thinking]
        (r'\[thinking\]\s*[\s\S]*?\s*\[/thinking\]', ''),
        # Colon prefix:  thinking: ...
        (r'(?:^|\n)\s*thinking:\s*(.+?)(?=\n|$)', ''),
        # Angle bracket variant: <think>...</think>
        (r'<think>\s*[\s\S]*?\s*</think>', ''),
        # Special token:  thinking.../thinking
        (r'<thinking>\s*[\s\S]*?\s*</thinking>', ''),
    ]

    cleaned = text
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Clean up excessive whitespace left behind
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()
