#stelle/services/task_utils.py

import requests
from config import GROQ_API_KEY_STELLE_MODEL

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _truncate_description(description: str, max_words: int = 40) -> str:
    """Truncate description to avoid overwhelming the title generator."""
    words = description.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return description


def _clean_task_name(raw: str) -> str:
    """Post-process LLM output to enforce short title rules."""
    # Take only the first line (ignore any extra content)
    first_line = raw.strip().split("\n")[0].strip()

    # Remove common prefixes the LLM might add
    for prefix in ["Task Title:", "Task Name:", "Title:", "**Task Title**:",
                   "1.", "2.", "**"]:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip()

    # Remove surrounding quotes and markdown bold
    first_line = first_line.strip('"\'\'\u201c\u201d*')

    # Enforce max 6 words
    words = first_line.split()
    if len(words) > 6:
        first_line = " ".join(words[:6])

    return first_line


def generate_task_name(description: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY_STELLE_MODEL}",
        "Content-Type": "application/json"
    }

    short_desc = _truncate_description(description)

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate SHORT task titles from user descriptions.\n"
                    "STRICT RULES:\n"
                    "- Output ONLY the title, nothing else\n"
                    "- Maximum 6 words\n"
                    "- No quotes, no numbering, no bullet points\n"
                    "- No emojis\n"
                    "- No meta descriptions, keywords, or strategies\n"
                    "- Do NOT list multiple titles\n"
                    "- Do NOT include any explanation\n"
                    "- Just the title, plain text, one line\n"
                )
            },
            {
                "role": "user",
                "content": f"Generate a single short title for this task: {short_desc}"
            }
        ],
        "temperature": 0.3,
        "max_tokens": 30
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()
    return _clean_task_name(raw)
