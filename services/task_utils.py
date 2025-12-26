#stelle/services/task_utils.py

import requests
from config import GROQ_API_KEY_STELLE_MODEL

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_task_name(description: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY_STELLE_MODEL}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate short task titles.\n"
                    "Rules:\n"
                    "- Max 6 words\n"
                    "- No quotes\n"
                    "- No emojis\n"
                    "- Clear and professional\n"
                )
            },
            {
                "role": "user",
                "content": f"Create a task name for: {description}"
            }
        ],
        "temperature": 0.3
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
