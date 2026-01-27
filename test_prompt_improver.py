from config import GROQ_API_KEY_STELLE_MODEL
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def ask_stelle(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY_STELLE_MODEL}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are Stelle, an intelligent assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def build_executor_prompt(description: str, run_count: int) -> str:
    return f"""
You are executing ONE step of a recurring task.

User goal:
{description}

Execution number:
{run_count}

Context:
This task runs repeatedly. Each execution must move the user forward.

Rules:
- Produce content for THIS execution only
- Do NOT include multiple days or future steps
- Do NOT use lists or numbering
- Do NOT mention recurrence or time
- Assume this is the next logical step
- Keep it concise and actionable

Output:
Return only the content.
"""


# -----------------------
# 🔥 TEST CASES
# -----------------------

tests = [
    ("give me cooking tips everyday", 1),
    ("give me cooking tips everyday", 2),
    ("I want to learn Python programming", 1),
    ("I want to learn Python programming", 2),
]

for desc, run in tests:
    print("\n" + "=" * 80)
    print(f"USER INPUT: {desc}")
    print(f"RUN COUNT: {run}")
    prompt = build_executor_prompt(desc, run)
    print("\n--- EXECUTOR PROMPT ---")
    print(prompt.strip())
    print("\n--- LLM OUTPUT ---")
    print(ask_stelle(prompt))
