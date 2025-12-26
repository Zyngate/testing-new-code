import json
from groq import Groq
from datetime import datetime, timedelta, timezone
from config import logger, GROK_REASONING

client = Groq(api_key=GROK_REASONING)


def build_timing_prompt(summary: str, platform: str) -> str:
    return f"""
You are a senior social media growth strategist.

A short-form video has the following meaning:

\"\"\"{summary}\"\"\"

Platform: {platform}

Your task:
1. Decide the BEST posting time (hour of day, 0–23) to maximize:
   - views
   - likes
   - comments
   - shares
   - follower growth
2. Explain WHY this timing works for THIS video specifically.
3. Base reasoning on:
   - audience psychology
   - emotional payoff timing
   - scrolling behavior
   - surprise vs attention patterns

Rules:
- No generic advice
- No hardcoded rules
- No mention of algorithms
- Reason MUST reference video content

Respond ONLY in JSON:

{{"best_hour": 21, "reason": "..." }}
"""


def ai_recommend_time(summary: str, platform: str):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": build_timing_prompt(summary, platform)}],
        temperature=0.4,
        max_tokens=300
    )

    data = json.loads(response.choices[0].message.content)
    return int(data["best_hour"]), data["reason"]


def compute_next_datetime(hour: int):
    now = datetime.now(timezone.utc)
    scheduled = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    if scheduled <= now:
        scheduled += timedelta(days=1)

    return scheduled
