import json
from groq import Groq
from datetime import datetime, timedelta, timezone
import pytz
from config import logger, GROK_REASONING

client = Groq(api_key=GROK_REASONING)

# Minimum and maximum posting hours - prevents scheduling during sleeping hours
MINIMUM_POSTING_HOUR = 8   # 8:00 AM
MAXIMUM_POSTING_HOUR = 23  # 11:00 PM


def build_timing_prompt(summary: str, platform: str) -> str:
    return f"""
You are a senior social media growth strategist.

A short-form video has the following meaning:

\"\"\"{summary}\"\"\"

Platform: {platform}

Your task:
1. Decide the BEST posting time (hour of day, 8–23 only, NO hours before 8 AM) to maximize:
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
- NEVER suggest times before 8 AM (hours 0-7) as audience is sleeping

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
    hour = int(data["best_hour"])
    
    # Ensure hour is within valid posting range (not during sleeping hours)
    if hour < MINIMUM_POSTING_HOUR:
        logger.warning(f"AI suggested hour {hour} (sleeping time), adjusting to {MINIMUM_POSTING_HOUR + 1}")
        hour = MINIMUM_POSTING_HOUR + 1  # 9 AM instead of early morning
    elif hour > MAXIMUM_POSTING_HOUR:
        hour = MAXIMUM_POSTING_HOUR
    
    return hour, data["reason"]


def compute_next_datetime(hour: int, user_timezone: str = None):
    """
    Compute the next datetime for the given hour in the user's timezone.
    
    Args:
        hour: Hour of day (0-23)
        user_timezone: User's timezone string (e.g., 'Asia/Kolkata', 'America/New_York')
    
    Returns:
        datetime in UTC
    """
    # Ensure hour is valid
    if hour < MINIMUM_POSTING_HOUR:
        hour = MINIMUM_POSTING_HOUR + 1
    elif hour > MAXIMUM_POSTING_HOUR:
        hour = MAXIMUM_POSTING_HOUR
    
    if user_timezone:
        try:
            tz = pytz.timezone(user_timezone)
            now_local = datetime.now(tz)
            scheduled = now_local.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            if scheduled <= now_local:
                scheduled += timedelta(days=1)
            
            # Convert to UTC for storage
            return scheduled.astimezone(timezone.utc)
        except Exception as e:
            logger.warning(f"Invalid timezone '{user_timezone}': {e}, falling back to UTC")
    
    # Fallback to UTC
    now = datetime.now(timezone.utc)
    scheduled = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    if scheduled <= now:
        scheduled += timedelta(days=1)

    return scheduled