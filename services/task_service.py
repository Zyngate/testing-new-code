import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

import requests
from dateutil.relativedelta import relativedelta

from config import logger, GROQ_API_KEY_STELLE_MODEL
from database import get_or_init_sync_collections

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

CALENDAR_DAY_MAP = {
    "Mon": 0, "Tue": 1, "Wed": 2,
    "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6
}

SCHEDULER_POLL_INTERVAL = 30  # seconds


# -------------------------------------------------------------------
# LLM Execution
# -------------------------------------------------------------------
def ask_stelle(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY_STELLE_MODEL}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1800,
        "messages": [
            {
                "role": "system",
                "content": "You are a task execution engine that completes tasks fully and decisively."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask_stelle_normalizer(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY_STELLE_MODEL}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 400,
        "messages": [
            {
                "role": "system",
                "content": "You rewrite user requests into clear, dense execution briefs."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


EXECUTION_CONTRACT = """
You are a TASK EXECUTION ENGINE.

Your job is to COMPLETE the user’s task fully and immediately.

NON-NEGOTIABLE RULES:
- You MUST NOT ask questions
- You MUST NOT request clarification
- You MUST NOT defer decisions to the user
- If information is missing, make reasonable assumptions
- Respect all constraints explicitly mentioned (numbers, people, format)
- NEVER give an overview when the user asked for output
- NEVER explain what you are about to do
- NEVER say "it depends"
- Produce a COMPLETE, USABLE result in one response
"""

DEPTH_AND_SIZE_RULES = """
CONTENT DEPTH REQUIREMENTS:
- Output must be DETAILED and SUBSTANTIAL
- Target length: 700–1000 words unless the task is inherently short
- Use multiple sections with clear headings
- Each section must add new information
- No filler, no repetition, no padding
"""


def normalize_prompt(user_prompt: str) -> str:
    improver_prompt = f"""
You are converting a user's request into a DENSE EXECUTION BRIEF
for a high-quality content generation system.

Your job is NOT to shorten.
Your job is to CLARIFY, ENRICH, and MAKE EXECUTION-READY.

Rules:
- Preserve original intent and topic
- Do NOT ask questions
- Do NOT add new topics
- Make reasonable assumptions when missing details
- Expand implicit requirements into explicit ones
- Include:
  • expected format
  • depth level
  • practical vs theoretical balance
  • target audience (assume intelligent general audience if unspecified)
- Do NOT mention time, frequency, or recurrence
- Do NOT explain the brief
- Output ONLY the execution brief

User request:
{user_prompt}

Now write the execution brief:
"""
    return ask_stelle_normalizer(improver_prompt).strip()


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


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# -------------------------------------------------------------------
# Scheduling Logic
# -------------------------------------------------------------------

def next_daily_run(scheduled_dt: datetime, now: datetime) -> datetime:
    base_time = scheduled_dt.time()
    candidate = datetime.combine(now.date(), base_time, tzinfo=timezone.utc)

    if candidate <= now:
        candidate += timedelta(days=1)

    return candidate


def calculate_next_run(
    scheduled_dt: datetime,
    frequency: Optional[str],
    days: Optional[List[str]],
    date: Optional[int] = None
) -> Optional[datetime]:

    now = datetime.now(timezone.utc)
    scheduled_dt = ensure_utc(scheduled_dt)

    if frequency == "daily":
        return next_daily_run(scheduled_dt, now)

    if frequency == "weekly":
        if not days:
            return None

        base_time = scheduled_dt.time()
        valid_days = [CALENDAR_DAY_MAP[d[:3]] for d in days if d[:3] in CALENDAR_DAY_MAP]

        for i in range(1, 8):
            candidate_date = now.date() + timedelta(days=i)
            if candidate_date.weekday() in valid_days:
                return datetime.combine(candidate_date, base_time, tzinfo=timezone.utc)

        return None

    if frequency == "monthly":
        base_time = scheduled_dt.time()
        run_day = date or scheduled_dt.day

        candidate = now + relativedelta(months=1)
        last_day = (candidate + relativedelta(day=31)).day
        safe_day = min(run_day, last_day)

        return datetime(
            candidate.year,
            candidate.month,
            safe_day,
            base_time.hour,
            base_time.minute,
            tzinfo=timezone.utc
        )

    return None


# -------------------------------------------------------------------
# Core Task Execution
# -------------------------------------------------------------------
def execute_task(task: Dict):
    # 🔐 State safety check (PAUSE / RESUME SUPPORT)
    if task.get("status") != "scheduled":
        logger.info(
            f"Task {task.get('_id')} skipped due to state: {task.get('status')}"
        )
        return

    tasks_col, output_col = get_or_init_sync_collections()
    if tasks_col is None or output_col is None:
        logger.error("DB unavailable. Task skipped.")
        return

    locked = tasks_col.find_one_and_update(
        {"_id": task["_id"], "retrieved": False, "status": {"$ne": "running"}},
        {"$set": {"status": "running", "last_run_at": datetime.now(timezone.utc)}}
    )

    if not locked:
        return

    user_id = task.get("user_id")
    user_description = task.get("description")

    if not user_id or not user_description:
        logger.error(f"Invalid task payload: {task.get('_id')}")
        return

    normalized_prompt = task.get("normalized_prompt")
    if not normalized_prompt or len(normalized_prompt.split()) < 30:
        logger.warning(f"Weak normalized prompt detected for task {task['_id']}")
        normalized_prompt = normalize_prompt(user_description)
        tasks_col.update_one(
            {"_id": task["_id"]},
            {"$set": {"normalized_prompt": normalized_prompt}}
        )

    try:
        time.sleep(2)

        run_count = task.get("run_count", 0) + 1
        tasks_col.update_one(
            {"_id": task["_id"]},
            {"$set": {"run_count": run_count}}
        )

        executor_prompt = f"""
{EXECUTION_CONTRACT}

{DEPTH_AND_SIZE_RULES}

USER TASK GOAL:
{normalized_prompt}

EXECUTION NUMBER:
{run_count}

FINAL INSTRUCTION:
Return ONLY the final content.
"""

        result = ask_stelle(executor_prompt)

        if len(result.split()) < 500:
            logger.warning("Short output detected. Retrying once.")
            result = ask_stelle(
                executor_prompt
                + "\n\nIMPORTANT: Expand the content significantly with more depth and examples."
            )

        output_col.insert_one({
            "task_id": task["_id"],
            "user_id": user_id,
            "content": result,
            "created_at": datetime.now(timezone.utc)
        })

        next_run = calculate_next_run(
            task.get("scheduled_datetime"),
            task.get("frequency"),
            task.get("days"),
            task.get("date")
        )

        if next_run:
            tasks_col.update_one(
                {"_id": task["_id"]},
                {"$set": {"scheduled_datetime": next_run, "retrieved": False, "status": "scheduled"}}
            )
        else:
            tasks_col.update_one(
                {"_id": task["_id"]},
                {"$set": {"status": "completed", "retrieved": True}}
            )

    except Exception as e:
        logger.exception(f"Task execution failed: {task['_id']}")
        tasks_col.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "scheduled", "retrieved": False, "last_error": str(e)}}
        )


# -------------------------------------------------------------------
# Scheduler Loop
# -------------------------------------------------------------------

def scheduler_loop():
    logger.info("Task scheduler started.")

    while True:
        try:
            tasks_col, _ = get_or_init_sync_collections()
            if tasks_col is None:
                time.sleep(SCHEDULER_POLL_INTERVAL)
                continue

            now = datetime.now(timezone.utc)
            grace_period = now - timedelta(seconds=10)

            due_tasks = list(tasks_col.find({
                "scheduled_datetime": {"$lte": grace_period},
                "status": "scheduled"
            }))

            for task in due_tasks:
                threading.Thread(target=execute_task, args=(task,), daemon=True).start()

        except Exception:
            logger.exception("Scheduler loop error")

        time.sleep(SCHEDULER_POLL_INTERVAL)


def start_task_scheduler():
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
