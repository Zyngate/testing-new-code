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
    """Generate content using Stelle LLM."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY_STELLE_MODEL}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are Stelle, an intelligent assistant that completes scheduled tasks."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def generate_task_name(description: str) -> str:
    """Generate a short, human-readable task name from user description."""
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
                return datetime.combine(
                    candidate_date,
                    base_time,
                    tzinfo=timezone.utc
                )

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

    # once
    return None

# -------------------------------------------------------------------
# Core Task Execution
# -------------------------------------------------------------------

def execute_task(task: Dict):
    """Execute a single scheduled task safely."""

    tasks_col, output_col = get_or_init_sync_collections()
    if tasks_col is None or output_col is None:
        logger.error("DB unavailable. Task skipped.")
        return

    # 🔒 Lock the task to prevent duplicate execution
    locked = tasks_col.find_one_and_update(
        {
            "_id": task["_id"],
            "retrieved": False,
            "status": {"$ne": "running"}
        },
        {
            "$set": {
                "status": "running",
                "last_run_at": datetime.now(timezone.utc)
            }
        }
    )

    if not locked:
        return

    user_id = task.get("user_id")
    description = task.get("description")

    if not user_id or not description:
        logger.error(f"Invalid task payload: {task.get('_id')}")
        return

    logger.info(f"Executing task {task['_id']} for user_id={user_id}")

    try:
        # 🔒 Small delay to avoid Groq rate-limit bursts
        time.sleep(2)

        # 🔢 Increment execution count (state)
        run_count = task.get("run_count", 0) + 1
        tasks_col.update_one(
            {"_id": task["_id"]},
            {"$set": {"run_count": run_count}}
        )

        # ✅ NORMALIZED EXECUTOR PROMPT
        executor_prompt = f"""
You are executing ONE step of a recurring task.

User goal:
{description}

Execution number:
{run_count}

Context:
This task runs repeatedly over time.
Each execution should progress the user toward the goal.

Rules:
- Produce content for THIS execution only
- Do NOT include multiple days or future steps
- Do NOT use lists or numbering unless absolutely necessary
- Do NOT mention recurrence, frequency, or time
- Assume this is the next logical step in a sequence
- Keep the output concise, focused, and actionable

Output:
Return only the content for this execution.
"""

        result = ask_stelle(executor_prompt)

        # 🧹 CLEAN OLD OUTPUTS FOR RECURRING TASKS
        if task.get("frequency") in ("daily", "weekly", "monthly"):
            output_col.delete_many(
                {"task_id": task["_id"]}
            )

        # ✅ STORE NEW RESULT
        output_col.insert_one({
            "task_id": task["_id"],
            "user_id": user_id,
            "content": result,
            "created_at": datetime.now(timezone.utc)
        })

        # 🔁 RESCHEDULE OR COMPLETE
        next_run = calculate_next_run(
            task.get("scheduled_datetime"),
            task.get("frequency"),
            task.get("days"),
            task.get("date")
        )

        if next_run:
            tasks_col.update_one(
                {"_id": task["_id"]},
                {"$set": {
                    "scheduled_datetime": next_run,
                    "retrieved": False,
                    "status": "scheduled"
                }}
            )
            logger.info(f"Task rescheduled for {next_run}")
        else:
            tasks_col.update_one(
                {"_id": task["_id"]},
                {"$set": {
                    "status": "completed",
                    "retrieved": True
                }}
            )
            logger.info(f"One-time task completed: {task['_id']}")

    except Exception as e:
        logger.exception(f"Task execution failed: {task['_id']}")
        tasks_col.update_one(
            {"_id": task["_id"]},
            {"$set": {
                "status": "scheduled",
                "retrieved": False,
                "last_error": str(e)
            }}
        )
        return

# -------------------------------------------------------------------
# Scheduler Loop
# -------------------------------------------------------------------

def scheduler_loop():
    """Background scheduler loop."""
    logger.info("Task scheduler started.")

    while True:
        try:
            tasks_col, _ = get_or_init_sync_collections()
            if tasks_col is None:
                time.sleep(SCHEDULER_POLL_INTERVAL)
                continue

            now = datetime.now(timezone.utc)

            due_tasks = list(tasks_col.find({
                "scheduled_datetime": {"$lte": now},
                "status": "scheduled"
            }))


            for task in due_tasks:
                threading.Thread(
                    target=execute_task,
                    args=(task,),
                    daemon=True
                ).start()

        except Exception:
            logger.exception("Scheduler loop error")

        time.sleep(SCHEDULER_POLL_INTERVAL)


# -------------------------------------------------------------------
# Thread Bootstrap
# -------------------------------------------------------------------

task_thread = threading.Thread(
    target=scheduler_loop,
    daemon=True
)
