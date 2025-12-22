# stelle_backend/services/task_service.py

import threading
import time
from datetime import datetime, timedelta
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
    """
    Generate a short, human-readable task name from user description.
    """
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



# -------------------------------------------------------------------
# Scheduling Logic
# -------------------------------------------------------------------

def calculate_next_run(
    scheduled_dt: datetime,
    frequency: Optional[str],
    days: Optional[List[str]],
    date: Optional[int] = None
) -> Optional[datetime]:
    """Calculate next execution time based on clean recurrence rules."""

    now = datetime.now()

    # -------------------------
    # DAILY
    # -------------------------
    if frequency == "daily":
        next_dt = scheduled_dt + timedelta(days=1)
        return next_dt if next_dt > now else None

    # -------------------------
    # WEEKLY (uses days)
    # -------------------------
    if frequency == "weekly":
        if not days:
            return None

        valid_days = [CALENDAR_DAY_MAP[d[:3]] for d in days if d[:3] in CALENDAR_DAY_MAP]
        next_day = scheduled_dt.date() + timedelta(days=1)

        while True:
            if next_day.weekday() in valid_days:
                next_dt = datetime.combine(next_day, scheduled_dt.time())
                return next_dt if next_dt > now else None
            next_day += timedelta(days=1)

    # -------------------------
    # MONTHLY (uses date-of-month)
    # -------------------------
    if frequency == "monthly":
        run_date = date or scheduled_dt.day

        candidate = scheduled_dt + relativedelta(months=1)
        last_day = (candidate + relativedelta(day=31)).day

        safe_day = min(run_date, last_day)

        next_dt = candidate.replace(day=safe_day)
        return next_dt if next_dt > now else None

    # -------------------------
    # ONE-TIME
    # -------------------------
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

    locked = tasks_col.find_one_and_update(
        {"_id": task["_id"], "retrieved": False},
        {"$set": {"retrieved": True, "last_run_at": datetime.now()}}
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
        result = ask_stelle(description)

        output_col.insert_one({
            "task_id": task["_id"],
            "user_id": user_id,
            "content": result,
            "created_at": datetime.now()
        })

        next_run = calculate_next_run(
            task["scheduled_datetime"],
            task.get("frequency"),
            task.get("days"),
            task.get("date")
        )


        if next_run:
            tasks_col.update_one(
                {"_id": task["_id"]},
                {"$set": {
                    "scheduled_datetime": next_run,
                    "retrieved": False
                }}
            )
            logger.info(f"Task rescheduled for {next_run}")
        else:
            logger.info(f"One-time task completed: {task['_id']}")

    except Exception as e:
        logger.exception(f"Task execution failed: {task['_id']}")
        tasks_col.update_one(
            {"_id": task["_id"]},
            {"$set": {"retrieved": False, "last_error": str(e)}}
        )

    task_name = task.get("task_name")
    if not task_name:
        try:
            task_name = generate_task_name(description)
            tasks_col.update_one(
                {"_id": task["_id"]},
                {"$set": {"task_name": task_name}}
            )
        except Exception:
            logger.exception("Failed to generate task name")
            task_name = "Automated Task"



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

            now = datetime.now()
            due_tasks = list(tasks_col.find({
                "scheduled_datetime": {"$lte": now},
                "retrieved": False
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
