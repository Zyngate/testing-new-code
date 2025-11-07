# stelle_backend/services/task_service.py
import threading
import time
from datetime import datetime, timedelta, date, time as dt_time, datetime as datetime_type
import requests
import json
import webbrowser
import speech_recognition as sr

from config import logger, GROQ_API_KEY_STELLE_MODEL
from database import get_or_init_sync_collections
from services.common_utils import get_current_datetime

# Optional: For Windows notifications (if desired)
try:
    from win10toast import ToastNotifier
    notifier = ToastNotifier()
except ImportError:
    notifier = None  # Notifications will be skipped if not installed

# --- Globals ---
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
calendar_day_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


# ====================================================================
# 1. UTILITY FUNCTIONS
# ====================================================================

def ask_stelle(prompt: str) -> str:
    """Sends a synchronous request to Groq API for content generation."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY_STELLE_MODEL}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are Stelle, an intelligent assistant that completes recurring tasks for users."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Error in ask_stelle: {e}")
        return "Error in API response: Failed to connect to Groq."
    except (KeyError, IndexError, ValueError) as e:
        response_text = getattr(response, "text", "N/A") if 'response' in locals() else 'N/A'
        logger.error(f"JSON Parsing Error in ask_stelle: {e}\nResponse: {response_text}")
        return f"Error in API response: {response_text}"


def get_next_run_time(task: dict, days: list) -> datetime_type | None:
    """Calculates the next scheduled run time for a recurring task."""
    now = datetime.now()
    freq = task.get("frequency")
    dt = task["scheduled_datetime"]
    next_dt = None

    if days:
        day_numbers = [calendar_day_map[d[:3]] for d in days if d[:3] in calendar_day_map]
        current_day = dt.date()
        next_day = current_day + timedelta(days=1)

        while True:
            if next_day.weekday() in day_numbers:
                next_dt = datetime.combine(next_day, dt.time())
                break
            next_day += timedelta(days=1)

    elif freq == "daily":
        next_dt = dt + timedelta(days=1)
    elif freq == "weekly":
        next_dt = dt + timedelta(weeks=1)
    elif freq == "monthly":
        next_dt = dt + timedelta(weeks=4)
    elif freq == "once" or not freq:
        return None

    if next_dt and next_dt <= now:
        time_diff = now - next_dt
        if freq == "daily" or days:
            next_dt += timedelta(days=time_diff.days + 1)
        elif freq == "weekly":
            next_dt += timedelta(weeks=(time_diff.days // 7) + 1)
        elif freq == "monthly":
            next_dt += timedelta(weeks=((time_diff.days // 7) // 4 + 1) * 4)

    return next_dt


def save_task_to_db(task: dict):
    """Saves or updates a task document in the synchronous task collection."""
    tasks_collection_sync, _ = get_or_init_sync_collections()
    if tasks_collection_sync is not None:
        tasks_collection_sync.update_one(
            {"_id": task.get("_id")},
            {"$set": task},
            upsert=True
        )
    else:
        logger.error("Cannot save task: Synchronous DB connection unavailable.")


def send_calendar_link(user_email: str, task: dict):
    """Opens a Google Calendar link in the default browser."""
    start_dt = task["scheduled_datetime"].strftime("%Y%m%dT%H%M%S")
    end_dt = (task["scheduled_datetime"] + timedelta(minutes=30)).strftime("%Y%m%dT%H%M%S")
    description = task["description"]
    recur_rule = ""
    freq = task.get("frequency")

    if freq == "daily":
        recur_rule = "&recur=RRULE:FREQ=DAILY"
    elif freq == "weekly":
        recur_rule = "&recur=RRULE:FREQ=WEEKLY"
    elif freq == "monthly":
        recur_rule = "&recur=RRULE:FREQ=MONTHLY"

    link = (
        f"https://calendar.google.com/calendar/render?action=TEMPLATE"
        f"&text={description.replace(' ', '+')}"
        f"&dates={start_dt}/{end_dt}"
        f"&details={description.replace(' ', '+')}"
        f"{recur_rule}"
    )
    webbrowser.open(link)
    logger.info(f"Calendar link opened for: {link}")


# ====================================================================
# 2. CORE SCHEDULER LOGIC
# ====================================================================

def execute_task(task: dict):
    """Executes the core task logic: Generates content, stores it, and schedules next run."""
    tasks_collection_sync, blogs_collection_sync = get_or_init_sync_collections()
    if tasks_collection_sync is None or blogs_collection_sync is None:
        logger.error("Task execution skipped: Synchronous DB collections unavailable.")
        return

    user_email = task['user_email']
    desc = task['description']

    logger.info(f"Running task for {user_email}: {desc}")
    result = ask_stelle(desc)
    logger.info(f"Task Result: {result[:100]}...")

    # Store content in blogs collection
    blogs_collection_sync.insert_one({
        "user_email": user_email,
        "task_description": desc,
        "content": result,
        "created_at": datetime.now()
    })

    # Update task document to include content and mark as retrieved
    tasks_collection_sync.update_one(
        {"_id": task["_id"]},
        {"$set": {"retrieved": True, "content": result}}
    )

    # Notification
    print(f"✅ Task Completed for {user_email}: {desc[:50]}...")
    if notifier:
        notifier.show_toast("Task Completed", desc)

    # Schedule next occurrence
    freq = task.get("frequency")
    days = task.get("days", [])

    task_for_scheduling = tasks_collection_sync.find_one({"_id": task["_id"]})
    if task_for_scheduling and (freq in ["daily", "weekly", "monthly"] or days):
        next_time = get_next_run_time(task_for_scheduling, days)
        if next_time:
            task_for_scheduling["scheduled_datetime"] = next_time
            task_for_scheduling["scheduled_time"] = next_time.strftime("%Y-%m-%d %H:%M")
            task_for_scheduling["retrieved"] = False
            task_for_scheduling.pop("content", None)
            save_task_to_db(task_for_scheduling)
            logger.info(f"Task for {user_email} rescheduled for: {next_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            logger.info(f"Non-recurring task for {user_email} finished.")


def background_scheduler():
    """The main loop for the synchronous task scheduler."""
    while True:
        try:
            tasks_collection_sync, _ = get_or_init_sync_collections()
            if tasks_collection_sync is not None:
                now = datetime.now()
                due_tasks = list(tasks_collection_sync.find({
                    "scheduled_datetime": {"$lte": now},
                    "retrieved": False
                }))
                for task in due_tasks:
                    execute_task(task)
        except Exception as e:
            logger.error(f"Error in background_scheduler loop: {e}")
        time.sleep(30)


def load_tasks():
    """Initializes scheduled_datetime for existing tasks if missing."""
    tasks_collection_sync, _ = get_or_init_sync_collections()
    if tasks_collection_sync is None:
        logger.warning("Skipping initial task load: Synchronous DB connection unavailable.")
        return

    for task in tasks_collection_sync.find():
        if "scheduled_datetime" not in task and "scheduled_time" in task:
            try:
                task["scheduled_datetime"] = datetime.strptime(task["scheduled_time"], "%Y-%m-%d %H:%M")
                save_task_to_db(task)
            except Exception as e:
                logger.error(f"Could not parse scheduled_time for task {task.get('_id')}: {e}")


# --- Scheduler Thread Setup ---
load_tasks()
task_thread = threading.Thread(target=background_scheduler, daemon=True)
# Start in main.py via FastAPI lifecycle
