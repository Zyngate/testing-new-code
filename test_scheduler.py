from datetime import datetime, timedelta
import time
from bson import ObjectId

from database import get_or_init_sync_collections
from services.task_service import scheduler_loop

# ----------------------------
# Insert scheduled task
# ----------------------------

tasks_col, _ = get_or_init_sync_collections()

task = {
    "_id": ObjectId(),
    "user_id": "test-user-456",
    "description": "Write a motivational quote",
    "scheduled_datetime": datetime.now() + timedelta(seconds=10),
    "frequency": None,
    "days": [],
    "retrieved": False
}

tasks_col.insert_one(task)
print("⏱ Task scheduled for 10 seconds later")

# ----------------------------
# Start scheduler
# ----------------------------

print("🚀 Scheduler started. Waiting...")
scheduler_loop()  # runs forever
