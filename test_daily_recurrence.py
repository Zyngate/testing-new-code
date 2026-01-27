from datetime import datetime
from bson import ObjectId

from database import get_or_init_sync_collections
from services.task_service import execute_task

tasks_col, output_col = get_or_init_sync_collections()

task = {
    "_id": ObjectId(),
    "user_id": "test-user-weekly",
    "description": "Give me one productivity tip",
    "scheduled_datetime": datetime.now(),
    "frequency": "weekly",
    "days": ["Mon", "Thu"],   # ✅ REQUIRED for weekly
    "retrieved": False
}

tasks_col.insert_one(task)
print("✅ Weekly task inserted")

execute_task(task)

updated = tasks_col.find_one({"_id": task["_id"]})

print("🔁 Next scheduled time:", updated["scheduled_datetime"])
print("Retrieved flag:", updated["retrieved"])
