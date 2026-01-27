from datetime import datetime, timedelta
import time
from bson import ObjectId

from database import get_or_init_sync_collections
from services.task_service import execute_task

# ----------------------------
# Create a sample task
# ----------------------------

tasks_col, output_col = get_or_init_sync_collections()

task = {
    "_id": ObjectId(),
    "user_id": "test-user-123",
    "description": "Give me 3 healthy breakfast ideas",
    "scheduled_datetime": datetime.now(),  # run immediately
    "frequency": None,                     # one-time task
    "days": [],
    "retrieved": False
}

tasks_col.insert_one(task)
print("✅ Task inserted")

# ----------------------------
# Execute task manually
# ----------------------------

execute_task(task)

# ----------------------------
# Verify output
# ----------------------------

time.sleep(2)

result = output_col.find_one({"task_id": task["_id"]})

if result:
    print("🎉 Task executed successfully!")
    print("Generated content:\n")
    print(result["content"])
else:
    print("❌ Task did not execute")
