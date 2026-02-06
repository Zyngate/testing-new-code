# test_blog_insert.py

from datetime import datetime, timezone
from bson import ObjectId
import time

from database import get_or_init_sync_collections
from services.task_service import execute_task

def test_blog_storage():
    tasks_col, blogs_col = get_or_init_sync_collections()

    print("📦 Blogs collection name:", blogs_col.name)

    # Create a test task
    task_id = ObjectId()
    task = {
        "_id": task_id,
        "user_id": "test-user-blogs",
        "task_name": "Test Blog Insert",
        "description": "give me a simple hello world program in c++",
        "normalized_prompt": "Write a simple C++ hello world program with explanation.",
        "scheduled_datetime": datetime.now(timezone.utc),
        "frequency": "once",
        "days": [],
        "retrieved": False,
        "status": "scheduled",
        "run_count": 0,
    }

    # Insert task
    tasks_col.insert_one(task)
    print("✅ Task inserted:", task_id)

    # Execute task DIRECTLY (no scheduler)
    execute_task(task)

    # Give DB a moment
    time.sleep(2)

    # Check blogs collection
    blog = blogs_col.find_one(
        {"task_id": task_id},
        sort=[("created_at", -1)]
    )

    if blog:
        print("🎉 BLOG SAVED SUCCESSFULLY")
        print("Blog ID:", blog["_id"])
        print("Content preview:\n", blog["content"][:300])
    else:
        print("❌ BLOG NOT FOUND")

if __name__ == "__main__":
    test_blog_storage()