from datetime import datetime, timezone
from typing import Optional, List
from dateutil import parser

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.task_utils import generate_task_name
from services.task_service import normalize_prompt, ask_stelle
from database import get_or_init_sync_collections
from config import logger
from bson import ObjectId

router = APIRouter(tags=["Tasks"])

# -------------------------------------------------------------------
# Request Models
# -------------------------------------------------------------------

class TaskCreateRequest(BaseModel):
    user_id: str
    task_name: Optional[str] = None
    description: str
    scheduled_datetime: str
    frequency: Optional[str] = "once"
    days: Optional[List[str]] = []
    date_of_month: Optional[int] = None
    category: Optional[str] = None


class ContentChatRequest(BaseModel):
    task_id: str
    user_message: str


class TaskUpdateRequest(BaseModel):
    scheduled_datetime: Optional[str] = None
    frequency: Optional[str] = None
    days: Optional[List[str]] = None
    date_of_month: Optional[int] = None
    category: Optional[str] = None


class ManualContentUpdateRequest(BaseModel):
    task_id: str
    content: str


def build_editor_prompt(content: str, user_message: str) -> str:
    return f"""
You are a senior content editor.

Original content:
{content}

User request:
{user_message}

Editing rules:
- Improve clarity, depth, and usefulness
- Expand sections if needed
- Add concrete examples where appropriate
- Preserve the original topic and intent
- Do NOT introduce unrelated ideas
- Do NOT explain what you changed
- Do NOT mention this is an edit
- Output ONLY the revised content
"""


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.get("/")
def get_tasks(user_id: str):
    tasks_col, _ = get_or_init_sync_collections()
    tasks = list(tasks_col.find({"user_id": user_id}))

    scheduled, completed = [], []

    for task in tasks:
        task["_id"] = str(task["_id"])

        if isinstance(task.get("scheduled_datetime"), datetime):
            task["scheduled_datetime"] = (
                task["scheduled_datetime"]
                .astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
            )

        if "status" not in task:
            task["status"] = "completed" if task.get("retrieved") else "scheduled"

        (completed if task["status"] == "completed" else scheduled).append(task)

    return scheduled + completed


@router.post("/")
def create_task(request: TaskCreateRequest):
    tasks_col, _ = get_or_init_sync_collections()
    if tasks_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        scheduled_datetime = parser.isoparse(
            request.scheduled_datetime
        ).astimezone(timezone.utc)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid scheduled_datetime. Must be ISO 8601 with timezone."
        )

    task_name = request.task_name or generate_task_name(request.description)

    normalized_prompt = (
        normalize_prompt(request.description)
        if request.frequency != "once"
        else request.description
    )

    task_doc = {
        "user_id": request.user_id,
        "task_name": task_name,
        "description": request.description,
        "normalized_prompt": normalized_prompt,
        "scheduled_datetime": scheduled_datetime,
        "frequency": request.frequency,
        "days": request.days or [],
        "date": request.date_of_month,
        "category": request.category,
        "retrieved": False,
        "status": "scheduled",
        "run_count": 0
    }

    result = tasks_col.insert_one(task_doc)

    return {
        "status": "Task scheduled",
        "task_id": str(result.inserted_id),
        "scheduled_datetime": scheduled_datetime.isoformat()
    }


@router.get("/blogs")
def get_generated_content(task_id: str):
    _, output_col = get_or_init_sync_collections()

    blog = output_col.find_one(
        {"task_id": ObjectId(task_id)},
        sort=[("created_at", -1)]
    )

    if not blog:
        return []

    blog["_id"] = str(blog["_id"])
    blog["task_id"] = str(blog["task_id"])

    return [blog]


@router.post("/content/chat")
def refine_task_content(request: ContentChatRequest):
    tasks_col, output_col = get_or_init_sync_collections()

    try:
        task_id = ObjectId(request.task_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid task_id")

    task = tasks_col.find_one({"_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    blog = output_col.find_one(
        {"task_id": task_id},
        sort=[("created_at", -1)]
    )

    if not blog:
        raise HTTPException(
            status_code=400,
            detail="No generated content available for this task"
        )

    prompt = build_editor_prompt(blog["content"], request.user_message)

    try:
        updated_content = ask_stelle(prompt)

        if len(updated_content.split()) < 500:
            logger.warning("Short edited content detected. Retrying once.")
            updated_content = ask_stelle(
                prompt + "\n\nIMPORTANT: Expand the content significantly with depth and examples."
            )

    except Exception:
        logger.exception("Content edit failed")
        raise HTTPException(status_code=500, detail="AI edit failed")

    output_col.update_one(
        {"_id": blog["_id"]},
        {"$set": {"content": updated_content, "updated_at": datetime.now(timezone.utc)}}
    )

    return {"content": updated_content}


@router.put("/content")
def update_content_manually(request: ManualContentUpdateRequest):
    _, output_col = get_or_init_sync_collections()

    try:
        task_id = ObjectId(request.task_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid task_id")

    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    blog = output_col.find_one(
        {"task_id": task_id},
        sort=[("created_at", -1)]
    )

    if not blog:
        raise HTTPException(status_code=404, detail="No generated content found")

    output_col.update_one(
        {"_id": blog["_id"]},
        {"$set": {"content": request.content, "updated_at": datetime.now(timezone.utc)}}
    )

    return {"status": "Content updated successfully", "content": request.content}


@router.put("/{task_id}")
def update_task(task_id: str, request: TaskUpdateRequest):
    tasks_col, _ = get_or_init_sync_collections()

    try:
        task_oid = ObjectId(task_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid task_id")

    task = tasks_col.find_one({"_id": task_oid})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    update_fields = {}

    if request.scheduled_datetime:
        update_fields["scheduled_datetime"] = parser.isoparse(
            request.scheduled_datetime
        ).astimezone(timezone.utc)

    if request.frequency:
        update_fields["frequency"] = request.frequency
        update_fields["days"] = request.days or []
        update_fields["date"] = request.date_of_month

    if request.category is not None:
        update_fields["category"] = request.category

    if not update_fields:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    update_fields["status"] = "scheduled"
    update_fields["retrieved"] = False

    tasks_col.update_one({"_id": task_oid}, {"$set": update_fields})

    return {"status": "Task updated successfully", "task_id": task_id}

@router.post("/{task_id}/state")
def toggle_task_state(task_id: str):
    tasks_col, _ = get_or_init_sync_collections()

    try:
        task_oid = ObjectId(task_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid task_id")

    task = tasks_col.find_one({"_id": task_oid})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    current_status = task.get("status")

    # 🚫 Protect execution integrity
    if current_status == "running":
        raise HTTPException(
            status_code=409,
            detail="Task is currently running and cannot be paused or resumed"
        )

    if current_status == "completed":
        raise HTTPException(
            status_code=400,
            detail="Completed tasks cannot be modified"
        )

    # 🔁 Toggle logic
    if current_status == "paused":
        update = {
            "status": "scheduled",
            "retrieved": False
        }
        message = "Task resumed successfully"

    elif current_status == "scheduled":
        update = {
            "status": "paused"
        }
        message = "Task paused successfully"

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task state: {current_status}"
        )

    tasks_col.update_one({"_id": task_oid}, {"$set": update})

    return {
        "status": message,
        "new_state": update["status"]
    }


@router.delete("/{task_id}")
def delete_task(task_id: str):
    """
    Delete a task permanently.
    
    - Running tasks cannot be deleted (wait for completion or pause first)
    - All other states (scheduled, paused, completed) can be deleted
    """
    tasks_col, _ = get_or_init_sync_collections()

    try:
        task_oid = ObjectId(task_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid task_id")

    task = tasks_col.find_one({"_id": task_oid})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    current_status = task.get("status", "scheduled")

    # Protect running tasks
    if current_status == "running":
        raise HTTPException(
            status_code=409,
            detail="Task is currently running. Wait for completion or pause first."
        )

    # Delete the task
    result = tasks_col.delete_one({"_id": task_oid})

    if result.deleted_count == 0:
        raise HTTPException(status_code=500, detail="Failed to delete task")

    logger.info(f"🗑️ Task deleted: {task_id} (was {current_status})")

    return {
        "success": True,
        "message": "Task deleted successfully",
        "task_id": task_id
    }
