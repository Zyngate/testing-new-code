from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
    date: str              # YYYY-MM-DD
    time: str              # HH:MM
    frequency: Optional[str] = "once"   # once | daily | weekly | monthly
    days: Optional[List[str]] = []
    date_of_month: Optional[int] = None   # for monthly
    category: Optional[str] = None


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------



from fastapi.encoders import jsonable_encoder
from datetime import datetime

@router.get("/")
def get_tasks(user_id: str):
    tasks_col, _ = get_or_init_sync_collections()

    tasks = list(tasks_col.find({"user_id": user_id}))

    scheduled = []
    completed = []

    for task in tasks:
        task["_id"] = str(task["_id"])

        if isinstance(task.get("scheduled_datetime"), datetime):
            task["scheduled_datetime"] = task["scheduled_datetime"].isoformat()

        # 🔴 THIS FIELD MUST EXIST
        if "status" not in task:
            task["status"] = "completed" if task.get("retrieved") else "scheduled"

        if task["status"] == "completed":
            completed.append(task)
        else:
            scheduled.append(task)

    return scheduled + completed



@router.post("/")
def create_task(request: TaskCreateRequest):
    tasks_col, _ = get_or_init_sync_collections()
    if tasks_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        scheduled_datetime = datetime.strptime(
            f"{request.date} {request.time}",
            "%Y-%m-%d %H:%M"
        ).replace(tzinfo=timezone.utc)  # ✅ REQUIRED FIX
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date or time format"
        )

    task_doc = {
        "user_id": request.user_id,
        "description": request.description,
        "scheduled_datetime": scheduled_datetime,
        "frequency": request.frequency,
        "days": request.days or [],
        "date": request.date_of_month,
        "category": request.category,
        "retrieved": False,
    }

    result = tasks_col.insert_one(task_doc)
    logger.info(f"Task created: {result.inserted_id}")

    return JSONResponse(
        content={
            "status": "Task scheduled",
            "task_id": str(result.inserted_id),
            "scheduled_datetime": scheduled_datetime.isoformat()
        },
        status_code=201
    )

@router.get("/blogs")
def get_generated_content(task_id: str):
    _, output_col = get_or_init_sync_collections()

    blogs = list(output_col.find(
        {"task_id": ObjectId(task_id)}
    ))

    for blog in blogs:
        blog["_id"] = str(blog["_id"])
        blog["task_id"] = str(blog["task_id"])

    return blogs

