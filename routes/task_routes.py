from datetime import datetime, timezone
from typing import Optional, List
from dateutil import parser
from datetime import timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from services.task_utils import generate_task_name
from database import get_or_init_sync_collections
from services.task_service import normalize_prompt
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
    scheduled_datetime: str  # ISO 8601 with timezone
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
        scheduled_datetime = parser.isoparse(
            request.scheduled_datetime
        ).astimezone(timezone.utc)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid scheduled_datetime. Must be ISO 8601 with timezone."
        )
    # ✅ ALWAYS generate task_name here (ONCE)
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

