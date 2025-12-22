# stelle_backend/routes/task_routes.py

from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database import get_or_init_sync_collections
from config import logger

router = APIRouter(prefix="/tasks", tags=["Tasks"])

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

@router.get("/tasks")
def get_tasks():
    tasks_col, _ = get_or_init_sync_collections()
    if tasks_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    tasks = list(tasks_col.find({}, {"_id": 0}))
    return JSONResponse(content=tasks)


@router.post("/tasks")
def create_task(request: TaskCreateRequest):
    tasks_col, _ = get_or_init_sync_collections()
    if tasks_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        scheduled_datetime = datetime.strptime(
            f"{request.date} {request.time}",
            "%Y-%m-%d %H:%M"
        )
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
def get_generated_content():
    _, blogs_col = get_or_init_sync_collections()
    if blogs_col is None:
        raise HTTPException(status_code=503, detail="Database unavailable")

    blogs = list(blogs_col.find({}, {"_id": 0}))
    return JSONResponse(content=blogs)
