# stelle_backend/routes/task_routes.py
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

# CORRECTED IMPORTS
from database import get_or_init_sync_collections # <-- Import the getter function
from services.task_service import (
    send_calendar_link, 
    save_task_to_db, 
    # Only import necessary helpers/constants if they are outside classes/functions
    logger 
)

router = APIRouter()

# --- Pydantic Models for Request Body ---
class TaskCreateRequest(BaseModel):
    user_email: EmailStr
    description: str
    date: str
    time: str
    frequency: Optional[str] = "once"
    days: Optional[List[str]] = []
    category: Optional[str] = None
    send_calendar: Optional[bool] = False

# --- Endpoints ---

@router.get("/tasks", response_model=Dict[str, List[Dict[str, Any]]])
def get_tasks_endpoint():
    """Returns all tasks from the synchronous task collection."""
    tasks_collection_sync, _ = get_or_init_sync_collections() # <-- FIXED USAGE (Error 1)
    
    if tasks_collection_sync is not None:
        tasks_list = list(tasks_collection_sync.find({}, {"_id": 0}))
        return JSONResponse(content=tasks_list)
    
    raise HTTPException(status_code=503, detail="Task service database unavailable.")

@router.post("/tasks", response_model=Dict[str, Any])
def create_task_endpoint(request: TaskCreateRequest):
    """Creates a new task in the synchronous collection."""
    tasks_collection_sync, _ = get_or_init_sync_collections() # <-- FIXED USAGE (Error 2)
    
    if tasks_collection_sync is None:
        raise HTTPException(status_code=503, detail="Task service database unavailable.")

    task = request.dict(exclude_none=True)
    
    try:
        scheduled_datetime = datetime.strptime(f"{task['date']}T{task['time']}", "%Y-%m-%dT%H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date or time format. Use YYYY-MM-DD and HH:MM.")
        
    task_doc = {
        "user_email": task['user_email'],
        "description": task['description'],
        "scheduled_time": scheduled_datetime.strftime("%Y-%m-%d %H:%M"),
        "scheduled_datetime": scheduled_datetime,
        "frequency": task.get("frequency", "once"),
        "days": task.get("days", []),
        "category": task.get("category"),
        "retrieved": False
    }
    
    result = tasks_collection_sync.insert_one(task_doc)
    task_doc["_id"] = result.inserted_id
    
    if task.get("send_calendar"):
        send_calendar_link(task['user_email'], task_doc)
        
    return JSONResponse(content={"status": "Task added", "task_id": str(task_doc["_id"])}, status_code=201)

@router.post("/tasks/calendar", response_model=Dict[str, str])
def create_task_with_calendar_endpoint(request: TaskCreateRequest):
    """Creates a task and immediately returns a Google Calendar link."""
    tasks_collection_sync, _ = get_or_init_sync_collections() # <-- FIXED USAGE (Error 3)

    if tasks_collection_sync is None:
        raise HTTPException(status_code=503, detail="Task service database unavailable.")
        
    task = request.dict(exclude_none=True)

    try:
        scheduled_datetime = datetime.strptime(f"{task['date']} {task['time']}", "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date or time format. Use YYYY-MM-DD and HH:MM.")

    task_doc = {
        "user_email": task['user_email'],
        "description": task['description'],
        "scheduled_time": scheduled_datetime.strftime("%Y-%m-%d %H:%M"),
        "scheduled_datetime": scheduled_datetime,
        "frequency": task.get("frequency", "once"),
        "retrieved": False
    }
    
    result = tasks_collection_sync.insert_one(task_doc)
    task_doc["_id"] = result.inserted_id

    start = scheduled_datetime.strftime("%Y%m%dT%H%M%S")
    end = (scheduled_datetime + timedelta(minutes=30)).strftime("%Y%m%dT%H%M%S")
    link = f"https://calendar.google.com/calendar/render?action=TEMPLATE&text={task['description'].replace(' ','+')}&dates={start}/{end}"
    
    return JSONResponse(content={"status": "Task added", "calendar_link": link, "task_id": str(task_doc["_id"])})

@router.get("/blogs", response_model=Dict[str, List[Dict[str, Any]]])
def get_blogs_endpoint():
    """Returns all generated blog/content entries."""
    _, blogs_collection_sync = get_or_init_sync_collections() # <-- FIXED USAGE (Error 4)
    
    if blogs_collection_sync is not None:
        blogs_list = list(blogs_collection_sync.find({}, {"_id": 0}))
        return JSONResponse(content=blogs_list)
        
    raise HTTPException(status_code=503, detail="Task service database unavailable.")