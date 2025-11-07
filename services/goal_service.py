# stelle_backend/services/goal_service.py
import re
import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import pytz # <-- Necessary for timezone operations

from fastapi import HTTPException
from database import goals_collection, users_collection, notifications_collection
from config import logger
from services.subscription_service import schedule_notification # Helper function for push notifications

async def update_task_goal_status(
    user_id: str,
    session_id: str,
    content: Optional[str],
    command: str,
    goal_id_str: Optional[str],
    task_id_str: Optional[str],
    new_goals_map: Dict[str, str], # Map of temp name to real ID
):
    """
    Processes a single LLM command tag (e.g., [TASK_ADD], [GOAL_COMPLETE]) 
    and updates the goals collection accordingly.
    """
    
    # Helper to resolve IDs created in the same response
    def resolve_id(id_str):
        return new_goals_map.get(id_str, id_str)

    if command == 'set_goal':
        goal_phrase = content
        goal_id = str(uuid.uuid4())
        new_goals_map[goal_phrase] = goal_id 
        
        # Check for duplicates before insertion
        existing_goal = await goals_collection.find_one(
            {"user_id": user_id, "title": goal_phrase, "status": {"$in": ["active", "in progress"]}}
        )
        if existing_goal:
            logger.info(f"Skipping creation of duplicate goal '{goal_phrase}' for user {user_id}.")
            return
            
        new_goal = {
            "user_id": user_id,
            "goal_id": goal_id,
            "session_id": session_id,
            "title": goal_phrase,
            "description": "",
            "status": "active",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "tasks": [],
        }
        await goals_collection.insert_one(new_goal)
        logger.info(f"Goal set: '{goal_phrase}' (ID: {goal_id}) for user {user_id}")
        return

    elif command == 'add_task':
        task_desc = content
        real_goal_id = resolve_id(goal_id_str)
        task_id = str(uuid.uuid4())
        new_task = {
            "task_id": task_id,
            "title": task_desc,
            "description": "",
            "status": "not started",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "deadline": None,
            "progress": [],
        }
        result = await goals_collection.update_one(
            {"user_id": user_id, "goal_id": real_goal_id},
            {"$push": {"tasks": new_task}, "$set": {"updated_at": datetime.now(timezone.utc)}},
        )
        if result.modified_count > 0:
            logger.info(f"Added task '{task_desc}' (ID: {task_id}) to goal {real_goal_id}")
        return

    elif command in ['GOAL_DELETE', 'GOAL_START', 'GOAL_COMPLETE']:
        real_goal_id = resolve_id(goal_id_str)
        
        if command == 'GOAL_DELETE':
            result = await goals_collection.delete_one({"user_id": user_id, "goal_id": real_goal_id})
            if result.deleted_count > 0: logger.info(f"Goal {real_goal_id} deleted.")
            return

        status_map = {'GOAL_START': 'in progress', 'GOAL_COMPLETE': 'completed'}
        new_status = status_map.get(command)
        result = await goals_collection.update_one(
            {"user_id": user_id, "goal_id": real_goal_id},
            {"$set": {"status": new_status, "updated_at": datetime.now(timezone.utc)}},
        )
        if result.modified_count > 0: logger.info(f"Goal {real_goal_id} marked as {new_status}.")
        return

    elif command in ['TASK_DELETE', 'TASK_START', 'TASK_COMPLETE', 'TASK_MODIFY']:
        tid = task_id_str
        
        if command == 'TASK_DELETE':
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {"$pull": {"tasks": {"task_id": tid}}},
            )
            if result.modified_count > 0: logger.info(f"Task {tid} deleted.")
            return

        if command == 'TASK_MODIFY':
            new_desc = content
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {"$set": {"tasks.$.title": new_desc, "tasks.$.updated_at": datetime.now(timezone.utc)}},
            )
            if result.modified_count > 0: logger.info(f"Task {tid} modified to '{new_desc}'.")
            return

        if command in ['TASK_START', 'TASK_COMPLETE']:
            status_map = {'TASK_START': 'in progress', 'TASK_COMPLETE': 'completed'}
            new_status = status_map.get(command)
            result = await goals_collection.update_one(
                {"user_id": user_id, "tasks.task_id": tid},
                {"$set": {"tasks.$.status": new_status, "tasks.$.updated_at": datetime.now(timezone.utc)}},
            )
            if result.modified_count > 0: logger.info(f"Task {tid} marked as {new_status}.")
            return
    
    elif command == 'TASK_DEADLINE':
        tid = task_id_str
        deadline_str = content
        try:
            deadline_dt = datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")
        except ValueError:
            logger.error(f"Invalid deadline format for task {tid}: {deadline_str}. Skipping notification.")
            return
            
        user_info = await users_collection.find_one({"user_id": user_id})
        tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
        try:
            user_tz = pytz.timezone(tz_name)
        except Exception:
            user_tz = pytz.UTC
            
        localized_deadline = user_tz.localize(deadline_dt)
        reminder_time = localized_deadline - timedelta(days=1)
        reminder_time_utc = reminder_time.astimezone(pytz.UTC)
        
        result = await goals_collection.update_one(
            {"user_id": user_id, "tasks.task_id": tid},
            {"$set": {"tasks.$.deadline": deadline_str, "tasks.$.updated_at": datetime.now(timezone.utc)}},
        )
        
        if result.modified_count > 0:
            # We rely on schedule_notification from subscription_service
            await schedule_notification(
                user_id,
                f"Reminder: Task {tid} is due on {deadline_str}",
                reminder_time_utc,
                notif_type="deadline_reminder",
            )
            logger.info(f"Set deadline and scheduled reminder for Task {tid}.")
        return

    elif command == 'TASK_PROGRESS':
        tid = task_id_str
        progress_desc = content
        progress_entry = {"timestamp": datetime.now(timezone.utc), "description": progress_desc}
        
        result = await goals_collection.update_one(
            {"user_id": user_id, "tasks.task_id": tid},
            {"$push": {"tasks.$.progress": progress_entry}, "$set": {"tasks.$.updated_at": datetime.now(timezone.utc)}},
        )
        if result.modified_count > 0: logger.info(f"Added progress entry to Task {tid}.")
        return

async def schedule_immediate_reminder(user_id: str, reminder_text: str):
    """Schedules a push notification reminder one minute from now."""
    # This utility function also requires pytz
    import pytz 
    
    user_info = await users_collection.find_one({"user_id": user_id})
    tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
    try:
        user_tz = pytz.timezone(tz_name)
    except Exception:
        user_tz = pytz.UTC
        
    now_utc = datetime.now(pytz.UTC)
    now_local = now_utc.astimezone(user_tz)
    scheduled_local = now_local + timedelta(minutes=1)
    scheduled_time_utc = scheduled_local.astimezone(pytz.UTC)
    
    # We rely on schedule_notification from subscription_service
    await schedule_notification(
        user_id, f"Reminder: {reminder_text}", scheduled_time_utc, notif_type="reminder"
    )
    logger.info(f"Immediate reminder scheduled for user {user_id}: {reminder_text}")