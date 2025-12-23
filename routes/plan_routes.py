# stelle_backend/routes/plan_routes.py

from datetime import datetime
from typing import Optional, Dict, Any
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import traceback

from database import users_collection, goals_collection, weekly_plans_collection
from services.plan_service import planner, SUPPORTED_PLATFORMS
from config import logger

# Import sync collections if they exist for tasks
try:
    from database import get_or_init_sync_collections
    tasks_collection, blogs_collection = get_or_init_sync_collections()
except:
    tasks_collection = None
    blogs_collection = None
    logger.warning("Sync tasks collection not available")

router = APIRouter()

# ==================== REQUEST MODEL ====================

class PlanActionRequest(BaseModel):
    action: str
    data: Dict[str, Any] = Field(default_factory=dict)


# ==================== HELPER FUNCTIONS ====================

def parse_date(date_str: str) -> datetime:
    """Parse date string with multiple format support"""
    for date_format in ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d']:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    raise ValueError(f'Invalid date format: {date_str}. Use MM/DD/YYYY, DD/MM/YYYY, or YYYY-MM-DD')


def convert_plan_dates(plan: Dict) -> Dict:
    """Convert datetime objects to strings for JSON serialization"""
    if 'start_date' in plan and isinstance(plan['start_date'], datetime):
        plan['start_date'] = plan['start_date'].strftime('%m/%d/%Y')
    if 'end_date' in plan and isinstance(plan['end_date'], datetime):
        plan['end_date'] = plan['end_date'].strftime('%m/%d/%Y')
    if 'created_at' in plan and isinstance(plan['created_at'], datetime):
        plan['created_at'] = plan['created_at'].isoformat()
    if 'updated_at' in plan and isinstance(plan['updated_at'], datetime):
        plan['updated_at'] = plan['updated_at'].isoformat()
    return plan


async def save_plan_tasks(plan_id: str, user_id: str, goal_id: Optional[str], plan_data: Dict) -> int:
    """Save tasks to tasks collection"""
    if tasks_collection is None:
        return 0
    
    tasks_saved = 0
    for week in plan_data.get('weekly_plans', []):
        for task in week.get('tasks', []):
            task_doc = {
                'plan_id': plan_id,
                'user_id': user_id,
                'goal_id': goal_id,
                'week_number': week['week_number'],
                'task_id': task['task_id'],
                'title': task['title'],
                'description': task['description'],
                'priority': task['priority'],
                'estimated_hours': task['estimated_hours'],
                'dependencies': task.get('dependencies', []),
                'day_of_week': task.get('day_of_week', 'Monday'),
                'status': task.get('status', 'pending'),
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            tasks_collection.insert_one(task_doc)
            tasks_saved += 1
    
    return tasks_saved


# ==================== UNIFIED ENDPOINT ====================

@router.post("/plan-manager", tags=["Plan Manager"])
async def plan_manager(request: Request, action_request: Optional[PlanActionRequest] = None):
    """
    🎯 UNIFIED API ENDPOINT FOR ALL PLAN OPERATIONS
    
    Supports 14 different actions via a single endpoint.
    
    Request Format:
    {
        "action": "action_name",
        "data": { ...action-specific parameters... }
    }
    
    Supported Actions:
    1. health_check
    2. get_user_goals
    3. get_plan_my_week_data
    4. create_plan_from_goal
    5. create_plan_custom
    6. get_plan
    7. get_user_plans
    8. get_plan_tasks
    9. update_task
    10. reorder_task
    11. update_task_date
    12. regenerate_plan
    13. delete_plan
    """
    try:
        # Handle different HTTP methods
        if request.method == 'GET':
            action = request.query_params.get('action')
            data = dict(request.query_params)
        else:
            if action_request:
                action = action_request.action
                data = action_request.data
            else:
                raise HTTPException(status_code=400, detail='Missing request body')
        
        if not action:
            raise HTTPException(status_code=400, detail='Missing required field: action')
        
        logger.info(f"📍 Action: {action}")
        
        # ==================== ACTION 1: HEALTH CHECK ====================
        
        if action == 'health_check':
            """Health check endpoint"""
            try:
                await users_collection.find_one({}, {"_id": 1})
                db_status = "connected"
            except Exception as e:
                db_status = f"error: {str(e)}"
            
            return {
                'status': 'healthy',
                'database': db_status,
                'groq_configured': planner is not None,
                'supported_platforms': SUPPORTED_PLATFORMS,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # ==================== ACTION 2: GET USER GOALS ====================
        
        elif action == 'get_user_goals':
            """Get all goals for a specific user"""
            user_id = data.get('user_id')
            if not user_id:
                raise HTTPException(status_code=400, detail='Missing user_id')
            
            # Try to find user - check multiple possible ID fields
            user = None
            try:
                # Try _id as ObjectId
                user = await users_collection.find_one({'_id': ObjectId(user_id)})
            except:
                pass
            
            if not user:
                # Try userId as string
                user = await users_collection.find_one({'userId': user_id})
            
            if not user:
                # Try user_id as string
                user = await users_collection.find_one({'user_id': user_id})
            
            # If still not found, continue anyway (user data is optional for goals)
            user_name = user.get('name', 'Unknown') if user else 'Unknown'
            
            goals = await goals_collection.find({'user_id': user_id}).sort('created_at', -1).to_list(None)
            
            for goal in goals:
                goal['_id'] = str(goal['_id'])
                if 'created_at' in goal and isinstance(goal['created_at'], datetime):
                    goal['created_at'] = goal['created_at'].isoformat()
                if 'updated_at' in goal and isinstance(goal['updated_at'], datetime):
                    goal['updated_at'] = goal['updated_at'].isoformat()
            
            return {
                'success': True,
                'user_id': user_id,
                'user_name': user_name,
                'goals_count': len(goals),
                'goals': goals
            }
        
        # ==================== ACTION 3: GET PLAN MY WEEK DATA ====================
        
        elif action == 'get_plan_my_week_data':
            """Get all data needed for Plan My Week section"""
            user_id = data.get('user_id')
            if not user_id:
                raise HTTPException(status_code=400, detail='Missing user_id')
            
            # Try to find user - check multiple possible ID fields
            user = None
            try:
                # Try _id as ObjectId
                user = await users_collection.find_one({'_id': ObjectId(user_id)})
            except:
                pass
            
            if not user:
                # Try userId as string
                user = await users_collection.find_one({'userId': user_id})
            
            if not user:
                # Try user_id as string
                user = await users_collection.find_one({'user_id': user_id})
            
            # If still not found, continue anyway with default values
            user_name = user.get('name', 'Unknown') if user else 'Unknown'
            user_email = user.get('email', '') if user else ''
            user_timezone = user.get('timeZone', 'gmt') if user else 'gmt'
            
            # Fetch goals
            goals = await goals_collection.find({'user_id': user_id}).sort('created_at', -1).to_list(None)
            for goal in goals:
                goal['_id'] = str(goal['_id'])
                if 'created_at' in goal and isinstance(goal['created_at'], datetime):
                    goal['created_at'] = goal['created_at'].isoformat()
                if 'updated_at' in goal and isinstance(goal['updated_at'], datetime):
                    goal['updated_at'] = goal['updated_at'].isoformat()
            
            # Fetch existing plans
            plans = await weekly_plans_collection.find({'user_id': user_id}).sort('created_at', -1).to_list(None)
            for plan in plans:
                plan['_id'] = str(plan['_id'])
                plan = convert_plan_dates(plan)
            
            return {
                'success': True,
                'user': {
                    'user_id': user_id,
                    'name': user_name,
                    'email': user_email,
                    'timezone': user_timezone
                },
                'goals': {
                    'count': len(goals),
                    'list': goals
                },
                'existing_plans': {
                    'count': len(plans),
                    'list': plans
                }
            }
        
        # ==================== ACTION 4: CREATE PLAN FROM EXISTING GOAL ====================
        
        elif action == 'create_plan_from_goal':
            """Create plan from existing goal in goals collection"""
            user_id = data.get('user_id')
            goal_id = data.get('goal_id')
            start_date_str = data.get('start_date')
            end_date_str = data.get('end_date')
            
            if not all([user_id, goal_id, start_date_str, end_date_str]):
                raise HTTPException(status_code=400, detail='Missing required fields: user_id, goal_id, start_date, end_date')
            
            # Fetch goal
            goal = await goals_collection.find_one({'user_id': user_id, 'goal_id': goal_id})
            if not goal:
                raise HTTPException(status_code=404, detail='Goal not found for this user')
            
            goal_text = goal.get('title', '')
            if goal.get('description'):
                goal_text += f" - {goal['description']}"
            
            # Parse dates
            start_date = parse_date(start_date_str)
            end_date = parse_date(end_date_str)
            
            if start_date >= end_date:
                raise HTTPException(status_code=400, detail='End date must be after start date')
            
            # Generate plan
            plan_data = planner.generate_weekly_plan(goal_text, start_date, end_date)
            if not plan_data:
                raise HTTPException(status_code=500, detail='Failed to generate plan')
            
            # Save plan
            plan_doc = {
                'user_id': user_id,
                'goal_id': goal_id,
                'goal': goal_text,
                'start_date': start_date,
                'end_date': end_date,
                'plan_data': plan_data,
                'plan_type': 'from_existing_goal',
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'status': 'active'
            }
            
            result = await weekly_plans_collection.insert_one(plan_doc)
            plan_id = str(result.inserted_id)
            
            # Save tasks
            tasks_saved = await save_plan_tasks(plan_id, user_id, goal_id, plan_data)
            
            return {
                'success': True,
                'plan_id': plan_id,
                'goal_id': goal_id,
                'goal_title': goal.get('title'),
                'plan': plan_data,
                'tasks_count': tasks_saved
            }
        
        # ==================== ACTION 5: CREATE PLAN WITH CUSTOM GOAL ====================
        
        elif action == 'create_plan_custom':
            """Create plan with custom goal (not from goals collection)"""
            user_id = data.get('user_id')
            custom_goal = data.get('custom_goal')
            start_date_str = data.get('start_date')
            end_date_str = data.get('end_date')
            
            if not all([user_id, custom_goal, start_date_str, end_date_str]):
                raise HTTPException(status_code=400, detail='Missing required fields: user_id, custom_goal, start_date, end_date')
            
            # Parse dates
            start_date = parse_date(start_date_str)
            end_date = parse_date(end_date_str)
            
            if start_date >= end_date:
                raise HTTPException(status_code=400, detail='End date must be after start date')
            
            # Generate plan
            plan_data = planner.generate_weekly_plan(custom_goal, start_date, end_date)
            if not plan_data:
                raise HTTPException(status_code=500, detail='Failed to generate plan')
            
            # Save plan
            plan_doc = {
                'user_id': user_id,
                'goal_id': None,
                'goal': custom_goal,
                'start_date': start_date,
                'end_date': end_date,
                'plan_data': plan_data,
                'plan_type': 'custom_goal',
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'status': 'active'
            }
            
            result = await weekly_plans_collection.insert_one(plan_doc)
            plan_id = str(result.inserted_id)
            
            # Save tasks
            tasks_saved = await save_plan_tasks(plan_id, user_id, None, plan_data)
            
            return {
                'success': True,
                'plan_id': plan_id,
                'custom_goal': custom_goal,
                'plan': plan_data,
                'tasks_count': tasks_saved
            }
        
        # ==================== ACTION 6: GET PLAN ====================
        
        elif action == 'get_plan':
            """Get a specific plan by ID"""
            plan_id = data.get('plan_id')
            if not plan_id or not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            plan = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id)})
            if not plan:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            plan['_id'] = str(plan['_id'])
            plan = convert_plan_dates(plan)
            
            return {
                'success': True,
                'plan': plan
            }
        
        # ==================== ACTION 7: GET USER PLANS ====================
        
        elif action == 'get_user_plans':
            """Get all plans for a user"""
            user_id = data.get('user_id')
            if not user_id:
                raise HTTPException(status_code=400, detail='Missing user_id')
            
            plans = await weekly_plans_collection.find({'user_id': user_id}).sort('created_at', -1).to_list(None)
            
            for plan in plans:
                plan['_id'] = str(plan['_id'])
                plan = convert_plan_dates(plan)
            
            return {
                'success': True,
                'count': len(plans),
                'plans': plans
            }
        
        # ==================== ACTION 8: GET PLAN TASKS ====================(Not Working Also Not Needed as tasks is displyed by plan_id)
        
        elif action == 'get_plan_tasks':
            """Get all tasks for a plan"""
            plan_id = data.get('plan_id')
            if not plan_id or not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            if tasks_collection is None:
                raise HTTPException(status_code=503, detail='Tasks collection not available')
            
            tasks = list(tasks_collection.find({'plan_id': plan_id}).sort('week_number', 1))
            
            for task in tasks:
                task['_id'] = str(task['_id'])
                if 'created_at' in task:
                    task['created_at'] = task['created_at'].isoformat()
                if 'updated_at' in task:
                    task['updated_at'] = task['updated_at'].isoformat()
            
            return {
                'success': True,
                'count': len(tasks),
                'tasks': tasks
            }
        
        # ==================== ACTION 9: UPDATE TASK ====================
        
        elif action == 'update_task':
            """Update task properties"""
            plan_id = data.get('plan_id')
            task_id = data.get('task_id')
            updates = data.get('updates', {})
            
            if not plan_id or not task_id:
                raise HTTPException(status_code=400, detail='Missing plan_id or task_id')
            
            if not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            # Get the plan document
            plan = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id)})
            if not plan:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            # Find and update the task in the plan document
            task_found = False
            updates['updated_at'] = datetime.utcnow()
            
            for week in plan['plan_data'].get('weekly_plans', []):
                for task in week.get('tasks', []):
                    if task['task_id'] == task_id:
                        task.update(updates)
                        task_found = True
                        break
                if task_found:
                    break
            
            if not task_found:
                raise HTTPException(status_code=404, detail='Task not found in plan')
            
            # Update the plan document with modified tasks
            await weekly_plans_collection.update_one(
                {'_id': ObjectId(plan_id)},
                {'$set': {
                    'plan_data': plan['plan_data'],
                    'updated_at': datetime.utcnow()
                }}
            )
            
            # Also update in separate tasks collection if it exists
            if tasks_collection is not None:
                try:
                    tasks_collection.update_one(
                        {'plan_id': plan_id, 'task_id': task_id},
                        {'$set': updates}
                    )
                except Exception as e:
                    logger.warning(f"Failed to update task in tasks_collection: {e}")
            
            return {
                'success': True,
                'message': 'Task updated successfully',
                'task_id': task_id,
                'updates': updates
            }
        
        # ==================== ACTION 10: REORDER TASK ====================
        
        elif action == 'reorder_task':
            """Reorder tasks in a week"""
            plan_id = data.get('plan_id')
            task_id = data.get('task_id')
            week_number = data.get('week_number')
            new_position = data.get('new_position')
            
            if not all([plan_id, task_id, week_number is not None, new_position is not None]):
                raise HTTPException(status_code=400, detail='Missing required fields')
            
            if not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            plan = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id)})
            if not plan:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            # Find the week
            week_plan = None
            for week in plan['plan_data']['weekly_plans']:
                if week['week_number'] == week_number:
                    week_plan = week
                    break
            
            if not week_plan:
                raise HTTPException(status_code=404, detail=f'Week {week_number} not found')
            
            # Reorder tasks
            success = planner.reorder_tasks_in_week(week_plan, task_id, new_position)
            if not success:
                raise HTTPException(status_code=404, detail='Task not found')
            
            # Update database
            await weekly_plans_collection.update_one(
                {'_id': ObjectId(plan_id)},
                {'$set': {
                    'plan_data': plan['plan_data'],
                    'updated_at': datetime.utcnow()
                }}
            )
            
            return {
                'success': True,
                'message': 'Task reordered successfully',
                'updated_plan': plan['plan_data']
            }
        
        # ==================== ACTION 11: UPDATE TASK DATE ====================
        
        elif action == 'update_task_date':
            """Update task date and regenerate plan"""
            plan_id = data.get('plan_id')
            task_id = data.get('task_id')
            new_task_date_str = data.get('new_task_date')
            new_start_str = data.get('new_start_date')
            new_end_str = data.get('new_end_date')
            
            if not plan_id or not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            if not task_id and not new_start_str and not new_end_str:
                raise HTTPException(status_code=400, detail='Nothing to update')
            
            plan = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id)})
            if not plan:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            original_goal = plan['goal']
            plan_data = plan['plan_data']
            user_id = plan['user_id']
            current_start = plan['start_date']
            current_end = plan['end_date']
            
            # Parse dates
            new_task_date = parse_date(new_task_date_str) if new_task_date_str else None
            new_start_date = parse_date(new_start_str) if new_start_str else None
            new_end_date = parse_date(new_end_str) if new_end_str else None
            
            # Update task date
            if task_id and new_task_date:
                found = False
                for week in plan_data['weekly_plans']:
                    for task in week['tasks']:
                        if task['task_id'] == task_id:
                            task['day_of_week'] = new_task_date.strftime('%A')
                            found = True
                            break
                if not found:
                    raise HTTPException(status_code=404, detail='Task not found')
            
            # Update dates
            if new_start_date:
                current_start = new_start_date
            if new_end_date:
                current_end = new_end_date
            
            # Regenerate plan
            new_plan_data = planner.generate_weekly_plan(
                original_goal,
                current_start,
                current_end,
                plan_data
            )
            
            if not new_plan_data:
                raise HTTPException(status_code=500, detail='Plan regeneration failed')
            
            # Update database
            await weekly_plans_collection.update_one(
                {'_id': ObjectId(plan_id)},
                {'$set': {
                    'start_date': current_start,
                    'end_date': current_end,
                    'plan_data': new_plan_data,
                    'updated_at': datetime.utcnow()
                }}
            )
            
            # Rewrite tasks
            if tasks_collection is not None:
                tasks_collection.delete_many({'plan_id': plan_id})
                await save_plan_tasks(plan_id, user_id, plan.get('goal_id'), new_plan_data)
            
            return {
                'success': True,
                'message': 'Plan updated successfully',
                'updated_start_date': current_start.strftime('%m/%d/%Y'),
                'updated_end_date': current_end.strftime('%m/%d/%Y'),
                'updated_plan': new_plan_data
            }
        
        # ==================== ACTION 12: REGENERATE PLAN ====================
        
        elif action == 'regenerate_plan':
            """Regenerate entire plan with AI"""
            plan_id = data.get('plan_id')
            new_goal = data.get('new_goal')
            
            if not plan_id or not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            plan = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id)})
            if not plan:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            goal = new_goal if new_goal else plan['goal']
            
            updated_plan = planner.regenerate_plan(
                goal,
                plan['start_date'],
                plan['end_date'],
                plan['plan_data']
            )
            
            if not updated_plan:
                raise HTTPException(status_code=500, detail='Failed to regenerate plan')
            
            # Update database
            await weekly_plans_collection.update_one(
                {'_id': ObjectId(plan_id)},
                {'$set': {
                    'goal': goal,
                    'plan_data': updated_plan,
                    'updated_at': datetime.utcnow()
                }}
            )
            
            # Update tasks
            if tasks_collection is not None:
                tasks_collection.delete_many({'plan_id': plan_id})
                await save_plan_tasks(plan_id, plan['user_id'], plan.get('goal_id'), updated_plan)
            
            return {
                'success': True,
                'message': 'Plan regenerated successfully',
                'updated_plan': updated_plan
            }
        
        # ==================== ACTION 13: DELETE PLAN ====================
        
        elif action == 'delete_plan':
            """Delete a plan and all its tasks"""
            plan_id = data.get('plan_id')
            if not plan_id or not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            result = await weekly_plans_collection.delete_one({'_id': ObjectId(plan_id)})
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            # Delete associated tasks
            tasks_deleted = 0
            if tasks_collection is not None:
                tasks_result = tasks_collection.delete_many({'plan_id': plan_id})
                tasks_deleted = tasks_result.deleted_count
            
            return {
                'success': True,
                'message': f'Deleted plan and {tasks_deleted} tasks'
            }
        
        else:
            raise HTTPException(status_code=400, detail=f'Unknown action: {action}')
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in plan_manager: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
