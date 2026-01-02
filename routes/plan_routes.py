# stelle_backend/routes/plan_routes.py

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import traceback

from database import users_collection, goals_collection, weekly_plans_collection, calendar_events_collection
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
                'date': task.get('date', ''),
                'subtasks': task.get('subtasks', []),
                'status': task.get('status', 'pending'),
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            tasks_collection.insert_one(task_doc)
            tasks_saved += 1
    
    return tasks_saved


def detect_plan_overlaps(plan_start: datetime, plan_end: datetime, existing_plans: list) -> list:
    """
    Detect overlapping plans for a given date range
    
    Args:
        plan_start: Start date of the new/current plan
        plan_end: End date of the new/current plan
        existing_plans: List of existing plans to check against
        
    Returns:
        List of overlapping plans with overlap details
    """
    overlaps = []
    
    for plan in existing_plans:
        existing_start = plan['start_date'] if isinstance(plan['start_date'], datetime) else parse_date(plan['start_date'])
        existing_end = plan['end_date'] if isinstance(plan['end_date'], datetime) else parse_date(plan['end_date'])
        
        # Check if there's an overlap
        if plan_start <= existing_end and plan_end >= existing_start:
            # Calculate overlap period
            overlap_start = max(plan_start, existing_start)
            overlap_end = min(plan_end, existing_end)
            overlap_days = (overlap_end - overlap_start).days + 1
            
            overlaps.append({
                'plan_id': str(plan['_id']),
                'plan_goal': plan.get('goal', 'Untitled Plan'),
                'plan_start': existing_start.strftime('%m/%d/%Y'),
                'plan_end': existing_end.strftime('%m/%d/%Y'),
                'overlap_start': overlap_start.strftime('%m/%d/%Y'),
                'overlap_end': overlap_end.strftime('%m/%d/%Y'),
                'overlap_days': overlap_days,
                'overlap_weeks': (overlap_days + 6) // 7
            })
    
    return overlaps


def get_overlapping_tasks(plan1_data: Dict, plan2_data: Dict, overlap_start: datetime, overlap_end: datetime, plan1_start: datetime = None, plan2_start: datetime = None) -> Dict:
    """
    Get tasks from both plans that fall within the overlapping period
    
    Args:
        plan1_data: First plan's data
        plan2_data: Second plan's data
        overlap_start: Start of overlap period
        overlap_end: End of overlap period
        plan1_start: Start date of plan 1 (optional)
        plan2_start: Start date of plan 2 (optional)
        
    Returns:
        Dictionary containing tasks from both plans during overlap
    """
    def get_week_date_range(plan_start: datetime, week_number: int) -> tuple:
        """Calculate the date range for a specific week"""
        week_start = plan_start + timedelta(days=(week_number - 1) * 7)
        week_end = week_start + timedelta(days=6)
        return week_start, week_end
    
    # Get plan start dates - try from parameters first, then from plan_data
    if plan1_start is None:
        plan1_start = plan1_data.get('start_date')
        if plan1_start is None:
            raise ValueError("plan1_start is required but not provided")
        if isinstance(plan1_start, str):
            plan1_start = parse_date(plan1_start)
    
    if plan2_start is None:
        plan2_start = plan2_data.get('start_date')
        if plan2_start is None:
            raise ValueError("plan2_start is required but not provided")
        if isinstance(plan2_start, str):
            plan2_start = parse_date(plan2_start)
    
    overlapping_tasks = {
        'plan1_tasks': [],
        'plan2_tasks': [],
        'total_plan1_tasks': 0,
        'total_plan2_tasks': 0,
        'overlap_period': {
            'start': overlap_start.strftime('%m/%d/%Y'),
            'end': overlap_end.strftime('%m/%d/%Y'),
            'days': (overlap_end - overlap_start).days + 1
        }
    }
    
    # Get tasks from plan 1
    for week in plan1_data.get('weekly_plans', []):
        week_start, week_end = get_week_date_range(plan1_start, week['week_number'])
        
        # Check if this week overlaps with the overlap period
        if week_start <= overlap_end and week_end >= overlap_start:
            for task in week['tasks']:
                overlapping_tasks['plan1_tasks'].append({
                    'week_number': week['week_number'],
                    'week_start': week_start.strftime('%m/%d/%Y'),
                    'week_end': week_end.strftime('%m/%d/%Y'),
                    'task_id': task['task_id'],
                    'title': task['title'],
                    'description': task.get('description', ''),
                    'priority': task.get('priority', 'medium'),
                    'estimated_hours': task.get('estimated_hours', 0),
                    'day_of_week': task.get('day_of_week', 'Monday'),
                    'status': task.get('status', 'pending')
                })
                overlapping_tasks['total_plan1_tasks'] += 1
    
    # Get tasks from plan 2
    for week in plan2_data.get('weekly_plans', []):
        week_start, week_end = get_week_date_range(plan2_start, week['week_number'])
        
        # Check if this week overlaps with the overlap period
        if week_start <= overlap_end and week_end >= overlap_start:
            for task in week['tasks']:
                overlapping_tasks['plan2_tasks'].append({
                    'week_number': week['week_number'],
                    'week_start': week_start.strftime('%m/%d/%Y'),
                    'week_end': week_end.strftime('%m/%d/%Y'),
                    'task_id': task['task_id'],
                    'title': task['title'],
                    'description': task.get('description', ''),
                    'priority': task.get('priority', 'medium'),
                    'estimated_hours': task.get('estimated_hours', 0),
                    'day_of_week': task.get('day_of_week', 'Monday'),
                    'status': task.get('status', 'pending')
                })
                overlapping_tasks['total_plan2_tasks'] += 1
    
    return overlapping_tasks


def calculate_workload_conflict(overlapping_tasks: Dict) -> Dict:
    """
    Calculate if there's a workload conflict during overlap period
    
    Args:
        overlapping_tasks: Dictionary from get_overlapping_tasks
        
    Returns:
        Workload analysis with conflict detection
    """
    total_hours_plan1 = sum(task['estimated_hours'] for task in overlapping_tasks['plan1_tasks'])
    total_hours_plan2 = sum(task['estimated_hours'] for task in overlapping_tasks['plan2_tasks'])
    total_combined_hours = total_hours_plan1 + total_hours_plan2
    
    overlap_days = overlapping_tasks['overlap_period']['days']
    overlap_weeks = (overlap_days + 6) // 7
    
    avg_hours_per_week = total_combined_hours / overlap_weeks if overlap_weeks > 0 else 0
    avg_hours_per_day = total_combined_hours / overlap_days if overlap_days > 0 else 0
    
    # Determine conflict level (assuming 40 hours/week is max capacity)
    if avg_hours_per_week > 50:
        conflict_level = 'critical'
        message = 'Critical workload conflict! You have too many tasks scheduled during the overlap period.'
    elif avg_hours_per_week > 40:
        conflict_level = 'high'
        message = 'High workload conflict. Consider rescheduling some tasks to avoid burnout.'
    elif avg_hours_per_week > 30:
        conflict_level = 'medium'
        message = 'Moderate workload. This is manageable but will require good time management.'
    else:
        conflict_level = 'low'
        message = 'Low conflict. Your workload during the overlap period is reasonable.'
    
    return {
        'conflict_level': conflict_level,
        'message': message,
        'total_hours_plan1': total_hours_plan1,
        'total_hours_plan2': total_hours_plan2,
        'total_combined_hours': total_combined_hours,
        'avg_hours_per_week': round(avg_hours_per_week, 1),
        'avg_hours_per_day': round(avg_hours_per_day, 1),
        'overlap_weeks': overlap_weeks,
        'overlap_days': overlap_days,
        'recommended_max_hours_per_week': 40
    }


def generate_sync_recommendations(workload_analysis: Dict, overlapping_tasks: Dict) -> List[str]:
    """
    Generate actionable recommendations for managing overlapping plans
    
    Args:
        workload_analysis: Workload analysis dictionary
        overlapping_tasks: Overlapping tasks dictionary
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    conflict_level = workload_analysis['conflict_level']
    avg_hours_per_week = workload_analysis['avg_hours_per_week']
    
    if conflict_level == 'critical':
        recommendations.append(
            f"⚠️ URGENT: You have {avg_hours_per_week} hours/week during the overlap. "
            "You must reschedule some tasks to avoid overwhelming yourself."
        )
        recommendations.append(
            "Move non-urgent tasks from one plan to weeks before or after the overlap period."
        )
        recommendations.append(
            "Consider extending one of your plan deadlines to reduce the workload."
        )
    elif conflict_level == 'high':
        recommendations.append(
            f"⚠️ High workload: {avg_hours_per_week} hours/week during overlap. "
            "This is manageable but challenging."
        )
        recommendations.append(
            "Prioritize high-priority tasks and consider postponing lower-priority ones."
        )
        recommendations.append(
            "Block specific time slots for each plan to maintain focus."
        )
    elif conflict_level == 'medium':
        recommendations.append(
            f"Your workload is {avg_hours_per_week} hours/week during the overlap period. "
            "This is reasonable with good time management."
        )
        recommendations.append(
            "Create a daily schedule to balance tasks from both plans effectively."
        )
    else:
        recommendations.append(
            f"✅ Your workload ({avg_hours_per_week} hours/week) during the overlap is manageable."
        )
        recommendations.append(
            "Continue with your current schedule. Just stay organized!"
        )
    
    # Task-specific recommendations
    total_plan1_tasks = overlapping_tasks['total_plan1_tasks']
    total_plan2_tasks = overlapping_tasks['total_plan2_tasks']
    
    if total_plan1_tasks + total_plan2_tasks > 0:
        recommendations.append(
            f"You have {total_plan1_tasks} tasks from Plan 1 and {total_plan2_tasks} tasks from Plan 2 "
            "during the overlap period."
        )
        
        if total_plan1_tasks + total_plan2_tasks > 15:
            recommendations.append(
                "Consider combining similar tasks or breaking larger tasks into smaller, more manageable chunks."
            )
    
    # General tips
    recommendations.append(
        "💡 Tip: Use the 'reorder_task' action to move tasks between weeks if needed."
    )
    recommendations.append(
        "💡 Tip: Review your progress weekly and adjust your schedule as needed."
    )
    
    return recommendations


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
    14. check_plan_overlaps
    15. get_overlap_details
    16. sync_overlapping_plans
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
            
            # Fetch existing plans - try multiple user_id formats
            plans = []
            
            # Try finding plans with user_id as string
            plans_by_string = await weekly_plans_collection.find({'user_id': user_id}).sort('created_at', -1).to_list(None)
            plans.extend(plans_by_string)
            
            # Also try finding by ObjectId if user_id is valid ObjectId format
            if ObjectId.is_valid(user_id):
                try:
                    plans_by_objectid = await weekly_plans_collection.find({'user_id': ObjectId(user_id)}).sort('created_at', -1).to_list(None)
                    # Deduplicate - only add if not already in list
                    existing_ids = {str(p['_id']) for p in plans}
                    for p in plans_by_objectid:
                        if str(p['_id']) not in existing_ids:
                            plans.append(p)
                except Exception as e:
                    logger.warning(f"Error searching by ObjectId: {e}")
            
            # Convert plan data
            converted_plans = []
            for plan in plans:
                try:
                    plan['_id'] = str(plan['_id'])
                    if 'goal_id' in plan and plan['goal_id'] and isinstance(plan['goal_id'], ObjectId):
                        plan['goal_id'] = str(plan['goal_id'])
                    converted_plan = convert_plan_dates(plan)
                    converted_plans.append(converted_plan)
                except Exception as e:
                    logger.error(f"Error converting plan: {e}")
            
            plans = converted_plans
            
            # Format plans for better readability
            formatted_plans = []
            for idx, plan in enumerate(plans, 1):
                formatted_plan = {
                    'plan_number': idx,
                    'plan_id': plan['_id'],
                    'goal': plan.get('goal', 'Untitled Plan'),
                    'plan_type': plan.get('plan_type', 'unknown'),
                    'start_date': plan.get('start_date'),
                    'end_date': plan.get('end_date'),
                    'status': plan.get('status', 'active'),
                    'created_at': plan.get('created_at'),
                    'total_weeks': len(plan.get('plan_data', {}).get('weekly_plans', [])),
                    'goal_id': plan.get('goal_id')
                }
                formatted_plans.append(formatted_plan)
            
            return {
                'success': True,
                'summary': {
                    'total_goals': len(goals),
                    'total_plans': len(plans),
                    'user_name': user_name
                },
                'user_details': {
                    'user_id': user_id,
                    'name': user_name,
                    'email': user_email,
                    'timezone': user_timezone
                },
                'goals': goals,
                'plans': formatted_plans
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
            
            # Check for overlapping plans
            existing_plans = await weekly_plans_collection.find({
                'user_id': user_id,
                '_id': {'$ne': result.inserted_id}
            }).to_list(length=100)
            
            overlaps = detect_plan_overlaps(start_date, end_date, existing_plans)
            
            response = {
                'success': True,
                'plan_id': plan_id,
                'goal_id': goal_id,
                'goal_title': goal.get('title'),
                'plan': plan_data,
                'tasks_count': tasks_saved
            }
            
            # Add overlap warning if found
            if overlaps:
                response['overlap_warning'] = {
                    'has_overlaps': True,
                    'total_overlaps': len(overlaps),
                    'message': f'This plan overlaps with {len(overlaps)} existing plan(s). Use action "get_overlap_details" to see details.',
                    'overlaps': overlaps
                }
            
            return response
        
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
            
            # Check for overlapping plans
            existing_plans = await weekly_plans_collection.find({
                'user_id': user_id,
                '_id': {'$ne': result.inserted_id}
            }).to_list(length=100)
            
            overlaps = detect_plan_overlaps(start_date, end_date, existing_plans)
            
            response = {
                'success': True,
                'plan_id': plan_id,
                'custom_goal': custom_goal,
                'plan': plan_data,
                'tasks_count': tasks_saved
            }
            
            # Add overlap warning if found
            if overlaps:
                response['overlap_warning'] = {
                    'has_overlaps': True,
                    'total_overlaps': len(overlaps),
                    'message': f'This plan overlaps with {len(overlaps)} existing plan(s). Use action \"get_overlap_details\" to see details.',
                    'overlaps': overlaps
                }
            
            return response
        
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
        
        # ==================== ACTION 10: REORDER/MOVE TASK (Multiple Weeks Problem Solved) ====================
        
        elif action == 'reorder_task':
            """
            Reorder tasks within a week OR move tasks between different weeks
            
            Parameters:
            - plan_id: ID of the plan
            - task_id: ID of the task to move
            - source_week_number: Current week of the task (optional, will be detected if not provided)
            - target_week_number: Target week number (defaults to source_week_number for reordering)
            - new_position: Position in target week (0-indexed)
            - new_date: Optional new date for the task (format: MM/DD/YYYY)
            """
            plan_id = data.get('plan_id')
            task_id = data.get('task_id')
            source_week_number = data.get('source_week_number')
            target_week_number = data.get('target_week_number') or data.get('week_number')  # Support old format
            new_position = data.get('new_position', 0)
            new_date = data.get('new_date')
            
            # Validate required fields
            if not plan_id or not task_id:
                raise HTTPException(status_code=400, detail='Missing required fields: plan_id and task_id')
            
            if not ObjectId.is_valid(plan_id):
                raise HTTPException(status_code=400, detail='Invalid plan_id')
            
            # Get the plan
            plan = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id)})
            if not plan:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            # If source week not provided, find it
            if source_week_number is None:
                for week in plan['plan_data']['weekly_plans']:
                    for task in week['tasks']:
                        if task['task_id'] == task_id:
                            source_week_number = week['week_number']
                            break
                    if source_week_number is not None:
                        break
                
                if source_week_number is None:
                    raise HTTPException(status_code=404, detail='Task not found in any week')
            
            # If target week not provided, assume same week (reorder)
            if target_week_number is None:
                target_week_number = source_week_number
            
            # Check if it's same week (reorder) or different week (move)
            if source_week_number == target_week_number:
                # Reorder within same week (old behavior)
                week_plan = None
                for week in plan['plan_data']['weekly_plans']:
                    if week['week_number'] == target_week_number:
                        week_plan = week
                        break
                
                if not week_plan:
                    raise HTTPException(status_code=404, detail=f'Week {target_week_number} not found')
                
                result = planner.reorder_tasks_in_week(week_plan, task_id, new_position, new_date)
                if not result['success']:
                    raise HTTPException(status_code=404, detail='Task not found')
                
                message = 'Task reordered successfully'
                response_data = {
                    'success': True,
                    'message': message,
                    'updated_plan': plan['plan_data']
                }
                
                # Add warnings if present
                if result.get('warnings'):
                    response_data['warnings'] = result['warnings']
            else:
                # Move between different weeks (new behavior)
                result = planner.move_task_between_weeks(
                    plan['plan_data'],
                    task_id,
                    source_week_number,
                    target_week_number,
                    new_position,
                    max_tasks_per_week=7  # You can make this configurable
                )
                
                if not result['success']:
                    raise HTTPException(status_code=400, detail=result['message'])
                
                message = result['message']
                response_data = {
                    'success': True,
                    'message': message,
                    'details': {
                        'moved_task': result['moved_task']['title'],
                        'from_week': result['source_week'],
                        'to_week': result['target_week'],
                        'new_position': result['new_position'],
                        'new_date': result.get('new_date'),
                        'new_day_of_week': result.get('new_day_of_week'),
                        'source_week_remaining_tasks': result['source_week_task_count'],
                        'target_week_total_tasks': result['target_week_task_count']
                    },
                    'updated_plan': plan['plan_data']
                }
                
                # Add warnings if present
                if 'warnings' in result:
                    response_data['warnings'] = result['warnings']
            
            # Update database
            await weekly_plans_collection.update_one(
                {'_id': ObjectId(plan_id)},
                {'$set': {
                    'plan_data': plan['plan_data'],
                    'updated_at': datetime.utcnow()
                }}
            )
            
            # Update tasks collection if it exists
            if tasks_collection is not None:
                try:
                    # Update the task document with new week number
                    tasks_collection.update_one(
                        {'plan_id': plan_id, 'task_id': task_id},
                        {'$set': {
                            'week_number': target_week_number,
                            'updated_at': datetime.utcnow()
                        }}
                    )
                except Exception as e:
                    logger.warning(f"Could not update task in tasks_collection: {str(e)}")
            
            return response_data
        
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
                            task['date'] = new_task_date.strftime('%m/%d/%Y')
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
            
            # Delete associated calendar events first
            calendar_result = await calendar_events_collection.delete_many({'plan_id': plan_id})
            calendar_deleted = calendar_result.deleted_count
            
            # Delete associated tasks from sync collection
            tasks_deleted = 0
            if tasks_collection is not None:
                tasks_result = tasks_collection.delete_many({'plan_id': plan_id})
                tasks_deleted = tasks_result.deleted_count
            
            # Delete the plan itself
            result = await weekly_plans_collection.delete_one({'_id': ObjectId(plan_id)})
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail='Plan not found')
            
            return {
                'success': True,
                'message': f'Deleted plan, {calendar_deleted} calendar events, and {tasks_deleted} tasks'
            }
        
        # ==================== ACTION 14: CHECK PLAN OVERLAPS ====================
        
        elif action == 'check_plan_overlaps':
            """
            Check if a new plan or existing plan overlaps with other user plans
            
            Parameters:
            - user_id: User's ID (required)
            - start_date: Plan start date (required, format: MM/DD/YYYY)
            - end_date: Plan end date (required, format: MM/DD/YYYY)
            - exclude_plan_id: Plan ID to exclude from check (optional, for updating existing plan)
            """
            user_id = data.get('user_id')
            start_date_str = data.get('start_date')
            end_date_str = data.get('end_date')
            exclude_plan_id = data.get('exclude_plan_id')
            
            if not user_id or not start_date_str or not end_date_str:
                raise HTTPException(status_code=400, detail='Missing required fields: user_id, start_date, end_date')
            
            try:
                plan_start = parse_date(start_date_str)
                plan_end = parse_date(end_date_str)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            if plan_end < plan_start:
                raise HTTPException(status_code=400, detail='End date must be after start date')
            
            # Get all user's plans
            query = {'user_id': user_id}
            if exclude_plan_id and ObjectId.is_valid(exclude_plan_id):
                query['_id'] = {'$ne': ObjectId(exclude_plan_id)}
            
            existing_plans = await weekly_plans_collection.find(query).to_list(length=100)
            
            if not existing_plans:
                return {
                    'success': True,
                    'has_overlaps': False,
                    'message': 'No overlapping plans found',
                    'overlaps': []
                }
            
            # Detect overlaps
            overlaps = detect_plan_overlaps(plan_start, plan_end, existing_plans)
            
            return {
                'success': True,
                'has_overlaps': len(overlaps) > 0,
                'total_overlaps': len(overlaps),
                'message': f'Found {len(overlaps)} overlapping plan(s)' if overlaps else 'No overlapping plans found',
                'overlaps': overlaps,
                'query_period': {
                    'start': start_date_str,
                    'end': end_date_str,
                    'total_days': (plan_end - plan_start).days + 1
                }
            }
        
        # ==================== ACTION 15: GET OVERLAP DETAILS ====================
        
        elif action == 'get_overlap_details':
            """
            Get detailed information about overlapping plans including tasks and workload
            
            Parameters:
            - plan_id_1: First plan ID (required)
            - plan_id_2: Second plan ID (required)
            """
            plan_id_1 = data.get('plan_id_1')
            plan_id_2 = data.get('plan_id_2')
            
            if not plan_id_1 or not plan_id_2:
                raise HTTPException(status_code=400, detail='Missing required fields: plan_id_1, plan_id_2')
            
            if not ObjectId.is_valid(plan_id_1) or not ObjectId.is_valid(plan_id_2):
                raise HTTPException(status_code=400, detail='Invalid plan ID(s)')
            
            # Get both plans
            plan1 = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id_1)})
            plan2 = await weekly_plans_collection.find_one({'_id': ObjectId(plan_id_2)})
            
            if not plan1 or not plan2:
                raise HTTPException(status_code=404, detail='One or both plans not found')
            
            # Check if they belong to the same user
            if plan1['user_id'] != plan2['user_id']:
                raise HTTPException(status_code=403, detail='Plans belong to different users')
            
            # Parse dates
            plan1_start = plan1['start_date'] if isinstance(plan1['start_date'], datetime) else parse_date(plan1['start_date'])
            plan1_end = plan1['end_date'] if isinstance(plan1['end_date'], datetime) else parse_date(plan1['end_date'])
            plan2_start = plan2['start_date'] if isinstance(plan2['start_date'], datetime) else parse_date(plan2['start_date'])
            plan2_end = plan2['end_date'] if isinstance(plan2['end_date'], datetime) else parse_date(plan2['end_date'])
            
            # Check if they actually overlap
            if plan1_start > plan2_end or plan2_start > plan1_end:
                return {
                    'success': True,
                    'has_overlap': False,
                    'message': 'These plans do not overlap',
                    'plan1_period': {
                        'start': plan1_start.strftime('%m/%d/%Y'),
                        'end': plan1_end.strftime('%m/%d/%Y')
                    },
                    'plan2_period': {
                        'start': plan2_start.strftime('%m/%d/%Y'),
                        'end': plan2_end.strftime('%m/%d/%Y')
                    }
                }
            
            # Calculate overlap period
            overlap_start = max(plan1_start, plan2_start)
            overlap_end = min(plan1_end, plan2_end)
            
            # Get overlapping tasks
            overlapping_tasks = get_overlapping_tasks(
                plan1['plan_data'], 
                plan2['plan_data'], 
                overlap_start, 
                overlap_end,
                plan1_start,
                plan2_start
            )
            
            # Calculate workload conflict
            workload_analysis = calculate_workload_conflict(overlapping_tasks)
            
            return {
                'success': True,
                'has_overlap': True,
                'plan1': {
                    'plan_id': str(plan1['_id']),
                    'goal': plan1.get('goal', 'Untitled Plan'),
                    'start_date': plan1_start.strftime('%m/%d/%Y'),
                    'end_date': plan1_end.strftime('%m/%d/%Y'),
                    'total_weeks': len(plan1['plan_data'].get('weekly_plans', []))
                },
                'plan2': {
                    'plan_id': str(plan2['_id']),
                    'goal': plan2.get('goal', 'Untitled Plan'),
                    'start_date': plan2_start.strftime('%m/%d/%Y'),
                    'end_date': plan2_end.strftime('%m/%d/%Y'),
                    'total_weeks': len(plan2['plan_data'].get('weekly_plans', []))
                },
                'overlap_analysis': {
                    'overlap_start': overlap_start.strftime('%m/%d/%Y'),
                    'overlap_end': overlap_end.strftime('%m/%d/%Y'),
                    'overlap_days': (overlap_end - overlap_start).days + 1,
                    'overlap_weeks': ((overlap_end - overlap_start).days + 7) // 7
                },
                'tasks_during_overlap': overlapping_tasks,
                'workload_analysis': workload_analysis,
                'recommendations': generate_sync_recommendations(workload_analysis, overlapping_tasks)
            }
        
        # ==================== ACTION 16: SYNC OVERLAPPING PLANS ====================
        
        elif action == 'sync_overlapping_plans':
            """
            Get a synchronized view of all user's plans with overlap warnings
            
            Parameters:
            - user_id: User's ID (required)
            - include_tasks: Whether to include task details (optional, default: false)
            """
            user_id = data.get('user_id')
            include_tasks = data.get('include_tasks', False)
            
            if not user_id:
                raise HTTPException(status_code=400, detail='Missing required field: user_id')
            
            # Get all user's plans sorted by start date
            all_plans = await weekly_plans_collection.find({'user_id': user_id}).sort('start_date', 1).to_list(length=100)
            
            if not all_plans:
                return {
                    'success': True,
                    'message': 'No plans found for this user',
                    'total_plans': 0,
                    'plans': [],
                    'overlaps': []
                }
            
            # Build plans summary
            plans_summary = []
            for plan in all_plans:
                plan_start = plan['start_date'] if isinstance(plan['start_date'], datetime) else parse_date(plan['start_date'])
                plan_end = plan['end_date'] if isinstance(plan['end_date'], datetime) else parse_date(plan['end_date'])
                
                plan_summary = {
                    'plan_id': str(plan['_id']),
                    'goal': plan.get('goal', 'Untitled Plan'),
                    'start_date': plan_start.strftime('%m/%d/%Y'),
                    'end_date': plan_end.strftime('%m/%d/%Y'),
                    'total_weeks': len(plan['plan_data'].get('weekly_plans', [])),
                    'total_days': (plan_end - plan_start).days + 1,
                    'created_at': plan['created_at'].isoformat() if 'created_at' in plan else None
                }
                
                if include_tasks:
                    total_tasks = sum(len(week.get('tasks', [])) for week in plan['plan_data'].get('weekly_plans', []))
                    total_hours = sum(
                        task.get('estimated_hours', 0) 
                        for week in plan['plan_data'].get('weekly_plans', []) 
                        for task in week.get('tasks', [])
                    )
                    plan_summary['total_tasks'] = total_tasks
                    plan_summary['total_estimated_hours'] = total_hours
                
                plans_summary.append(plan_summary)
            
            # Detect all overlaps between plans
            all_overlaps = []
            for i, plan1 in enumerate(all_plans):
                plan1_start = plan1['start_date'] if isinstance(plan1['start_date'], datetime) else parse_date(plan1['start_date'])
                plan1_end = plan1['end_date'] if isinstance(plan1['end_date'], datetime) else parse_date(plan1['end_date'])
                
                for j, plan2 in enumerate(all_plans[i+1:], start=i+1):
                    plan2_start = plan2['start_date'] if isinstance(plan2['start_date'], datetime) else parse_date(plan2['start_date'])
                    plan2_end = plan2['end_date'] if isinstance(plan2['end_date'], datetime) else parse_date(plan2['end_date'])
                    
                    # Check for overlap
                    if plan1_start <= plan2_end and plan1_end >= plan2_start:
                        overlap_start = max(plan1_start, plan2_start)
                        overlap_end = min(plan1_end, plan2_end)
                        overlap_days = (overlap_end - overlap_start).days + 1
                        
                        overlap_info = {
                            'plan1_id': str(plan1['_id']),
                            'plan1_goal': plan1.get('goal', 'Untitled Plan'),
                            'plan2_id': str(plan2['_id']),
                            'plan2_goal': plan2.get('goal', 'Untitled Plan'),
                            'overlap_start': overlap_start.strftime('%m/%d/%Y'),
                            'overlap_end': overlap_end.strftime('%m/%d/%Y'),
                            'overlap_days': overlap_days,
                            'overlap_weeks': (overlap_days + 6) // 7
                        }
                        
                        if include_tasks:
                            # Get overlapping tasks and workload analysis
                            overlapping_tasks = get_overlapping_tasks(
                                plan1['plan_data'], 
                                plan2['plan_data'], 
                                overlap_start, 
                                overlap_end,
                                plan1_start,
                                plan2_start
                            )
                            workload_analysis = calculate_workload_conflict(overlapping_tasks)
                            overlap_info['total_tasks_plan1'] = overlapping_tasks['total_plan1_tasks']
                            overlap_info['total_tasks_plan2'] = overlapping_tasks['total_plan2_tasks']
                            overlap_info['conflict_level'] = workload_analysis['conflict_level']
                            overlap_info['avg_hours_per_week'] = workload_analysis['avg_hours_per_week']
                        
                        all_overlaps.append(overlap_info)
            
            # Generate summary statistics
            total_overlap_days = sum(overlap['overlap_days'] for overlap in all_overlaps)
            
            response = {
                'success': True,
                'message': f'Found {len(all_plans)} plan(s) with {len(all_overlaps)} overlap(s)',
                'total_plans': len(all_plans),
                'total_overlaps': len(all_overlaps),
                'total_overlap_days': total_overlap_days,
                'plans': plans_summary,
                'overlaps': all_overlaps
            }
            
            # Add warnings if there are overlaps
            if all_overlaps:
                if include_tasks:
                    critical_overlaps = [o for o in all_overlaps if o.get('conflict_level') == 'critical']
                    high_overlaps = [o for o in all_overlaps if o.get('conflict_level') == 'high']
                    
                    if critical_overlaps:
                        response['warning'] = f'You have {len(critical_overlaps)} critical workload conflict(s)! Please review and reschedule tasks.'
                    elif high_overlaps:
                        response['warning'] = f'You have {len(high_overlaps)} high workload conflict(s). Consider adjusting your schedule.'
                else:
                    response['warning'] = f'You have {len(all_overlaps)} overlapping plan(s). Use include_tasks=true to see workload analysis.'
            
            return response
        
        else:
            raise HTTPException(status_code=400, detail=f'Unknown action: {action}')
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in plan_manager: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
