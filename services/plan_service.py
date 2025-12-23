# stelle_backend/services/plan_service.py

import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from groq import Groq
import os
from config import logger

# Groq Setup
GROQ_API_KEY = os.getenv('GROQ_API_WEEKLY_PLANNER')
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables!")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Supported platforms
SUPPORTED_PLATFORMS = ['instagram', 'linkedin', 'facebook', 'threads', 'youtube', 'tiktok', 'other']


class WeeklyPlanner:
    def __init__(self):
        self.model = "llama-3.3-70b-versatile"
    
    def calculate_weeks(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate number of weeks between dates"""
        delta = end_date - start_date
        weeks = max(1, (delta.days + 6) // 7)
        return weeks
    
    def detect_goal_type(self, goal: str) -> tuple:
        """Detect goal type based on keywords"""
        goal_lower = goal.lower()
        
        # Check for social media platforms
        for platform in SUPPORTED_PLATFORMS:
            if platform in goal_lower:
                return 'social_media', platform
        
        # Check for cooking/recipe keywords
        cooking_keywords = ['cook', 'recipe', 'pasta', 'dish', 'meal', 'food', 'bake', 'prepare']
        if any(keyword in goal_lower for keyword in cooking_keywords):
            return 'cooking', None
        
        # Default to general
        return 'general', None
    
    def generate_weekly_plan(self, goal: str, start_date: datetime, end_date: datetime, previous_plan: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Generate a weekly action plan using Groq AI"""
        if not groq_client:
            raise Exception("Groq API key not configured")
        
        weeks = self.calculate_weeks(start_date, end_date)
        
        # Limit too long plans to prevent JSON truncation
        if weeks > 20:
            weeks = 20
        
        # Dynamic context — no predefined goal type
        context = (
            "Analyze the goal text and derive needed themes, skills, tasks, "
            "and progress structure based on logical reasoning."
        )
        
        # 5 task per week rule
        focus_areas = (
            "Each week must include exactly 5 tasks assigned strictly to weekdays "
            "(Monday to Friday). Saturday and Sunday must not contain any tasks."
        )
        
        prompt = f"""
You are a senior AI planning system. Create a structured weekly plan.

Goal:
{goal}

Timeline: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}
Duration: {weeks} weeks

Context:
{context}

⚠️ STRICT RULES:
1. JSON ONLY — no extra words, no markdown.
2. Weekly plan MUST contain exactly {weeks} week blocks.
3. EACH WEEK must contain exactly 5 tasks.
4. Tasks MUST only be scheduled Monday–Friday.
5. No weekend tasks allowed.
6. Each task must be unique.
7. Do not leave any field empty.
8. DO NOT include overall_strategy.

JSON FORMAT TEMPLATE:
{{
  "weekly_plans": [
    {{
      "week_number": 1,
      "week_start": "{start_date.strftime('%m/%d/%Y')}",
      "week_end": "{(start_date + timedelta(days=6)).strftime('%m/%d/%Y')}",
      "milestone": "Specific target of the week",
      "focus_area": "Theme of the week",
      "tasks": [
        {{
          "task_id": "task_w1_t1",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "High",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Monday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t2",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Medium",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Tuesday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t3",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Medium",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Wednesday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t4",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Low",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Thursday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t5",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Low",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Friday",
          "status": "pending"
        }}
      ]
    }}
  ]
}}
"""
        
        try:
            # Call Groq (response format JSON enforced)
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return pure JSON output only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.15,
                response_format={"type": "json_object"},
                max_tokens=3000 + (weeks * 200)
            )
            
            content = response.choices[0].message.content
            
            # Convert string JSON to dict
            plan_dict = json.loads(content)
            
            # Validate weekday rule
            valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for week in plan_dict["weekly_plans"]:
                for i, task in enumerate(week["tasks"]):
                    if task["day_of_week"] not in valid_days:
                        task["day_of_week"] = valid_days[i]  # auto correct fallback
            
            return plan_dict
            
        except json.JSONDecodeError:
            logger.error("AI JSON parsing failed — output too large or malformed")
            raise Exception("AI JSON parsing failed — output too large or malformed")
        except Exception as e:
            logger.error(f"Error in generate_weekly_plan: {str(e)}")
            raise
    
    def reorder_tasks_in_week(self, week_plan: Dict, moved_task_id: str, new_position: int) -> bool:
        """Reorder tasks within a week without AI - simple drag and drop"""
        tasks = week_plan['tasks']
        
        # Find the moved task
        moved_task = None
        old_index = None
        
        for i, task in enumerate(tasks):
            if task['task_id'] == moved_task_id:
                moved_task = task
                old_index = i
                break
        
        if moved_task is None:
            return False
        
        # Remove task from old position
        tasks.pop(old_index)
        
        # Insert at new position
        new_index = max(0, min(new_position, len(tasks)))
        tasks.insert(new_index, moved_task)
        
        return True
    
    def update_task_date(self, plan_data: Dict, task_id: str, new_day_of_week: str) -> bool:
        """Update task's day of week without AI regeneration"""
        for week in plan_data['weekly_plans']:
            for task in week['tasks']:
                if task['task_id'] == task_id:
                    task['day_of_week'] = new_day_of_week
                    return True
        return False
    
    def regenerate_plan(self, original_goal: str, start_date: datetime, end_date: datetime, current_progress: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Regenerate plan only when explicitly requested (e.g., goal changed, Date range changed)"""
        if not groq_client:
            raise Exception("Groq API key not configured")
        
        # Include what user has completed - with defensive checks
        completed_tasks = []
        
        # Handle case where current_progress might be None or malformed
        if current_progress and isinstance(current_progress, dict):
            weekly_plans = current_progress.get('weekly_plans', [])
            
            # Ensure weekly_plans is a list
            if isinstance(weekly_plans, list):
                for week in weekly_plans:
                    # Ensure week is a dictionary
                    if isinstance(week, dict):
                        tasks = week.get('tasks', [])
                        
                        # Ensure tasks is a list
                        if isinstance(tasks, list):
                            for task in tasks:
                                # Ensure task is a dictionary, not a string
                                if isinstance(task, dict):
                                    if task.get('status') == 'completed':
                                        completed_tasks.append(task.get('title', 'Unnamed task'))
                                else:
                                    logger.warning(f"Task is not a dictionary: {type(task)}")
                        else:
                            logger.warning(f"Tasks is not a list: {type(tasks)}")
                    else:
                        logger.warning(f"Week is not a dictionary: {type(week)}")
            else:
                logger.warning(f"weekly_plans is not a list: {type(weekly_plans)}")
        else:
            logger.warning(f"current_progress is invalid: {type(current_progress)}")
        
        completed_info = f"\n\nUser has already completed these tasks:\n- " + "\n- ".join(completed_tasks) if completed_tasks else ""
        
        weeks = self.calculate_weeks(start_date, end_date)
        
        # Create detailed prompt with strict JSON format
        prompt = f"""You are regenerating a weekly plan.

Goal: {original_goal}
Timeline: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}
Duration: {weeks} weeks{completed_info}

⚠️ STRICT RULES:
1. JSON ONLY — no extra words, no markdown.
2. Weekly plan MUST contain exactly {weeks} week blocks.
3. EACH WEEK must contain exactly 5 tasks.
4. Tasks MUST only be scheduled Monday–Friday.
5. Each task must be unique.
6. Each week MUST have: week_number, milestone, focus_area, tasks array
7. DO NOT include overall_strategy field.

JSON FORMAT (follow exactly):
{{
  "weekly_plans": [
    {{
      "week_number": 1,
      "week_start": "{start_date.strftime('%m/%d/%Y')}",
      "week_end": "{(start_date + timedelta(days=6)).strftime('%m/%d/%Y')}",
      "milestone": "Specific target of the week",
      "focus_area": "Theme of the week",
      "tasks": [
        {{
          "task_id": "task_w1_t1",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "High",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Monday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t2",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Medium",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Tuesday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t3",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Medium",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Wednesday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t4",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Low",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Thursday",
          "status": "pending"
        }},
        {{
          "task_id": "task_w1_t5",
          "title": "Task title",
          "description": "Detailed explanation",
          "priority": "Low",
          "estimated_hours": 3,
          "dependencies": [],
          "day_of_week": "Friday",
          "status": "pending"
        }}
      ]
    }}
  ]
}}"""
        
        try:
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert project manager. Always respond with valid JSON only. NEVER include 'overall_strategy' field."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.15,  # Lower temperature for more consistent output
                response_format={"type": "json_object"},  # Force JSON response
                max_tokens=3000 + (weeks * 200)
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            content = content.strip()
            plan_data = json.loads(content)
            
            # Remove overall_strategy if present
            if 'overall_strategy' in plan_data:
                del plan_data['overall_strategy']
            
            # Validate and fix the plan structure
            if 'weekly_plans' not in plan_data:
                raise Exception("Invalid plan structure: missing 'weekly_plans'")
            
            current_date = start_date
            valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            
            for week_index, week_plan in enumerate(plan_data['weekly_plans']):
                # Ensure week_number exists
                if 'week_number' not in week_plan:
                    week_plan['week_number'] = week_index + 1
                    logger.warning(f"Added missing week_number: {week_index + 1}")
                
                # Add/update dates
                week_end = min(current_date + timedelta(days=6), end_date)
                week_plan['week_start'] = current_date.strftime('%m/%d/%Y')
                week_plan['week_end'] = week_end.strftime('%m/%d/%Y')
                current_date = week_end + timedelta(days=1)
                
                # Validate tasks
                if 'tasks' not in week_plan:
                    week_plan['tasks'] = []
                    logger.warning(f"Added missing tasks array for week {week_plan['week_number']}")
                
                # Ensure milestone and focus_area exist
                if 'milestone' not in week_plan:
                    week_plan['milestone'] = f"Week {week_plan['week_number']} milestone"
                if 'focus_area' not in week_plan:
                    week_plan['focus_area'] = f"Week {week_plan['week_number']} focus"
                
                # Ensure each task has required fields
                for task_index, task in enumerate(week_plan['tasks']):
                    if 'task_id' not in task:
                        task['task_id'] = f"task_w{week_plan['week_number']}_t{task_index + 1}"
                    if 'title' not in task:
                        task['title'] = f"Task {task_index + 1}"
                    if 'description' not in task:
                        task['description'] = "Task description"
                    if 'priority' not in task:
                        task['priority'] = "Medium"
                    if 'estimated_hours' not in task:
                        task['estimated_hours'] = 3
                    if 'day_of_week' not in task or task['day_of_week'] not in valid_days:
                        task['day_of_week'] = valid_days[task_index % 5]
                    if 'status' not in task:
                        task['status'] = 'pending'
                    if 'dependencies' not in task:
                        task['dependencies'] = []
            
            logger.info(f"Successfully regenerated plan with {len(plan_data['weekly_plans'])} weeks")
            return plan_data
            
        except json.JSONDecodeError as je:
            logger.error(f"JSON parsing error: {str(je)}")
            logger.error(f"Content received: {content[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Error in regenerate_plan: {str(e)}")
            logger.error(traceback.format_exc())
            return None


# Singleton instance
planner = WeeklyPlanner()
