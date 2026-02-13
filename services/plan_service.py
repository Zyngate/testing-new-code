# stelle_backend/services/plan_service.py

import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from groq import Groq
import os
from config import logger
from services.recommendation_plan_integration import get_recommendation_context_sync

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
    
    def generate_subtask_resources(self, goal: str, task_title: str, subtask_title: str) -> Optional[Dict[str, Any]]:
        """Generate learning resources (YouTube videos and websites) for a specific subtask"""
        if not groq_client:
            return None
        
        try:
            prompt = f"""
You are a learning resource curator. Generate helpful learning resources for this subtask.

Goal: {goal}
Task: {task_title}
Subtask: {subtask_title}

Provide 1-2 YouTube videos and 1-2 websites that would help complete this specific subtask.
Be practical and specific - only suggest resources that directly help with this action.

Return ONLY valid JSON in this exact format:
{{
  "youtube_videos": [
    {{
      "title": "Specific video topic",
      "search_query": "Exact YouTube search term"
    }}
  ],
  "websites": [
    {{
      "name": "Website/documentation name",
      "url": "https://actual-url.com"
    }}
  ]
}}

If no relevant resources are needed for this subtask, return empty arrays.
"""
            
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a learning resource curator. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=500
            )
            
            resources = json.loads(response.choices[0].message.content)
            
            # Only return if there are actual resources
            if (resources.get("youtube_videos") and len(resources["youtube_videos"]) > 0) or \
               (resources.get("websites") and len(resources["websites"]) > 0):
                return resources
            return None
            
        except Exception as e:
            logger.error(f"Error generating subtask resources: {str(e)}")
            return None
    
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
    
    def _generate_week_outline(self, goal: str, weeks: int, start_date: datetime, end_date: datetime, recommendation_context: str = "") -> Optional[List[Dict]]:
        """
        Phase 1: Generate a unique week-by-week outline (milestones + task titles only).
        This prevents repetition by planning all weeks at a high level BEFORE filling details.
        Small output = model stays creative and doesn't fall into copy-paste patterns.
        """
        if not groq_client:
            return None
        
        try:
            prompt = f"""Create a UNIQUE week-by-week outline for this {weeks}-week plan.

Goal: {goal}
Timeline: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}
{recommendation_context}

CRITICAL UNIQUENESS RULES:
1. Each week has a UNIQUE milestone, focus_area, and 5 UNIQUE task titles (Mon-Fri)
2. NO task title may repeat or be similar across ANY weeks
3. A task type may appear at MOST ONCE across ALL weeks:
   - "Monitor comments" → max 1 time total
   - "Create social media posts" → max 1 time total
   - "Review/adjust strategy" → max 1 time total
   - "Engage audience" → max 1 time total
4. Setup tasks (create accounts, set up analytics, install tools) → Week 1 ONLY
5. Each week MUST introduce a DIFFERENT type of work. Vary between:
   - Content creation (different pieces each time)
   - SEO optimization (technical SEO, on-page, off-page — each once)
   - Email marketing (list building, sequences, newsletters)
   - Outreach (guest posts, partnerships, backlinks)
   - Repurposing (turn blog to video, infographic, podcast)
   - Community building (forums, comments, groups)
   - Paid promotion (ads setup, retargeting, A/B testing)
   - Data analysis (traffic, conversions, user behavior)
   - Technical improvements (site speed, mobile, schema markup)
   - Networking (collaborations, interviews, webinars)
6. Later weeks must BUILD on earlier weeks' results (reference prior output)
7. Week titles must show clear PROGRESSION toward the goal

BANNED words in task titles: "Research", "Find", "Search", "Identify", "Explore", "Investigate", "Brainstorm", "Gather", "Determine"

Return ONLY this JSON:
{{
  "weeks": [
    {{
      "week_number": 1,
      "milestone": "Unique milestone for this week",
      "focus_area": "Unique focus area",
      "tasks": ["Mon task title", "Tue task title", "Wed task title", "Thu task title", "Fri task title"]
    }}
  ]
}}"""

            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative project planner. Generate MAXIMALLY DIVERSE week outlines. Every single week must have completely different task types — NO repetition allowed. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=min(1500 + (weeks * 120), 8000)
            )
            
            outline = json.loads(response.choices[0].message.content)
            weeks_data = outline.get("weeks", [])
            
            if weeks_data and len(weeks_data) >= weeks * 0.7:  # Accept if at least 70% of weeks generated
                logger.info(f"Phase 1 outline generated: {len(weeks_data)} weeks with unique tasks")
                return weeks_data
            else:
                logger.warning(f"Outline incomplete: got {len(weeks_data)} weeks, expected {weeks}")
                return None
                
        except Exception as e:
            logger.warning(f"Outline generation failed, proceeding without: {e}")
            return None
    
    def _post_process_plan(self, plan_dict: Dict) -> Dict:
        """
        Post-process the generated plan to:
        1. Replace banned words ONLY at the start of titles (context-safe)
        2. Ensure recommendation fields exist on all tasks
        3. Detect and log repetitive patterns
        """
        import re
        
        # Only replace banned words when they START a title/sentence.
        # This avoids nonsense like "Apply high-authority sites in the niche"
        # (from "Search for high-authority sites in the niche").
        TITLE_START_REPLACEMENTS = {
            r'^Research\s+and\s+': '',           # "Research and analyze..." → "Analyze..."
            r'^Research\s+': 'Apply ',           # "Research trends" → "Apply trends"
            r'^Find\s+': 'Select ',              # "Find tools" → "Select tools"
            r'^Search\s+for\s+': 'Select ',      # "Search for sites" → "Select sites"
            r'^Look\s+up\s+': 'Review ',         # "Look up docs" → "Review docs"
            r'^Gather\s+': 'Compile ',           # "Gather data" → "Compile data"
            r'^Identify\s+': 'Define ',          # "Identify audience" → "Define audience"
            r'^Explore\s+': 'Implement ',        # "Explore options" → "Implement options"
            r'^Investigate\s+': 'Analyze ',      # "Investigate issue" → "Analyze issue"
            r'^Determine\s+': 'Establish ',      # "Determine scope" → "Establish scope"
            r'^Discover\s+': 'Leverage ',        # "Discover trends" → "Leverage trends"
            r'^Brainstorm\s+': 'Draft ',         # "Brainstorm ideas" → "Draft ideas"
        }
        
        # For descriptions, just strip the verb prefix if it starts with a banned word
        BANNED_WORDS_RE = re.compile(
            r'\b(research|find|search for|look up|gather|identify|explore|'
            r'investigate|determine|discover|brainstorm)\b',
            re.IGNORECASE
        )
        
        def fix_title(text: str) -> str:
            """Fix banned words only at the START of titles — safe replacements."""
            if not text:
                return text
            for pattern, replacement in TITLE_START_REPLACEMENTS.items():
                new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if new_text != text:
                    # Ensure first char is capitalized after replacement
                    if new_text and new_text[0].islower():
                        new_text = new_text[0].upper() + new_text[1:]
                    return new_text
            return text
        
        def check_description(text: str, task_title: str) -> str:
            """Log banned words in descriptions but don't blindly replace mid-sentence."""
            if not text:
                return text
            matches = BANNED_WORDS_RE.findall(text)
            if matches:
                logger.info(f"Banned words in description of '{task_title}': {matches} (kept as-is for context)")
            return text
        
        # Track title frequencies to detect repetition
        title_counts = {}
        
        for week in plan_dict.get("weekly_plans", []):
            for task in week.get("tasks", []):
                # Fix banned words in title (start-of-title only — safe)
                task["title"] = fix_title(task.get("title", ""))
                # Check descriptions but don't blindly replace mid-sentence
                task["description"] = check_description(task.get("description", ""), task["title"])
                
                # Track repetitive titles
                title_key = task["title"].lower().strip()
                title_counts[title_key] = title_counts.get(title_key, 0) + 1
                
                # Ensure recommendation fields exist
                if "recommended_posting_time" not in task:
                    task["recommended_posting_time"] = None
                if "target_platform" not in task:
                    task["target_platform"] = None
                
                # Fix banned words in subtasks (title-start only)
                for subtask in task.get("subtasks", []):
                    if isinstance(subtask, dict):
                        subtask["title"] = fix_title(subtask.get("title", ""))
                        if "description" in subtask:
                            subtask["description"] = check_description(subtask.get("description", ""), subtask.get("title", ""))
        
        # Log repetition warnings
        repeated = {k: v for k, v in title_counts.items() if v > 1}
        if repeated:
            logger.warning(f"Repetitive task titles detected (post-processing): {repeated}")
        
        return plan_dict
    
    def _build_outline_constraint(self, outline: List[Dict], weeks: int, start_date: datetime) -> str:
        """Convert Phase 1 outline into a prompt constraint for Phase 2."""
        if not outline:
            return ""
        
        lines = ["\n\n🎯 MANDATORY WEEK STRUCTURE (YOU MUST USE THESE EXACT TITLES — DO NOT CHANGE THEM):"]
        for week_data in outline[:weeks]:
            wn = week_data.get("week_number", "?")
            milestone = week_data.get("milestone", "")
            focus = week_data.get("focus_area", "")
            tasks = week_data.get("tasks", [])
            
            lines.append(f"\nWeek {wn}: milestone=\"{milestone}\", focus=\"{focus}\"")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for i, task_title in enumerate(tasks[:5]):
                day = days[i] if i < len(days) else days[0]
                lines.append(f"  {day}: \"{task_title}\"")
        
        lines.append("\n⚠️ USE THE EXACT TASK TITLES ABOVE. Add detailed descriptions, subtasks, and real data for each.")
        return "\n".join(lines)
    
    def generate_weekly_plan(self, goal: str, start_date: datetime, end_date: datetime, previous_plan: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Generate a weekly action plan using Groq AI, integrated with recommendation engine data."""
        if not groq_client:
            raise Exception("Groq API key not configured")
        
        weeks = self.calculate_weeks(start_date, end_date)
        
        # Limit too long plans to prevent JSON truncation
        if weeks > 20:
            weeks = 20
        
        # Fetch recommendation engine context for social media goals
        recommendation_context = get_recommendation_context_sync(goal)
        
        # Phase 1: Generate unique week outline to prevent repetition
        outline = self._generate_week_outline(goal, weeks, start_date, end_date, recommendation_context)
        outline_constraint = self._build_outline_constraint(outline, weeks, start_date) if outline else ""
        
        prompt = f"""You are a senior AI planning system and domain expert. Create a detailed weekly plan.

Goal: {goal}
Timeline: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}
Duration: {weeks} weeks
{recommendation_context}
{outline_constraint}

RULES:
1. Return ONLY valid JSON. {weeks} weeks, exactly 5 tasks/week (Mon-Fri only), 2-3 subtasks per task.
2. YOU are the domain expert. Every description MUST contain REAL specifics — actual numbers, actual tools, actual steps, actual content. User executes immediately with zero Googling.
3. BANNED title/subtask words: "Research", "Find", "Search", "Identify", "Explore", "Investigate", "Brainstorm", "Gather", "Determine", "Discover", "Look up"
4. Subtasks = execution actions with embedded data. Verbs: Write, Publish, Configure, Apply, Draft, Submit, Schedule, Deploy, Launch, Record.
5. Subtasks can include optional resources: {{"youtube_videos": [{{"title": "...", "search_query": "..."}}], "websites": [{{"name": "...", "url": "..."}}]}}
6. For social media tasks: set recommended_posting_time and target_platform fields.

QUALITY STANDARD (every description must match this detail level):
BAD: "Write blog post about AI trends"
GOOD: "Write 2000-word post: '5 AI Agent Frameworks Reshaping Enterprise Automation in 2026' covering LangGraph (event-driven graphs), CrewAI (role-based agents), AutoGen (multi-agent conversations), Semantic Kernel (.NET integration), BabyAGI (task-driven). Include comparison table, code snippet for each, 3 use cases per framework. Target keyword: 'AI agent frameworks 2026' (8.1K monthly searches, difficulty: 34)."

JSON structure:
{{"weekly_plans": [{{"week_number": 1, "week_start": "{start_date.strftime('%m/%d/%Y')}", "week_end": "{(start_date + timedelta(days=6)).strftime('%m/%d/%Y')}", "milestone": "...", "focus_area": "...", "tasks": [{{"task_id": "task_w1_t1", "title": "...", "description": "DETAILED with real data/numbers/names", "priority": "High|Medium|Low", "estimated_hours": 3, "dependencies": [], "day_of_week": "Monday", "status": "pending", "recommended_posting_time": null, "target_platform": null, "subtasks": [{{"title": "Specific action with real data embedded", "resources": {{}}}}]}}]}}]}}

Each week_start = previous + 7 days. task_id format: task_wN_tM.
"""
        
        try:
            # Call Groq (response format JSON enforced)
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior expert who provides COMPLETE, READY-TO-USE data — actual trends, numbers, titles, strategies. The user NEVER needs to Google anything. CRITICAL: Each week MUST be completely UNIQUE — different task titles, different focus, different deliverables. NEVER repeat the same task pattern across weeks. Week N must build on Week N-1 results. Return pure JSON output only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
                max_tokens=min(10000 + (weeks * 800), 32000)
            )
            
            content = response.choices[0].message.content
            
            # Convert string JSON to dict
            plan_dict = json.loads(content)
            
            # Validate weekday rule and add dates
            valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for week in plan_dict["weekly_plans"]:
                # Parse week start date
                week_start = datetime.strptime(week["week_start"], "%m/%d/%Y")
                
                for i, task in enumerate(week["tasks"]):
                    if task["day_of_week"] not in valid_days:
                        task["day_of_week"] = valid_days[i]  # auto correct fallback
                    
                    # Calculate and add actual date for the task
                    day_offset = valid_days.index(task["day_of_week"])
                    task_date = week_start + timedelta(days=day_offset)
                    task["date"] = task_date.strftime("%m/%d/%Y")
                    
                    # Ensure subtasks field exists and has content
                    if "subtasks" not in task or not task["subtasks"] or len(task["subtasks"]) == 0:
                        # Generate default actionable subtasks based on task title
                        task["subtasks"] = [
                            {"title": f"Break down {task['title']} into smaller steps"},
                            {"title": f"Execute the main action for {task['title']}"},
                            {"title": f"Review and verify {task['title']} completion"}
                        ]
                    else:
                        # Normalize subtasks - handle both string and object formats
                        normalized_subtasks = []
                        for subtask in task["subtasks"]:
                            if isinstance(subtask, str):
                                # Convert string subtask to object format
                                normalized_subtasks.append({"title": subtask})
                            elif isinstance(subtask, dict):
                                # Ensure dict has title
                                if "title" not in subtask:
                                    subtask["title"] = "Subtask action"
                                normalized_subtasks.append(subtask)
                        task["subtasks"] = normalized_subtasks
            
            # Phase 3: Post-process to fix banned words and ensure fields
            plan_dict = self._post_process_plan(plan_dict)
            
            return plan_dict
            
        except json.JSONDecodeError:
            logger.error("AI JSON parsing failed — output too large or malformed")
            raise Exception("AI JSON parsing failed — output too large or malformed")
        except Exception as e:
            logger.error(f"Error in generate_weekly_plan: {str(e)}")
            raise
    
    def reorder_tasks_in_week(self, week_plan: Dict, moved_task_id: str, new_position: int, new_date: str = None) -> Dict:
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
            return {'success': False, 'warnings': []}
        
        # Remove task from old position
        tasks.pop(old_index)
        
        # Insert at new position
        new_index = max(0, min(new_position, len(tasks)))
        tasks.insert(new_index, moved_task)
        
        warnings = []
        
        # Update date if provided
        if new_date:
            moved_task['date'] = new_date
            # Update day_of_week based on date
            try:
                task_date = datetime.strptime(new_date, "%m/%d/%Y")
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                moved_task['day_of_week'] = days[task_date.weekday()]
                
                # Check if the new date is a weekend
                if task_date.weekday() >= 5:  # Saturday=5, Sunday=6
                    warnings.append(f"⚠️ Task moved to {days[task_date.weekday()]} ({new_date}). Consider moving to a weekday for better productivity.")
            except:
                pass
        
        # Check for multiple tasks on same date
        if moved_task.get('date'):
            tasks_on_same_date = []
            for task in tasks:
                if task.get('date') == moved_task['date'] and task['task_id'] != moved_task_id:
                    tasks_on_same_date.append(task['title'])
            
            if tasks_on_same_date:
                warnings.append(f"ℹ️ Multiple tasks scheduled for {moved_task['date']}: {', '.join([moved_task['title']] + tasks_on_same_date)}")
        
        return {'success': True, 'warnings': warnings}
    
    def move_task_between_weeks(
        self, 
        plan_data: Dict, 
        task_id: str, 
        source_week_number: int, 
        target_week_number: int, 
        new_position: int = 0,
        max_tasks_per_week: int = 7
    ) -> Dict[str, Any]:
        """
        Move a task from one week to another week
        
        Args:
            plan_data: The plan data containing weekly_plans
            task_id: ID of the task to move
            source_week_number: Week number where task currently exists
            target_week_number: Week number where task should be moved
            new_position: Position in target week (0-indexed)
            max_tasks_per_week: Maximum tasks allowed per week (default 7)
            
        Returns:
            Dict with success status, message, and optional warnings
        """
        weekly_plans = plan_data.get('weekly_plans', [])
        
        # Find source and target weeks
        source_week = None
        target_week = None
        
        for week in weekly_plans:
            if week['week_number'] == source_week_number:
                source_week = week
            if week['week_number'] == target_week_number:
                target_week = week
        
        # Validate weeks exist
        if not source_week:
            return {
                'success': False,
                'message': f'Source week {source_week_number} not found'
            }
        
        if not target_week:
            return {
                'success': False,
                'message': f'Target week {target_week_number} not found'
            }
        
        # Find the task in source week
        moved_task = None
        task_index = None
        
        for i, task in enumerate(source_week['tasks']):
            if task['task_id'] == task_id:
                moved_task = task.copy()  # Create a copy
                task_index = i
                break
        
        if moved_task is None:
            return {
                'success': False,
                'message': f'Task {task_id} not found in week {source_week_number}'
            }
        
        # Check if target week has space (if it's the same week, we're just reordering)
        if source_week_number != target_week_number:
            current_task_count = len(target_week['tasks'])
            
            if current_task_count >= max_tasks_per_week:
                return {
                    'success': False,
                    'message': f'Cannot move task. Week {target_week_number} already has {current_task_count} tasks (maximum is {max_tasks_per_week}). Please remove or move a task from that week first.',
                    'target_week_task_count': current_task_count,
                    'max_tasks_allowed': max_tasks_per_week
                }
        
        # Remove task from source week
        source_week['tasks'].pop(task_index)
        
        # Update task's week reference if it exists
        moved_task['week_number'] = target_week_number
        
        # Calculate new date based on target week - FIXED LOGIC
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        warnings = []
        
        try:
            target_week_start = datetime.strptime(target_week['week_start'], "%m/%d/%Y")
            
            # Maintain same day of week in new week
            if moved_task.get('day_of_week') in days:
                day_offset = days.index(moved_task['day_of_week'])
                new_task_date = target_week_start + timedelta(days=day_offset)
                moved_task['date'] = new_task_date.strftime("%m/%d/%Y")
                
                # Check if the new date is a weekend
                if new_task_date.weekday() >= 5:  # Saturday=5, Sunday=6
                    warnings.append(f"⚠️ Task moved to {days[new_task_date.weekday()]} ({moved_task['date']}). Consider moving to a weekday for better productivity.")
            else:
                # If day_of_week not set, use first weekday of target week
                moved_task['day_of_week'] = days[0]  # Default to Monday
                new_task_date = target_week_start
                moved_task['date'] = new_task_date.strftime("%m/%d/%Y")
        except Exception as e:
            # If date calculation fails, use target week start
            try:
                target_week_start = datetime.strptime(target_week['week_start'], "%m/%d/%Y")
                moved_task['date'] = target_week_start.strftime("%m/%d/%Y")
                moved_task['day_of_week'] = days[target_week_start.weekday()]
            except:
                pass
        
        # Insert task at specified position in target week
        target_position = max(0, min(new_position, len(target_week['tasks'])))
        target_week['tasks'].insert(target_position, moved_task)
        
        # Check for multiple tasks on same date
        tasks_on_same_date = []
        for task in target_week['tasks']:
            if task.get('date') == moved_task.get('date') and task['task_id'] != moved_task['task_id']:
                tasks_on_same_date.append(task['title'])
        
        if tasks_on_same_date:
            warnings.append(f"ℹ️ Multiple tasks scheduled for {moved_task['date']}: {', '.join([moved_task['title']] + tasks_on_same_date)}")
        
        # Success response
        result = {
            'success': True,
            'message': f'Task "{moved_task["title"]}" moved from week {source_week_number} to week {target_week_number}',
            'moved_task': moved_task,
            'source_week': source_week_number,
            'target_week': target_week_number,
            'new_position': target_position,
            'source_week_task_count': len(source_week['tasks']),
            'target_week_task_count': len(target_week['tasks']),
            'new_date': moved_task.get('date'),
            'new_day_of_week': moved_task.get('day_of_week')
        }
        
        # Add warnings
        if warnings:
            result['warnings'] = warnings
        
        # Add warning if target week is getting full
        if len(target_week['tasks']) >= max_tasks_per_week - 1:
            if 'warnings' not in result:
                result['warnings'] = []
            result['warnings'].append(f'Week {target_week_number} now has {len(target_week["tasks"])} tasks. Consider spreading tasks across weeks for better planning.')
        
        return result
    
    def update_task_date(self, plan_data: Dict, task_id: str, new_date: str = None, new_day_of_week: str = None) -> bool:
        """Update task's date and day of week without AI regeneration"""
        for week in plan_data['weekly_plans']:
            for task in week['tasks']:
                if task['task_id'] == task_id:
                    if new_date:
                        task['date'] = new_date
                        # Update day_of_week based on date
                        try:
                            task_date = datetime.strptime(new_date, "%m/%d/%Y")
                            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                            task['day_of_week'] = days[task_date.weekday()]
                        except:
                            if new_day_of_week:
                                task['day_of_week'] = new_day_of_week
                    elif new_day_of_week:
                        task['day_of_week'] = new_day_of_week
                        # Try to update date based on week and day
                        try:
                            week_start = datetime.strptime(week['week_start'], "%m/%d/%Y")
                            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                            if new_day_of_week in days:
                                day_offset = days.index(new_day_of_week)
                                task_date = week_start + timedelta(days=day_offset)
                                task['date'] = task_date.strftime("%m/%d/%Y")
                        except:
                            pass
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
        
        # Fetch recommendation engine context for social media goals
        recommendation_context = get_recommendation_context_sync(original_goal)
        
        # Phase 1: Generate unique week outline to prevent repetition
        outline = self._generate_week_outline(original_goal, weeks, start_date, end_date, recommendation_context)
        outline_constraint = self._build_outline_constraint(outline, weeks, start_date) if outline else ""
        
        # Create detailed prompt with strict JSON format
        prompt = f"""You are regenerating a weekly plan. Build on completed work, skip finished tasks.

Goal: {original_goal}
Timeline: {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')}
Duration: {weeks} weeks{completed_info}
{recommendation_context}
{outline_constraint}

RULES:
1. Return ONLY valid JSON. {weeks} weeks, exactly 5 tasks/week (Mon-Fri only), 2-3 subtasks per task.
2. YOU are the domain expert. Every description MUST contain REAL specifics — actual numbers, tools, steps, content. Zero Googling needed.
3. BANNED title/subtask words: "Research", "Find", "Search", "Identify", "Explore", "Investigate", "Brainstorm", "Gather", "Determine", "Discover", "Look up"
4. Subtasks = execution actions with embedded data. Verbs: Write, Publish, Configure, Apply, Draft, Submit, Schedule, Deploy, Launch, Record.
5. Subtasks can include optional resources: {{"youtube_videos": [{{"title": "...", "search_query": "..."}}], "websites": [{{"name": "...", "url": "..."}}]}}
6. For social media tasks: set recommended_posting_time and target_platform fields.
7. DO NOT include overall_strategy field.

QUALITY STANDARD (every description must match this detail level):
BAD: "Write blog post about AI trends"
GOOD: "Write 2000-word post: '5 AI Agent Frameworks Reshaping Enterprise Automation in 2026' covering LangGraph, CrewAI, AutoGen, Semantic Kernel, BabyAGI. Include comparison table, code snippet for each. Target keyword: 'AI agent frameworks 2026' (8.1K monthly searches, difficulty: 34)."

JSON structure:
{{"weekly_plans": [{{"week_number": 1, "week_start": "{start_date.strftime('%m/%d/%Y')}", "week_end": "{(start_date + timedelta(days=6)).strftime('%m/%d/%Y')}", "milestone": "...", "focus_area": "...", "tasks": [{{"task_id": "task_w1_t1", "title": "...", "description": "DETAILED with real data", "priority": "High|Medium|Low", "estimated_hours": 3, "dependencies": [], "day_of_week": "Monday", "status": "pending", "recommended_posting_time": null, "target_platform": null, "subtasks": [{{"title": "Specific action with real data", "resources": {{}}}}]}}]}}]}}

Each week_start = previous + 7 days. task_id format: task_wN_tM.
"""
        
        try:
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior domain expert who provides COMPLETE data — actual trends, numbers, strategies, titles, content. The user NEVER needs to Google anything. CRITICAL: Each week MUST be completely UNIQUE — different tasks, different focus, different deliverables. NEVER repeat the same weekly pattern. Always respond with valid JSON only. NEVER include 'overall_strategy' field."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,  # Balanced: varied enough to avoid repetition, consistent enough for quality
                response_format={"type": "json_object"},  # Force JSON response
                max_tokens=min(10000 + (weeks * 800), 32000)
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
            
            # Post-process to fix banned words and ensure fields
            plan_data = self._post_process_plan(plan_data)
            
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
