# stelle_backend/routes/goal_routes.py
# stelle_backend/routes/goal_routes.py
import uuid
import json
import re
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional # <-- FIX 18-22: Missing typing imports (List, Dict, Any)
from groq import AsyncGroq # <-- FIX 23: Missing 'AsyncGroq'

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import pytz

from models.common_models import PlanWeekRequest
from database import goals_collection, users_collection, weekly_plans_collection, chats_collection, db
from services.common_utils import get_current_datetime, convert_object_ids, filter_think_messages
from services.ai_service import get_groq_client, generate_text_embedding # <-- FIX 24: Missing 'generate_text_embedding'
from services.recommendation_plan_integration import get_recommendation_context_for_plan
from config import logger, PLANNING_KEY, GOAL_SETTING_KEY

router = APIRouter()

# Collection for user post analytics (used by recommendation engine)
user_post_analytics_collection = db["user_post_analytics"]


# --- Utility for Plan Generation ---

async def generate_weekly_plan(user_id: str, goals_data: List[Dict[str, Any]], user_tz: pytz.BaseTzInfo) -> Dict[str, Any]:
    """Generates the strategic weekly plan using the LLM, integrated with recommendation engine data."""
    
    now_local = datetime.now(user_tz)
    today = now_local.date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    
    # Fetch recommendation engine context for social media goals
    goal_titles = ' '.join([g.get('title', '') for g in goals_data])
    recommendation_context = await get_recommendation_context_for_plan(
        user_id=user_id,
        goal_text=goal_titles,
        goals_data=goals_data,
        analytics_collection=user_post_analytics_collection
    )
    
    # Prepare goals and tasks context
    goals_summary = []
    for goal in goals_data:
        tasks_summary = []
        for task in goal.get('tasks', []):
            task_status = task.get('status', 'not started')
            task_deadline_str = task.get('deadline')
            deadline_status_info = ""
            if task_deadline_str and task_status not in ['completed', 'cancelled']:
                try:
                    # Deadline format is 'YYYY-MM-DD HH:MM'
                    deadline_dt = datetime.strptime(task_deadline_str.split(' ')[0], "%Y-%m-%d").date()
                    if deadline_dt < today:
                        task_status = "Deadline Exceeded"
                        deadline_status_info = f"(Deadline: {task_deadline_str} - EXCEEDED)"
                    else:
                        deadline_status_info = f"(Deadline: {task_deadline_str})"
                except (ValueError, TypeError):
                    deadline_status_info = f"(Deadline: {task_deadline_str})"
            
            tasks_summary.append(
                f"  - Task: {task.get('title')} (ID: {task.get('task_id')}) "
                f"(Status: {task_status}) {deadline_status_info}"
            )
    
        goals_summary.append(
            f"Goal: {goal.get('title')} (ID: {goal.get('goal_id')})\n"
            f"  Description: {goal.get('description', 'N/A')}\n"
            f"  Tasks:\n" + "\n".join(tasks_summary)
        )

    full_context = "\n\n".join(goals_summary)

    # Construct the AI prompt
    prompt = f"""
You are a senior domain expert AND project manager. You DO all analysis, research, and thinking yourself. Create a COMPLETE weekly plan with REAL DATA that the user can execute IMMEDIATELY.

🚨 ABSOLUTE RULES - YOU ARE THE EXPERT, USER DOES ZERO RESEARCH:
- BANNED words in task titles, descriptions, and subtasks: "Research", "Find", "Search", "Look up", "Gather", "Identify", "Explore", "Investigate", "Determine", "Discover", "Brainstorm"
- NEVER say "research trends" → YOU list the actual current trends with data points and numbers
- NEVER say "identify target audience" → YOU define the exact audience with demographics, interests, and behavior
- NEVER say "brainstorm ideas" → YOU provide the actual ready-to-use ideas/titles/strategies
- NEVER say "gather documents" → YOU list EVERY document with exact names
- NEVER say "find contacts" → YOU provide actual company names, emails, phones
- If the goal involves trends, market data, or analysis → YOU provide the actual analyzed data in descriptions
- Every task description MUST contain REAL DATA: actual trends, numbers, examples, strategies, or content
- User opens this plan → Executes immediately → No Googling needed

🚫 ANTI-REPETITION RULES (CRITICAL — HIGHEST PRIORITY):
- EVERY day MUST have completely DIFFERENT task types and titles — NO two days should look alike
- NEVER repeat the same task title pattern across days (e.g., do NOT have "Optimize for SEO" on multiple days)
- NEVER use generic filler tasks like "Monitor performance", "Engage with audience", "Create content calendar" across multiple days
- Each day must represent genuine PROGRESS — Tuesday delivers something NEW that Monday didn't
- VARY task types across the week: strategy, content creation, distribution, outreach, technical setup, optimization, measurement
- Setup/configuration tasks (analytics, accounts, tools) belong on Day 1-2 ONLY — never repeat them
- If a task appears on Monday, a DIFFERENT type of work must appear on Tuesday
- Task descriptions on Day 3+ MUST reference outcomes from Day 1-2 work

📈 WEEKLY PROGRESSION MANDATE:
- Plan the week as a JOURNEY with a clear beginning, middle, and end
- Monday-Tuesday: Foundation work + first major deliverables with SPECIFIC data
- Wednesday: Build on Mon-Tue output, create DIFFERENT deliverables
- Thursday: Distribution, outreach, or scaling based on earlier work
- Friday: Measure results, optimize, prepare for next week
- NEVER have the same type of task (e.g., "write blog post") on more than 2 days

EXAMPLE:
❌ BAD: "Analyze current tech trends" with description "Research and identify current tech trends"
✅ GOOD: "Apply top 5 tech trends to content strategy" with description "Current top trends: (1) AI Agents (enterprise adoption up 340%), (2) Edge AI (on-device processing), (3) Quantum-Safe Encryption ($2.1B market), (4) Spatial Computing, (5) Green Tech. Write content targeting developers and CTOs searching for these topics."

❌ BAD week pattern: Mon=Write blog, Tue=SEO, Wed=Social media calendar, Thu=Engage audience, Fri=Monitor
✅ GOOD week pattern: Mon=Draft blog #1 with full outline+data, Tue=Build email opt-in + lead magnet, Wed=Pitch 5 guest post sites (emails ready), Thu=Create 3 social media variants from blog, Fri=A/B test headlines + review traffic data

❌ BAD subtask: "Research SEO best practices"
✅ GOOD subtask: "Apply SEO settings: meta title under 60 chars, description 150-160 chars, keyword density 1.5-2.5%, add 3-5 internal links, include BlogPosting schema markup, target long-tail keywords with 1K-10K monthly search volume"

Current Date: {today.strftime('%Y-%m-%d')}
Current Calendar Week: {start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}

User's goals and tasks:
---
{full_context}
---
{recommendation_context}

Generate plan in this JSON structure:
{{
  "strategic_approach": "Week's execution strategy with key milestones. If social media goals exist, mention the recommendation engine alignment and optimal posting strategy.",
  "recommendation_alignment": {{
    "platforms_detected": ["List of social media platforms found in goals"],
    "posting_schedule_source": "user_analytics OR research_data",
    "key_insights": ["Top insight from recommendation engine that shaped this plan"]
  }},
  "this_week_plan": [
    {{
      "date": "YYYY-MM-DD",
      "day_of_week": "Monday",
      "daily_focus": "Concrete deliverable (Submit X, Contact Y, Complete Z).",
      "tasks": [
        {{
          "task_id": "Original task ID",
          "title": "Original task title",
          "status": "Current status",
          "description": "Single sentence outcome.",
          "recommended_posting_time": "8:00 AM (only for social media posting tasks, null for others)",
          "target_platform": "instagram (only for social media tasks, null for others)",
          "ai_provided_content": {{
            "complete_data": "ALL information user needs to execute",
            "company_list": [{{"name": "...", "email": "...", "phone": "...", "website": "..."}}],
            "document_checklist": [{{"document": "...", "where_to_get": "...", "fee": "...", "time": "..."}}],
            "email_templates": ["Ready-to-send emails"],
            "step_by_step_guide": "Exact steps: Click X, Type Y, Submit Z",
            "cost_breakdown": "All fees and costs",
            "timeline": "How long each step takes",
            "optimal_posting_info": "For social media tasks: Why this time was chosen based on recommendation engine data. Include engagement multiplier and expected reach."
          }},
          "sub_tasks": [
            {{
              "title": "ACTION verb (Submit/Send/Fill/Contact/Post/Schedule - NOT Research/Find)",
              "description": "One sentence action.",
              "ai_content": "COMPLETE info: exact text to type, exact fields to fill, exact emails to send, exact links to click. For social media tasks: include the EXACT time to post and WHY (from recommendation data). Zero blanks."
            }}
          ]
        }}
      ]
    }}
  ]
}}

**VALIDATION RULES**:
1. Can user complete every sub-task without opening Google? If NO → Add more data
2. Does user know exact companies/contacts? If NO → Add company_list with names, emails, phones
3. Does user know every document needed? If NO → Add document_checklist with all documents
4. Does user have ready emails/templates? If NO → Add email_templates
5. Overdue tasks MUST be first priority
6. Generate all 7 days (empty arrays for no-task days)
7. For social media goals: ALWAYS use the RECOMMENDATION ENGINE DATA above to set posting times, frequencies, and best days
8. For social media content tasks: Schedule content CREATION before the optimal posting window
9. If recommendation engine shows user-specific analytics, PRIORITIZE those over general research data
10. Include "recommended_posting_time" and "target_platform" for every social media related task
11. ANTI-REPETITION CHECK: Are any two days using the same task pattern? If YES → Rewrite to make each day unique
12. PROGRESSION CHECK: Does each day build on the previous day's output? If NO → Add references to prior work

Output ONLY valid JSON.
"""
    # Call the AI
    try:
        from services.ai_service import rate_limited_groq_call
        async_client = AsyncGroq(api_key=PLANNING_KEY)
        response = await rate_limited_groq_call(
            async_client,
            messages=[
                {"role": "system", "content": "You are a senior domain expert who DOES all analysis and research yourself. Provide COMPLETE, READY-TO-USE data in every task — actual trends with numbers, actual strategies, actual content, actual titles. The user should NEVER need to Google anything. NEVER use words like Research, Find, Search, Identify, Explore, Investigate, Determine, Brainstorm in task titles or descriptions. CRITICAL: Every day in the plan MUST have completely DIFFERENT task types. NEVER repeat the same task pattern across days. Each day must deliver unique value and build on previous days. Output a single, valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        plan_data_str = response.choices[0].message.content
        plan_data = json.loads(plan_data_str)
        return plan_data
            
    except Exception as e:
        logger.error(f"Error during plan generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate a valid plan from AI.")


# --- Endpoints ---

@router.get("/get-goals")
async def get_goals_endpoint(user_id: str = Query(...)):
    """Retrieves all goals for a user, converting ObjectIds for serialization."""
    try:
        goals = await goals_collection.find({"user_id": user_id}).to_list(None)
        if not goals:
            return {"goals": []}
            
        # Convert necessary fields for JSON serialization
        for goal in goals:
            goal = convert_object_ids(goal)
            
            # Ensure datetime objects are converted to ISO format
            if isinstance(goal.get("created_at"), datetime):
                 goal["created_at"] = goal["created_at"].isoformat()
            if isinstance(goal.get("updated_at"), datetime):
                 goal["updated_at"] = goal["updated_at"].isoformat()
                 
            for task in goal.get("tasks", []):
                if isinstance(task.get("created_at"), datetime):
                    task["created_at"] = task["created_at"].isoformat()
                if isinstance(task.get("updated_at"), datetime):
                    task["updated_at"] = task["updated_at"].isoformat()
                if task.get("deadline") and isinstance(task["deadline"], datetime):
                    task["deadline"] = task["deadline"].strftime("%Y-%m-%d %H:%M") 
                
                for progress in task.get("progress", []):
                    if isinstance(progress.get("timestamp"), datetime):
                        progress["timestamp"] = progress["timestamp"].isoformat()

        return {"goals": goals}
    except Exception as e:
        logger.error(f"Error retrieving goals for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving goals.")


@router.post("/Plan_my_week")
async def plan_my_week_endpoint(request: PlanWeekRequest):
    """Generates a new strategic weekly plan based on active goals."""
    user_id = request.user_id
    try:
        # 1. Get user's timezone
        user_info = await users_collection.find_one({"user_id": user_id})
        tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
        try:
            user_tz = pytz.timezone(tz_name)
        except Exception:
            user_tz = pytz.UTC

        now_local = datetime.now(user_tz)
        today = now_local.date()
        start_of_week = today - timedelta(days=today.weekday())

        # 2. Fetch active and in-progress goals
        all_goals_data = await goals_collection.find({
            "user_id": user_id,
            "status": {"$in": ["active", "in progress"]}
        }).to_list(None)

        if not all_goals_data:
            raise HTTPException(status_code=404, detail="No active goals found to plan for.")
            
        # 3. Generate plan
        plan_data = await generate_weekly_plan(user_id, all_goals_data, user_tz)

        # 4. Save/Update the plan to MongoDB
        week_start_date_str = start_of_week.strftime("%Y-%m-%d")
        existing_plan = await weekly_plans_collection.find_one({
            "user_id": user_id,
            "week_start_date": week_start_date_str
        })

        if existing_plan:
            await weekly_plans_collection.update_one(
                {"_id": existing_plan["_id"]},
                {"$set": {"plan": plan_data, "updated_at": datetime.now(timezone.utc)}}
            )
            logger.info(f"Weekly plan updated for user {user_id} starting {week_start_date_str}.")
        else:
            await weekly_plans_collection.insert_one(
                {
                    "user_id": user_id,
                    "week_start_date": week_start_date_str,
                    "plan": plan_data,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
            )
            logger.info(f"New weekly plan created for user {user_id} starting {week_start_date_str}.")

        return plan_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /Plan_my_week endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/Update_my_week_plan")
async def update_my_week_plan_endpoint(request: PlanWeekRequest):
    """Regenerates/updates the current weekly plan based on the latest goal statuses."""
    user_id = request.user_id
    try:
        # 1. Get user's timezone
        user_info = await users_collection.find_one({"user_id": user_id})
        tz_name = user_info.get("time_zone", "UTC") if user_info else "UTC"
        try:
            user_tz = pytz.timezone(tz_name)
        except Exception:
            user_tz = pytz.UTC

        now_local = datetime.now(user_tz)
        today = now_local.date()
        start_of_week = today - timedelta(days=today.weekday())
        week_start_date_str = start_of_week.strftime("%Y-%m-%d")

        # 2. Check for existing plan
        existing_plan = await weekly_plans_collection.find_one({
            "user_id": user_id,
            "week_start_date": week_start_date_str
        })
        if not existing_plan:
            raise HTTPException(status_code=404, detail="No existing weekly plan found. Please create one first using /Plan_my_week.")

        # 3. Re-fetch all active/in-progress goals
        all_goals_data = await goals_collection.find({
            "user_id": user_id,
            "status": {"$in": ["active", "in progress"]}
        }).to_list(None)

        if not all_goals_data:
            # Plan exists, but all goals might be complete/deleted
            logger.info(f"User {user_id} requested plan update, but no active goals remain.")
            return {"message": "All goals completed. Weekly plan is now empty.", "updated_plan": {"strategic_approach": "All key objectives achieved for the week. Maintain momentum.", "this_week_plan": []}}


        # 4. Generate the updated plan
        plan_data = await generate_weekly_plan(user_id, all_goals_data, user_tz)

        # 5. Update the MongoDB weekly plan
        await weekly_plans_collection.update_one(
            {"_id": existing_plan["_id"]},
            {"$set": {"plan": plan_data, "updated_at": datetime.now(timezone.utc)}}
        )
        logger.info(f"Weekly plan successfully updated for user {user_id} starting {week_start_date_str}.")

        return {"message": "Weekly plan updated successfully.", "updated_plan": plan_data}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /Update_my_week_plan endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.websocket("/goal_setting")
async def goal_setting_endpoint(websocket: WebSocket):
    """Interactive WebSocket for conversational goal setting and task planning."""
    await websocket.accept()

    # Imports needed dynamically for the loop
    from services.ai_service import rate_limited_groq_call
    
    current_date = get_current_datetime()
    initial_data = await websocket.receive_json()
    user_id = initial_data.get("user_id")
    session_id = initial_data.get("session_id")
    async_client = AsyncGroq(api_key=GOAL_SETTING_KEY)

    if not user_id or not session_id:
        await websocket.send_json({"error": "Missing user_id or session_id"})
        await websocket.close()
        return

    # Initialize conversation state
    conversation_history = []
    goal_details = {}
    plan = None

    try:
        initial_prompt = "Hello! I'm here to help you set and achieve your goals. To start, what is the goal you have in mind?"
        await websocket.send_json({"message": initial_prompt})
        conversation_history.append({"role": "assistant", "content": initial_prompt})

        while True:
            user_input = await websocket.receive_text()
            conversation_history.append({"role": "user", "content": user_input})

            if plan and user_input.lower() == "confirm":
                break  # Exit loop to save the plan

            # Check if user is asking for help/resources for a specific task
            if plan and any(keyword in user_input.lower() for keyword in ["help", "first task", "how to", "tutorial", "video", "resource", "link", "guide", "start", "do", "complete"]):
                resource_prompt = f"""
                The user has a plan and is asking for help with a specific task.
                
                ⚠️ CRITICAL: YOU must do the work - don't just point to resources. PROVIDE:
                - Actual code snippets they can copy-paste
                - Exact terminal commands to run
                - Step-by-step instructions with specific actions
                - Ready-to-use templates or starter files
                
                User's request: {user_input}
                Current Plan: {json.dumps(plan, indent=2)}
                
                Provide a COMPLETE, EXECUTABLE response:
                1. The ACTUAL CODE or content they need to start (not just links to learn)
                2. Exact commands to run in terminal
                3. A few curated resources for deeper understanding (not for doing the work)
                4. Specific next actions they can take RIGHT NOW
                
                Return ONLY valid JSON format:
                {{
                  "message": "Brief encouraging message (1-2 sentences)",
                  "ready_to_use_content": {{
                    "code_snippets": [
                      {{
                        "title": "What this code does",
                        "language": "python/javascript/etc",
                        "code": "Actual working code they can copy-paste",
                        "explanation": "Brief explanation of how to use this code"
                      }}
                    ],
                    "terminal_commands": [
                      {{
                        "command": "Exact command to run",
                        "explanation": "What this command does"
                      }}
                    ],
                    "starter_template": "A ready-to-use template, configuration file, or project structure if applicable"
                  }},
                  "youtube_videos": [
                    {{
                      "title": "Video title",
                      "channel": "Channel name",
                      "duration": "Duration",
                      "description": "What they'll learn (for deeper understanding, not required to start)",
                      "search_query": "YouTube search query"
                    }}
                  ],
                  "websites": [
                    {{
                      "name": "Resource name",
                      "url": "Actual URL",
                      "description": "Reference documentation (for when they need to look things up)"
                    }}
                  ],
                  "immediate_actions": [
                    {{
                      "step": 1,
                      "action": "Exact action to take RIGHT NOW",
                      "time_estimate": "5 minutes"
                    }}
                  ]
                }}
                
                IMPORTANT: The user should be able to START WORKING immediately from your response without visiting any external links first.
                Current date/time: {current_date}
                """
                
                response = await rate_limited_groq_call(
                    async_client,
                    messages=[{"role": "system", "content": resource_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
                resource_data = json.loads(response.choices[0].message.content)
                
                await websocket.send_json({"resources": resource_data})
                conversation_history.append({"role": "assistant", "content": json.dumps(resource_data)})
                continue

            if not plan:
                # 1. Decision: Ask question or Generate plan?
                # --- Count how many assistant questions have been asked ---
                assistant_question_count = sum(
                    1 for m in conversation_history 
                    if m["role"] == "assistant" and not m["content"].startswith("{") and not m["content"].startswith("[")
                )
                # Detect if user is unsure/doesn't know
                unsure_phrases = ["no idea", "i don't know", "not sure", "idk", "no clue", "don't know", "i have no", "no preference", "you decide", "up to you", "whatever", "anything"]
                user_unsure_count = sum(
                    1 for m in conversation_history 
                    if m["role"] == "user" and any(phrase in m["content"].lower() for phrase in unsure_phrases)
                )
                # HARD LIMIT: Force plan generation after 4 questions OR 2 "no idea" responses
                force_generate = assistant_question_count >= 4 or user_unsure_count >= 2

                if force_generate:
                    # Skip asking, go straight to plan generation with smart defaults
                    goal_details["title"] = "Goal from conversation"
                    # Extract a better title from conversation
                    for m in conversation_history:
                        if m["role"] == "user" and len(m["content"]) > 15:
                            goal_details["title"] = m["content"][:80]
                            break
                else:
                    decision_prompt = f"""
                You are an expert goal coach - warm, encouraging, and strategically brilliant.
                
                Conversation History:
                {json.dumps(conversation_history, indent=2)}

                📊 CONVERSATION STATUS:
                - Questions already asked: {assistant_question_count}
                - Times user said "no idea" or similar: {user_unsure_count}
                - HARD LIMIT: You can ask AT MOST {max(0, 3 - assistant_question_count)} more questions
                
                🚨 CRITICAL RULES:
                1. NEVER ask more than 3 questions TOTAL across the entire conversation
                2. If user says "no idea", "I don't know", "all", or gives vague answers → DO NOT ask about that topic again. Use smart defaults instead.
                3. Ask MULTI-PART questions (combine 2-3 things in one question) to minimize back-and-forth
                4. If you already have: what they want + who it's for + budget OR timeline → GENERATE THE PLAN immediately
                5. You are NOT a survey. You are a helpful coach who fills in gaps with expertise.

                🧠 SMART DEFAULTS (use these when user doesn't know):
                - Budget unknown → Suggest a beginner-friendly budget range based on the domain
                - Target audience unknown → Infer from the goal type
                - KPIs unknown → Set standard industry benchmarks
                - Experience unknown → Assume beginner and provide extra guidance
                - Timeline unknown → Suggest a reasonable timeline based on the goal
                
                QUESTION STYLE (ask MULTI-PART, educational questions):
                ✅ "Great! Two quick things: 1) What's your monthly budget for this? (most beginners start with $5-10/day) and 2) Do you already have any ad creatives or landing pages ready?"
                ❌ Single-topic questions that drag out the conversation
                ❌ Asking the same thing the user already answered
                ❌ Asking for details the user clearly doesn't know

                Response format - valid JSON only:
                To generate plan: {{ "action": "generate_plan", "goal_title": "specific goal title" }}
                To ask question: {{ "action": "ask_question", "question": "warm, multi-part, educational question" }}
                
                Current date/time: {current_date}
                """
                
                    response = await rate_limited_groq_call(
                        async_client,
                        messages=[{"role": "system", "content": decision_prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.7,
                        response_format={"type": "json_object"},
                    )
                    decision = json.loads(response.choices[0].message.content)

                    if decision.get("action") == "ask_question":
                        question = decision.get("question", "Could you tell me more about that?")
                        await websocket.send_json({"message": question})
                        conversation_history.append({"role": "assistant", "content": question})
                        continue

                    if decision.get("action") == "generate_plan":
                        goal_details["title"] = decision.get("goal_title", "Untitled Goal")
            
            # 2. Generate or Revise the plan
            # --- Step 2a: Classify the goal type to select the right example ---
            goal_type = "general"
            conv_text = " ".join([m["content"].lower() for m in conversation_history])
            if any(kw in conv_text for kw in ["ads manager", "campaign", "ad campaign", "meta ads", "google ads", "brand awareness", "advertising", "marketing campaign", "facebook ads", "instagram ads", "ppc", "paid ads"]):
                goal_type = "marketing"
            elif any(kw in conv_text for kw in ["social media", "instagram", "tiktok", "youtube", "followers", "content creator", "influencer", "reels", "posting"]):
                goal_type = "social_media"
            elif any(kw in conv_text for kw in ["learn", "course", "python", "programming", "skill", "certification", "study", "education", "tutorial", "training"]):
                goal_type = "learning"
            elif any(kw in conv_text for kw in ["manufacturer", "supplier", "import", "export", "factory", "molding", "production", "industrial", "machine"]):
                goal_type = "manufacturing"
            elif any(kw in conv_text for kw in ["job", "career", "resume", "interview", "hiring", "employment", "apply", "position", "recruiter"]):
                goal_type = "job_search"
            
            # --- Step 2b: Build the domain-specific example ---
            domain_example = ""
            
            if goal_type == "marketing":
                domain_example = """
            DOMAIN: MARKETING/ADVERTISING CAMPAIGN
            
            🚨 CRITICAL - GENERATE EXACTLY 3 TASKS (not more, not less):
            Task 1: Campaign Setup & Audience Targeting (create campaign, set objective, configure pixel, define audiences, set interests/age/location)
            Task 2: Ad Creative, Copy & Budget (upload creatives, write copy, set budget allocation, configure bidding)
            Task 3: Launch, Learning Phase & Optimization (publish campaign, what to monitor during learning phase, when/how to optimize after)
            
            EACH TASK must have UNIQUE:
            - strategy_tips: Tips specific to THAT phase only (not generic "monitor performance")
            - best_practices: DO/DON'T specific to THAT task (not repeated across tasks)
            - common_mistakes: Mistakes relevant to THAT specific task only
            - success_metrics: Metrics relevant to THAT phase (setup metrics ≠ scaling metrics)
            - beginner_tips: Jargon relevant to THAT task only (don't repeat CPM definition in every task)
            - pro_tips: Advanced tips for THAT specific phase
            - example_content: Only for ad creative tasks (headlines, copy, CTAs). Other tasks get relevant templates instead.
            - expected_results: Results expected from THAT specific phase with realistic numbers
            
            SUB-TASK SPECIFICITY REQUIREMENTS:
            - NEVER say "Log in to Meta Ads Manager, click Create Campaign" — instead specify:
              * EXACT campaign objective to select and WHY
              * EXACT toggle/setting names (e.g., "Turn ON Advantage Campaign Budget", "Set optimization to Link Clicks")
              * EXACT audience size to aim for (e.g., "2M - 10M people")
              * EXACT interests to type in targeting (e.g., "Social media marketing, Hootsuite, Buffer, Later")
              * EXACT placements to select/deselect
              * EXACT ad format for each placement (1080x1080 for Feed, 1080x1920 for Stories)
            - Every sub-task URL must link to a SPECIFIC page, not just adsmanager.meta.com
            
            CURRENCY RULES:
            - If user mentions INR, rupees, or ₹ → use INR as primary currency
            - If user mentions $, USD, dollars → use USD
            - If budget is a round number like 50k without currency → ASK or show BOTH (INR and USD)
            - Always break budget into: daily spend, per-ad-set spend, testing budget, scaling budget
            
            KEY MARKETING RULES:
            - Provide 3+ ad copy variations (benefit-focused, problem-focused, social-proof focused)
            - Include 4+ headline options
            - Specify EXACT targeting: age range, location, interests, behaviors, exclusions
            - Explain: Awareness → Traffic → Conversion funnel with timeline
            - Warn about Learning Phase (3-7 days, don't touch anything)
            - Include specific audience suggestions based on the user's product/service
            """
            elif goal_type == "social_media":
                domain_example = """
            DOMAIN: SOCIAL MEDIA GROWTH
            Most social media tasks DON'T need email templates, documents, or free/paid tools!
            Only include email_templates for collaboration/outreach tasks.
            Only include free/paid options for scheduling/analytics tools.
            
            Required fields for each task:
            - strategy_tips, best_practices, optimal_times, example_content (captions, hashtags, bio templates)
            - why_this_matters, expected_results, common_mistakes, beginner_tips, pro_tips
            - success_metrics, next_phase, timeline_summary
            
            Profile tasks: strategy_tips + example_content + timeline_summary
            Content tasks: strategy_tips + best_practices + optimal_times + example_content  
            Collab tasks: strategy_tips + email_templates (DM templates) + timeline_summary
            Tool tasks: free_options + paid_options + total_cost_estimate
            """
            elif goal_type == "learning":
                domain_example = """
            DOMAIN: LEARNING/SKILL DEVELOPMENT
            Required fields for each task:
            - free_options: Free courses/platforms with specific URLs, what you get, limitations
            - paid_options: Paid courses with prices, what you get, recommended_for
            - youtube_tutorials: Specific video search queries with channel names and durations
            - why_this_matters, expected_results, common_mistakes, beginner_tips, pro_tips
            - total_cost_estimate: FREE PATH ($0) vs PAID PATH with specific amounts
            - timeline_summary: Week-by-week learning progression
            """
            elif goal_type == "manufacturing":
                domain_example = """
            DOMAIN: MANUFACTURING/BUSINESS
            CRITICAL: Provide COMPLETE company lists with real names, emails, websites, phone numbers.
            Required fields: paid_options (companies with full contact info), email_templates (ready-to-send),
            document_checklist (every document needed), total_cost_estimate, timeline_summary.
            Also include: why_this_matters, expected_results, common_mistakes, beginner_tips, pro_tips.
            NEVER say "find manufacturers" - YOU list them with complete contact info.
            """
            elif goal_type == "job_search":
                domain_example = """
            DOMAIN: JOB SEARCH/CAREER
            Required fields: free_options (job platforms), paid_options (specific employers with career page URLs),
            email_templates (cover letter and follow-up templates), youtube_tutorials (interview prep).
            Also include: why_this_matters, expected_results, common_mistakes, beginner_tips, pro_tips,
            pre_start_checklist, success_metrics, total_cost_estimate, timeline_summary.
            """
            else:
                domain_example = """
            DOMAIN: GENERAL
            Detect the goal type from conversation and include only RELEVANT sections.
            ALWAYS include: why_this_matters, expected_results, common_mistakes, beginner_tips, pro_tips,
            success_metrics, next_phase, pre_start_checklist, timeline_summary, total_cost_estimate.
            Only include email_templates, free/paid options, document_checklist if actually relevant.
            """
            
            # ============================================================
            # MULTI-CALL PLAN GENERATION
            # Each task gets its own API call for deeper, non-repetitive content
            # ============================================================

            SHARED_RULES = """
🚨 RULES:
1. NEVER use "Research/Find/Search/Look up/Gather/Identify/Explore" in steps — give DIRECT actions only
2. NEVER link to google.com — ONLY real direct URLs
3. If user said "no idea" → YOU decide with expert defaults
4. Assume beginner. Over-explain everything.
5. NEVER say "understand the importance of" or "learn about" — NOT tips
6. pro_tips = advanced expert tactics with numbers. strategy_tips = basic approach. They must be DIFFERENT.
7. expected_results must have SPECIFIC NUMBERS, not vague text
8. success_metrics format: "Metric: Good = X, Average = Y, Poor = Z"
9. beginner_tips format: "Term (Abbrev) = plain English definition"
10. Sub-task ai_content MUST be a single STRING (not an array). Format: "1. Go to [URL] 2. Click [button] 3. Select [option] ..."
11. example_content (ad copy, headlines, CTAs) belongs ONLY in creative/copy tasks. Campaign setup and optimization tasks should NOT have example_content.
12. KPI benchmarks must be CONSISTENT across all tasks — if CPC Good = ₹5 in one task, it must be ₹5 in all tasks.
13. Each task's ad copy/headlines/CTAs must be COMPLETELY DIFFERENT themes — no rewording the same message.

REAL URLs TO USE (for Meta/Facebook campaigns):
- Ads Manager: https://business.facebook.com/adsmanager/manage/campaigns
- Pixel/Events: https://business.facebook.com/events_manager/pixel
- Audiences: https://business.facebook.com/adsmanager/audiences
- Creative Hub: https://www.facebook.com/ads/creativehub
- Ad Library: https://www.facebook.com/ads/library
- Design tool: https://www.canva.com/create/facebook-ads/
"""

            if plan and user_input.lower() != "confirm":
                # === REVISION MODE: Single call to revise existing plan ===
                revision_prompt = f"""
                Revise this plan based on the user's feedback. Keep all existing fields and structure. Only change what the user asked for.
                
                Current plan:
                {json.dumps(plan, indent=2)}
                
                User's requested changes: {user_input}
                
                {SHARED_RULES}
                {domain_example}
                
                Return the COMPLETE revised plan as valid JSON:
                {{"plan": [... all 3 tasks with all fields ...]}}
                
                Current date/time: {current_date}
                """
                response = await rate_limited_groq_call(
                    async_client,
                    messages=[{"role": "system", "content": revision_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
                plan_data = json.loads(response.choices[0].message.content)
                plan = plan_data.get("plan", [])
            else:
                # === INITIAL GENERATION: Multi-call for deep content per task ===
                await websocket.send_json({"message": "🔄 Building your expert plan... generating detailed content for each phase."})
                
                # --- Call 1: Generate task skeleton (lightweight) ---
                conv_summary = json.dumps(conversation_history, indent=2)
                skeleton_prompt = f"""
Based on this conversation, create a 3-task action plan outline.

Conversation:
{conv_summary}

{domain_example}

Return ONLY valid JSON with EXACTLY 3 tasks:
{{
  "tasks": [
    {{"title": "Action Verb + Specific Task", "description": "One sentence measurable outcome", "deadline": "YYYY-MM-DD"}}
  ]
}}

Each task title must start with an action verb.
Current date/time: {current_date}
"""
                skeleton_response = await rate_limited_groq_call(
                    async_client,
                    messages=[{"role": "system", "content": skeleton_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )
                skeleton = json.loads(skeleton_response.choices[0].message.content)
                task_outlines = skeleton.get("tasks", [])[:3]
                
                # --- Calls 2-4: Generate each task in full detail ---
                plan = []
                covered_beginner_terms = []  # short keywords like "CPC", "CTR"
                covered_pro_tips = []        # short phrases like "Lookalike Audiences"
                covered_mistakes = []        # short phrases like "targeting too broad"
                covered_strategy_tips = []   # short phrases
                covered_metrics = []         # short phrases like "CPC Good=₹5"
                
                for idx, outline in enumerate(task_outlines):
                    # Build "already covered" context to prevent repetition
                    already_covered_section = ""
                    if covered_beginner_terms:
                        already_covered_section = f"""
⛔⛔⛔ MANDATORY — READ THIS FIRST ⛔⛔⛔

These EXACT terms/topics are BANNED in this task (already used in previous tasks):

🚫 BANNED beginner terms (do NOT define these again): {', '.join(covered_beginner_terms)}
🚫 BANNED pro tip topics (do NOT mention these): {', '.join(covered_pro_tips)}
🚫 BANNED mistake topics (do NOT warn about these): {', '.join(covered_mistakes)}
🚫 BANNED strategy topics (do NOT repeat these): {', '.join(covered_strategy_tips)}
🚫 BANNED metrics (use same numbers but different metric names): {', '.join(covered_metrics)}

If ANY of the above terms appear in your output, the response will be REJECTED.
You MUST define DIFFERENT jargon terms, give DIFFERENT tactical advice, warn about DIFFERENT mistakes.
"""
                    
                    is_later_phase = idx > 0
                    later_phase_note = ""
                    if is_later_phase:
                        later_phase_note = "Sub-tasks for this phase must describe EDITING/OPTIMIZING the existing campaign, NOT creating a new one."
                    
                    # Determine if this task should have example_content
                    task_title_lower = outline.get('title', '').lower()
                    has_creative = any(kw in task_title_lower for kw in ['creative', 'copy', 'ad copy', 'content', 'design'])
                    example_content_note = ""
                    if has_creative:
                        example_content_note = "This is a CREATIVE task — include example_content with 3+ unique ad_copy_variations, 4+ headlines, 3+ CTAs. Make them COMPLETELY different themes (benefit-focused, problem-focused, social-proof, urgency-based)."
                    else:
                        example_content_note = "This is NOT a creative task — do NOT include example_content. Focus on strategy, setup steps, and technical configuration."
                    
                    task_prompt = f"""
{already_covered_section}

Generate Task {idx + 1} of {len(task_outlines)} in FULL expert detail. This is ONE task — put ALL your depth into it.

Conversation context:
{conv_summary}

TASK TO GENERATE:
Title: {outline.get('title', '')}
Description: {outline.get('description', '')}
Deadline: {outline.get('deadline', '')}

{example_content_note}

{SHARED_RULES}
{domain_example}
{later_phase_note}

📊 MINIMUM CONTENT (this is ONE task so be THOROUGH):
- strategy_tips: 5+ specific instructions with numbers/thresholds
- best_practices: 5+ DO/DON'T with WHY explanation
- common_mistakes: 4+ using "MISTAKE: [specific] → FIX: [specific how-to]"
- beginner_tips: 4+ term definitions in "Term = plain English" format
- pro_tips: 4+ advanced tactical tips with specific numbers/thresholds
- sub_tasks: 4-5, each with 7+ click-by-click steps

🎯 ALL REQUIRED FIELDS in ai_provided_content:
strategy_tips, best_practices, why_this_matters, expected_results (what_to_expect + what_not_to_expect + kpi_benchmarks), common_mistakes, success_metrics, pre_start_checklist, beginner_tips, pro_tips, next_phase, total_cost_estimate, timeline_summary

Also include if relevant: example_content (ad_copy_variations, headlines, CTAs), optimal_times

Return valid JSON for this ONE task:
{{
  "title": "...",
  "description": "...",
  "deadline": "YYYY-MM-DD",
  "ai_provided_content": {{
    "strategy_tips": ["5+ tips with numbers"],
    "best_practices": ["5+ DO/DON'T with WHY"],
    "example_content": {{"ad_copy_variations": ["3+ variations"], "headlines": ["4+ options"], "CTAs": ["3+ options"]}},
    "optimal_times": "Best times with reasoning",
    "total_cost_estimate": "Budget breakdown in user's currency",
    "timeline_summary": "Day-by-day timeline for this phase",
    "why_this_matters": "WHY this task matters",
    "expected_results": {{
      "what_to_expect": ["Specific outcomes with NUMBERS"],
      "what_not_to_expect": ["Common misconceptions"],
      "kpi_benchmarks": ["Metric: Good = X, Average = Y, Poor = Z"]
    }},
    "common_mistakes": ["MISTAKE: ... → FIX: ..."],
    "success_metrics": ["Metric: Good = X, Average = Y, Poor = Z"],
    "next_phase": "What comes after this task",
    "pre_start_checklist": ["Items needed BEFORE starting"],
    "beginner_tips": ["Term = plain English definition"],
    "pro_tips": ["Advanced tactic with numbers"]
  }},
  "sub_tasks": [
    {{
      "title": "Specific action",
      "description": "One sentence outcome.",
      "ai_content": "7+ step click-by-click guide with exact URLs, button names, dropdown values, toggle states",
      "resources": {{
        "websites": [{{"name": "...", "url": "https://real-url.com/path", "description": "What to do here"}}]
      }}
    }}
  ]
}}

Current date/time: {current_date}
"""
                    # Rate limit buffer between calls
                    if idx > 0:
                        await asyncio.sleep(1.5)
                    
                    task_response = await rate_limited_groq_call(
                        async_client,
                        messages=[{"role": "system", "content": task_prompt}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.7,
                        response_format={"type": "json_object"},
                    )
                    task_data = json.loads(task_response.choices[0].message.content)
                    
                    # Handle case where model wraps in {"plan": [...]} or {"task": {...}}
                    if "plan" in task_data and isinstance(task_data["plan"], list):
                        task_data = task_data["plan"][0]
                    elif "task" in task_data and isinstance(task_data["task"], dict):
                        task_data = task_data["task"]
                    
                    plan.append(task_data)
                    
                    # Track what was covered — extract SHORT keywords for effective banning
                    ai_content = task_data.get("ai_provided_content", {})
                    
                    # Extract key terms from beginner_tips (e.g., "CPC", "CTR", "Ad Set")
                    for tip in ai_content.get("beginner_tips", []):
                        # Extract term name from patterns like "Term: CPC (Cost Per Click) = ..." or "CPC = ..."
                        term_match = re.match(r'^(?:Term:\s*)?([A-Za-z\s/]+?)(?:\s*\(|\s*=)', tip)
                        if term_match:
                            covered_beginner_terms.append(term_match.group(1).strip())
                        else:
                            covered_beginner_terms.append(tip[:30])
                    
                    # Extract short phrases from pro_tips
                    for tip in ai_content.get("pro_tips", []):
                        # Take first meaningful phrase (before numbers/details)
                        short = tip.split(',')[0].split('to ')[0] if ',' in tip or 'to ' in tip else tip[:50]
                        covered_pro_tips.append(short.strip()[:50])
                    
                    # Extract mistake topics
                    for m in ai_content.get("common_mistakes", []):
                        mistake_match = re.match(r'MISTAKE:\s*(.+?)\s*→', m)
                        if mistake_match:
                            covered_mistakes.append(mistake_match.group(1).strip()[:50])
                        else:
                            covered_mistakes.append(m[:40])
                    
                    # Extract strategy tip keywords
                    for tip in ai_content.get("strategy_tips", []):
                        covered_strategy_tips.append(tip.split(',')[0].strip()[:50])
                    
                    # Extract metric names
                    for m in ai_content.get("success_metrics", []):
                        metric_match = re.match(r'(?:Metric:\s*)?(.+?)(?:\s*[-,:]\s*Good)', m)
                        if metric_match:
                            covered_metrics.append(metric_match.group(1).strip())
                    
                    logger.info(f"Generated task {idx + 1}/{len(task_outlines)}: {task_data.get('title', 'unknown')}")

            await websocket.send_json(
                {
                    "message": "Here is a plan to get you started. Does this look right? You can ask for changes or type 'confirm' to save it.",
                    "plan": plan,
                }
            )
            conversation_history.append(
                {"role": "assistant", "content": json.dumps(plan)}
            )

        # 3. Save the confirmed goal and tasks
        goal_id = str(uuid.uuid4())
        new_goal = {
            "user_id": user_id,
            "goal_id": goal_id,
            "session_id": session_id,
            "title": goal_details.get("title", "Untitled Goal"),
            "description": "\n".join([msg["content"] for msg in conversation_history if msg["role"] == "user"]),
            "status": "active",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "tasks": [],
        }

        for task_item in plan:
            task_id = str(uuid.uuid4())
            
            # Process sub_tasks with complete AI content
            processed_sub_tasks = []
            for sub in task_item.get("sub_tasks", []):
                processed_sub_tasks.append({
                    "title": sub.get("title"),
                    "description": sub.get("description"),
                    "ai_content": sub.get("ai_content"),  # Step-by-step execution guide
                    "resources": sub.get("resources", {}),
                    "status": "not started"
                })
            
            new_task = {
                "task_id": task_id,
                "title": task_item.get("title"),
                "description": task_item.get("description"),
                "status": "not started",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "deadline": task_item.get("deadline"),
                "progress": [],
                # AI-provided intelligence content
                "ai_provided_content": task_item.get("ai_provided_content", {}),
                "sub_tasks": processed_sub_tasks,
            }
            new_goal["tasks"].append(new_task)

        await goals_collection.insert_one(new_goal)
        await websocket.send_json(
            {"message": "Great! Your goal has been saved. You can track your progress in the goals section."}
        )

        # 4. Save conversation to chat history with embeddings
        messages_to_save = []
        for msg in conversation_history:
            embedding = await generate_text_embedding(msg["content"])
            messages_to_save.append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "embedding": (embedding if embedding and len(embedding) == 768 else None),
                }
            )

        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        if chat_entry:
            await chats_collection.update_one(
                {"_id": chat_entry["_id"]},
                {"$push": {"messages": {"$each": messages_to_save}}},
            )
        else:
            await chats_collection.insert_one(
                {
                    "user_id": user_id, "session_id": session_id, "messages": messages_to_save,
                    "last_updated": datetime.now(timezone.utc),
                }
            )

    except WebSocketDisconnect:
        logger.info("Client disconnected from goal setting.")
    except Exception as e:
        logger.error(f"Error in goal setting endpoint: {e}")
        try:
            await websocket.send_json({"error": "An unexpected error occurred. Please try again."})
        except:
            pass
    finally:
        await websocket.close()
