# stelle_backend/routes/goal_routes.py
# stelle_backend/routes/goal_routes.py
import uuid
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional # <-- FIX 18-22: Missing typing imports (List, Dict, Any)
from groq import AsyncGroq # <-- FIX 23: Missing 'AsyncGroq'

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import pytz

from models.common_models import PlanWeekRequest
from database import goals_collection, users_collection, weekly_plans_collection, chats_collection
from services.common_utils import get_current_datetime, convert_object_ids, filter_think_messages
from services.ai_service import get_groq_client, generate_text_embedding # <-- FIX 24: Missing 'generate_text_embedding'
from config import logger, PLANNING_KEY, GOAL_SETTING_KEY

router = APIRouter()

# ... (rest of the file remains the same) ...

# --- Utility for Plan Generation ---

async def generate_weekly_plan(user_id: str, goals_data: List[Dict[str, Any]], user_tz: pytz.BaseTzInfo) -> Dict[str, Any]:
    """Generates the strategic weekly plan using the LLM."""
    
    now_local = datetime.now(user_tz)
    today = now_local.date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    
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
You are an expert project manager and personal coach. Your task is to create a comprehensive and strategic weekly plan for a user based on their goals.

Current Date: {today.strftime('%Y-%m-%d')}
Current Calendar Week: {start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}

Here is the complete context of the user's current goals and tasks:
---
{full_context}
---

Please generate a plan with the following structure in a single, valid JSON object:
{{
  "strategic_approach": "A high-level overview for the upcoming weeks until the final goal deadline.",
  "this_week_plan": [
    {{
      "date": "YYYY-MM-DD",
      "day_of_week": "Monday",
      "daily_focus": "Main objective for the day.",
      "tasks": [
        {{
          "task_id": "The original task ID.",
          "title": "The original task title.",
          "status": "The current status of the task.",
          "description": "Detailed, actionable guide on HOW to complete the task.",
          "sub_tasks": [
            {{
              "title": "A smaller, actionable sub-task.",
              "description": "Detailed 'how-to' guide for this sub-task."
            }}
          ]
        }}
      ]
    }}
  ]
}}

**CRITICAL RULES**:
1. Prioritize Overdue Tasks: Any task marked "Deadline Exceeded" MUST be scheduled for the first day of the plan.
2. Detailed 'How-To' Descriptions: Provide specific, step-by-step guidance for tasks and sub-tasks.
3. Logical Sequencing: Schedule tasks based on dependencies and logical workflow.
4. Full Week Plan: Generate a plan for all 7 days (Monday to Sunday). Use empty lists for days with no tasks.
5. Output ONLY JSON: Return a single, valid JSON object.

Now, create the weekly plan.
"""
    # Call the AI
    try:
        from services.ai_service import rate_limited_groq_call
        async_client = AsyncGroq(api_key=PLANNING_KEY)
        response = await rate_limited_groq_call(
            async_client,
            messages=[
                {"role": "system", "content": "You are an expert AI project manager. Output a single, valid JSON object."},
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

            if not plan:
                # 1. Decision: Ask question or Generate plan?
                decision_prompt = f"""
                You are an expert goal-setting assistant. Your task is to have a friendly conversation to help a user define their goal.
                
                Conversation History:
                {json.dumps(conversation_history, indent=2)}

                If the user has clearly stated their goal and you have enough information (what, why, how, when), respond with a JSON object to generate a plan:
                {{ "action": "generate_plan", "goal_title": "The user's goal title" }}

                Otherwise, ask a friendly, specific question to get more details. The question should guide the user to provide information related to the SMART framework (Specific, Measurable, Achievable, Relevant, Time-bound) without explicitly mentioning it. Respond with a JSON object:
                {{ "action": "ask_question", "question": "Your friendly, contextual question" }}
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
            plan_generation_prompt = f"""
            Based on the following conversation, generate a detailed and actionable plan for the user's goal.
            
            Conversation:
            {json.dumps(conversation_history, indent=2)}
            
            Your task is to:
            1. Create a list of small, manageable tasks.
            2. For each task, provide a clear 'title' (the what) and a 'description' (the how). The description should give the user practical steps.
            3. Suggest a 'deadline' for each task in 'YYYY-MM-DD' format.
            4. The output must be a JSON object with a single key "plan" which is a list of task objects.
            
            Example of a task object:
            {{ "title": "Research local gyms", "description": "Use online maps...", "deadline": "2024-07-30" }}

            Now, generate the plan based on the conversation.
            Current date/time: {current_date}
            """
            if plan and user_input.lower() != "confirm":
                plan_generation_prompt += f"\nThe user has requested the following adjustments: {user_input}. Please revise the plan accordingly."
                
            response = await rate_limited_groq_call(
                async_client,
                messages=[{"role": "system", "content": plan_generation_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            plan_data = json.loads(response.choices[0].message.content)
            plan = plan_data.get("plan", [])

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
            new_task = {
                "task_id": task_id,
                "title": task_item.get("title"),
                "description": task_item.get("description"),
                "status": "not started",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "deadline": task_item.get("deadline"),
                "progress": [],
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