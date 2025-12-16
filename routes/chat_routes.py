# stelle_backend/routes/chat_routes.py
import asyncio
import json
import re
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Union, Tuple
from groq import Groq, AsyncGroq
import uuid
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query, WebSocket,WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np

from models.common_models import (
    GenerateRequest,
    GenerateResponse,
    RegenerateRequest,
    NLPRequest,
    UserInput,
)
from database import chats_collection, memory_collection, uploads_collection, goals_collection
from services.ai_service import (
    get_groq_client,
    query_internet_via_groq,
    retrieve_multimodal_context,
    content_for_website,
    detailed_explanation,
    classify_prompt,
    generate_text_embedding,
    store_long_term_memory,
    synthesize_result,
    # newly used helpers (must exist in ai_service.py)
    query_deepsearch,
    visualize_content,
)
from services.goal_service import update_task_goal_status, schedule_immediate_reminder
from services.common_utils import get_current_datetime, filter_think_messages, convert_object_ids
from config import logger, GENERATE_API_KEYS
from database import doc_index, code_index, file_doc_memory_map, code_memory_map

router = APIRouter(tags=["Chat"])

visualize_queries = {}
visualize_jobs = {}
deepsearch_queries = {}
deepsearch_jobs={}
# -------------------------
# Helper: normalize platform names and detect chosen platform field
# -------------------------

POSSIBLE_PLATFORM_FIELDS = ["platform", "default_platform", "selected_platform", "social_platform"]


def extract_chosen_platform_from_input(data: Any) -> Union[str, None]:
    """
    Accepts either a Pydantic object (with attributes) or a dict.
    Looks for any of POSSIBLE_PLATFORM_FIELDS and returns its value (lowercased) if found.
    """
    try:
        obj_dict = data if isinstance(data, dict) else getattr(data, "__dict__", None) or {}
    except Exception:
        obj_dict = {}

    for field in POSSIBLE_PLATFORM_FIELDS:
        if isinstance(obj_dict, dict) and field in obj_dict and obj_dict[field]:
            return str(obj_dict[field]).strip().lower()
        try:
            if hasattr(data, field):
                val = getattr(data, field)
                if val:
                    return str(val).strip().lower()
        except Exception:
            pass
    return None


def normalize_platform_key(name: str) -> str:
    """Normalize various platform names to canonical lowercase keys."""
    if not name:
        return ""
    n = name.strip().lower()
    mapping = {
        "x": "twitter",
        "twitter": "twitter",
        "insta": "instagram",
        "instagram": "instagram",
        "linkedin": "linkedin",
        "facebook": "facebook",
        "pinterest": "pinterest",
        "threads": "threads",
        "tiktok": "tiktok",
        "youtube": "youtube",
        "yt": "youtube",
        "thread": "threads",
    }
    return mapping.get(n, n)


# -------------------------
# Helper: safe platform list extraction (from input_data or JSON)
# -------------------------
def extract_platforms_list(data: Any) -> List[str]:
    """
    Looks for 'platforms' in input (list or comma-separated string).
    If not found returns default ['instagram'].
    """
    platforms = []
    if isinstance(data, dict):
        platforms_val = data.get("platforms") or data.get("platform_list") or data.get("platformOptions") or None
    else:
        platforms_val = None
        for attr in ["platforms", "platform_list", "platformOptions", "platforms_list"]:
            if hasattr(data, attr):
                try:
                    platforms_val = getattr(data, attr)
                    break
                except Exception:
                    platforms_val = None

    if platforms_val:
        if isinstance(platforms_val, str):
            raw = [p.strip() for p in re.split(r"[,\n]+", platforms_val) if p.strip()]
            platforms = [normalize_platform_key(p) for p in raw]
        elif isinstance(platforms_val, (list, tuple, set)):
            platforms = [normalize_platform_key(str(p)) for p in platforms_val if p]
    if not platforms:
        platforms = ["instagram"]
    seen = set()
    out = []
    for p in platforms:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ----------------------------------------
# Utility: Convert Platforms Enum (if needed)
# ----------------------------------------
def platform_strings_to_enum_list(platform_strings: List[str]):
    """Attempt to import Platforms enum from post_generator_service and convert; fallback to returning raw strings."""
    try:
        from services.post_generator_service import Platforms

        enums = []
        for p in platform_strings:
            name = p.strip().lower()
            mapping = {
                "instagram": "Instagram",
                "x": "X",
                "twitter": "X",
                "reddit": "Reddit",
                "linkedin": "LinkedIn",
                "facebook": "Facebook",
                "pinterest": "Instagram",
                "threads": "Instagram",
                "tiktok": "Instagram",
                "youtube": "Instagram",
            }
            member_name = mapping.get(name, None)
            if member_name and hasattr(Platforms, member_name):
                enums.append(getattr(Platforms, member_name))
            else:
                found = False
                for member in Platforms:
                    if member.name.lower() == name:
                        enums.append(member)
                        found = True
                        break
                if not found:
                    enums.append(list(Platforms)[0])
        return enums
    except Exception:
        return platform_strings


# ---------------------------------------------------------------------
# Helper functions for RAG cleanup, goal parsing, and persistence
# ---------------------------------------------------------------------


async def update_rag_usage_and_cleanup(user_id: str, session_id: str):
    """Increments RAG chunk usage counts and removes expired chunks from DB and FAISS."""
    try:
        cursor = uploads_collection.find(
            {
                "user_id": user_id,
                "session_id": session_id,
                "query_count": {"$gte": 15},
            }
        )
        documents_to_remove = set()
        async for chunk in cursor:
            documents_to_remove.add(chunk["filename"])

        for filename in documents_to_remove:
            await uploads_collection.delete_many({"user_id": user_id, "session_id": session_id, "filename": filename})
            logger.info(f"Cleaned up MongoDB uploads for expired file: {filename}")

            indices_to_remove = [
                idx for idx, m in file_doc_memory_map.items() if m.get("filename") == filename and m.get("session_id") == session_id
            ]
            if indices_to_remove:
                try:
                    doc_index.remove_ids(np.array(indices_to_remove, dtype="int64"))
                except Exception:
                    logger.warning("doc_index.remove_ids failed or unsupported; skipping index removal.")
                for idx in indices_to_remove:
                    if idx in file_doc_memory_map:
                        del file_doc_memory_map[idx]
                logger.info(f"Cleaned up {len(indices_to_remove)} FAISS doc chunks for {filename}.")

            code_indices_to_remove = [
                idx for idx, m in code_memory_map.items() if m.get("filename") == filename and m.get("session_id") == session_id
            ]
            if code_indices_to_remove:
                try:
                    code_index.remove_ids(np.array(code_indices_to_remove, dtype="int64"))
                except Exception:
                    logger.warning("code_index.remove_ids failed or unsupported; skipping index removal.")
                for idx in code_indices_to_remove:
                    if idx in code_memory_map:
                        del code_memory_map[idx]
                logger.info(f"Cleaned up {len(code_indices_to_remove)} FAISS code chunks for {filename}.")

        logger.info("RAG cleanup complete.")
    except Exception as e:
        logger.error(f"Error in update_rag_usage_and_cleanup: {e}")


async def handle_goal_updates_and_cleanup(reply_content: str, user_id: str, session_id: str):
    """Parses LLM output for goal/task commands and applies database changes."""
    try:
        new_goals_map = {}
        goal_set_matches = re.findall(r"\[GOAL_SET: (.*?)\]", reply_content)
        for goal_phrase in goal_set_matches:
            await update_task_goal_status(user_id, session_id, goal_phrase, "set_goal", None, None, new_goals_map)

        task_matches = re.findall(r"\[TASK: (.*?)\]", reply_content)
        for task_desc in task_matches:
            if new_goals_map:
                goal_id_placeholder = next(iter(new_goals_map.keys()))
                goal_id = new_goals_map.get(goal_id_placeholder, goal_id_placeholder)
                await update_task_goal_status(user_id, session_id, task_desc, "add_task", goal_id, None, new_goals_map)

        task_add_matches = re.findall(r"\[TASK_ADD:\s*(.*?):\s*(.*?)\]", reply_content)
        for goal_id_str, task_desc in task_add_matches:
            await update_task_goal_status(user_id, session_id, task_desc, "add_task", goal_id_str, None, new_goals_map)

        commands = {
            "GOAL_DELETE": re.findall(r"\[GOAL_DELETE: (.*?)\]", reply_content),
            "TASK_DELETE": re.findall(r"\[TASK_DELETE: (.*?)\]", reply_content),
            "TASK_MODIFY": re.findall(r"\[TASK_MODIFY:\s*(.*?):\s*(.*?)\]", reply_content),
            "GOAL_START": re.findall(r"\[GOAL_START: (.*?)\]", reply_content),
            "TASK_START": re.findall(r"\[TASK_START: (.*?)\]", reply_content),
            "GOAL_COMPLETE": re.findall(r"\[GOAL_COMPLETE: (.*?)\]", reply_content),
            "TASK_COMPLETE": re.findall(r"\[TASK_COMPLETE: (.*?)\]", reply_content),
        }

        for command, matches in commands.items():
            if command in ["TASK_MODIFY"]:
                for tid, new_desc in matches:
                    await update_task_goal_status(user_id, session_id, new_desc, command, None, tid, new_goals_map)
            elif command in ["GOAL_START", "GOAL_COMPLETE", "GOAL_DELETE"]:
                for gid in matches:
                    await update_task_goal_status(user_id, session_id, None, command, gid, None, new_goals_map)
            elif command in ["TASK_START", "TASK_COMPLETE", "TASK_DELETE"]:
                for tid in matches:
                    await update_task_goal_status(user_id, session_id, None, command, None, tid, new_goals_map)

        task_deadline_matches = re.findall(r"\[TASK_DEADLINE:\s*(.*?):\s*(.*?)\]", reply_content)
        for tid, deadline_str in task_deadline_matches:
            await update_task_goal_status(user_id, session_id, deadline_str, "TASK_DEADLINE", None, tid, new_goals_map)

        task_progress_matches = re.findall(r"\[TASK_PROGRESS:\s*(.*?):\s*(.*?)\]", reply_content)
        for tid, progress_desc in task_progress_matches:
            await update_task_goal_status(user_id, session_id, progress_desc, "TASK_PROGRESS", None, tid, new_goals_map)

        await update_rag_usage_and_cleanup(user_id, session_id)
    except Exception as e:
        logger.error(f"Error in handle_goal_updates_and_cleanup: {e}")


# --- Endpoints ---


@router.post("/generate", response_model=GenerateResponse)
async def generate_response_endpoint(request: Request, background_tasks: BackgroundTasks):
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = GenerateRequest(**data)
        user_id, session_id, user_message, filenames = req.user_id, req.session_id, req.prompt, req.filenames
        if not user_id or not session_id or not user_message:
            raise HTTPException(status_code=400, detail="Invalid request parameters.")

        # --- 1. Gather Context (Goals, Files, URLs, Memory) ---

        # a) Goal Context
        active_goals = await goals_collection.find({"user_id": user_id, "status": {"$in": ["active", "in progress"]}}).to_list(None)
        goals_context = ""
        if active_goals:
            goals_context = "User's current goals and tasks:\n"
            for goal in active_goals:
                goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal.get('goal_id','N/A')}]\n"
                for task in goal.get("tasks", []):
                    goals_context += f"  - Task: {task['title']} ({task['status']}) [ID: {task.get('task_id','N/A')}]\n"

        # b) File Mention Context
        uploaded_files = await uploads_collection.distinct("filename", {"session_id": session_id})
        mentioned_filenames = [fn for fn in uploaded_files if fn.lower() in user_message.lower()]
        hooked_filenames = filenames if filenames else mentioned_filenames

        # c) URL/External Content Context
        external_content = ""
        url_match = re.search(r"https?://[^\s]+", user_message)
        if url_match:
            url = url_match.group(0)
            if "youtube.com" in url or "youtu.be" in url:
                summary = await query_internet_via_groq(f"Summarize the content of the YouTube video at {url}")
                external_content = await detailed_explanation(summary)
            else:
                summary = await query_internet_via_groq(f"Summarize the content of the webpage at {url}")
                external_content = await content_for_website(summary)

        # d) Multimodal RAG Context
        multimodal_context, used_filenames = await retrieve_multimodal_context(user_message, session_id, hooked_filenames)

        # --- 2. Construct Unified Prompt ---
        unified_prompt = f"User Query: {user_message}\n"
        if external_content:
            unified_prompt += f"\n[External Content]:\n{external_content}\n"
        if multimodal_context:
            unified_prompt += f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
        unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a detailed and context-aware response."

        # e) Dynamic Research
        research_needed = await classify_prompt(user_message)
        if research_needed == "research" and not multimodal_context:
            # Prefer DeepSearch -> fallback to internet/groq
            try:
                ds_content, ds_sources = await query_deepsearch(user_message)
                if ds_content and "Error" not in ds_content and "unavailable" not in ds_content:
                    unified_prompt += f"\n\n[DeepSearch Results]:\n{ds_content}\n"
                    if ds_sources:
                        unified_prompt += "\nSources:\n" + "\n".join([f"- {s['title']}: {s['url']}" for s in ds_sources])
                else:
                    research_results = await query_internet_via_groq(user_message)
                    if research_results and research_results != "Error accessing internet information.":
                        unified_prompt += f"\n\n[Additional Research]:\n{research_results}"
            except Exception as e:
                logger.warning(f"DeepSearch failed in /generate, falling back: {e}")
                research_results = await query_internet_via_groq(user_message)
                if research_results and research_results != "Error accessing internet information.":
                    unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

        # Visualization integration: if there's multimodal content and user asked for visual or analysis
        if multimodal_context and any(k in user_message.lower() for k in ("visual", "visualize", "analyze", "analysis", "insight")):
            try:
                viz = await visualize_content(multimodal_context)
                # Attach a short summary into the LLM prompt so it can reference visualization findings
                unified_prompt += (
                    f"\n\n[Visualization Summary]:\nSummary: {viz.get('summary','')}\n"
                    f"Themes: {', '.join(viz.get('themes',[]))}\n"
                )
            except Exception as e:
                logger.warning(f"visualize_content failed in /generate: {e}")

        # --- 3. Build Message History (Retrieval Augmented) ---
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        past_messages = chat_entry.get("messages", []) if chat_entry else []
        chat_history = []

        if past_messages:
            past_embeddings = [
                msg["embedding"] for msg in past_messages if "embedding" in msg and isinstance(msg["embedding"], list) and len(msg["embedding"]) == 768
            ]
            if past_embeddings:
                past_embeddings_np = np.array(past_embeddings)
                current_embedding = await generate_text_embedding(user_message)

                if current_embedding and len(current_embedding) == 768:
                    distances = np.linalg.norm(past_embeddings_np - np.array(current_embedding), axis=1)
                    n = len(past_messages)
                    ages = np.array([n - 1 - i for i in range(n)])
                    lambda_val = 0.05
                    modified_distances = distances + lambda_val * ages
                    k = 7
                    top_k_indices = np.argsort(modified_distances)[:k]
                    m = 5
                    last_m_indices = list(range(n - m, n)) if n >= m else list(range(n))
                    combined_indices = sorted(list(set(top_k_indices.tolist() + last_m_indices)))
                    chat_history = [past_messages[i] for i in combined_indices]
                else:
                    chat_history = filter_think_messages(past_messages[-5:])
            else:
                chat_history = filter_think_messages(past_messages[-5:])

        # Long-term memory
        long_term_memory = await memory_collection.find_one({"user_id": user_id})
        long_term_memory_summary = long_term_memory.get("summary", "") if long_term_memory else ""

        system_prompt = (
            "You are Stelle, a strategic, empathetic AI assistant with autonomous goal/task management. Remember to speak like a chatbot if the user addresses you as such and just output normal chat no other things . "
            "If you have to add tasks to a goal, beforehand make the task id then add it to the goal. "
            "When the user sets a new goal, use '[GOAL_SET: <goal_title>]' Must use '[TASK: <task_desc>]' lines. for adding tasks. "
            "To delete a goal: '[GOAL_DELETE: <goal_id>]'. To delete a task: '[TASK_DELETE: <task_id>]'. "
            "To add a new task: '[TASK_ADD: <goal_id>: <task_description>]'. "
            "To modify a task's title: '[TASK_MODIFY: <task_id>: <new_title_or_description>]'. "
            "To start a goal: '[GOAL_START: <goal_id>]'. To start a task: '[TASK_START: <task_id>]'. "
            "To complete a goal: '[GOAL_COMPLETE: <goal_id>]'. To complete a task: '[TASK_COMPLETE: <task_id>]'. "
            "Must ask user for deadlines using '[TASK_DEADLINE: <task_id>: <YYYY-MM-DD HH:MM>]' and log progress using '[TASK_PROGRESS: <task_id>: <progress_description>]'.\n"
            f"Current date/time: {current_date}\n"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if long_term_memory_summary:
            messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory_summary}"})
        if goals_context:
            messages.append({"role": "system", "content": goals_context})

        for msg in chat_history:
            cleaned = re.sub(r"<think>.*?</think>", "", msg.get("content", ""), flags=re.DOTALL).strip()
            if cleaned:
                cleaned = cleaned[:800] + "…" if len(cleaned) > 800 else cleaned
                messages.append({"role": msg["role"], "content": cleaned})

        messages.append({"role": "user", "content": unified_prompt})
        logger.info(f"LLM prompt messages prepared, count: {len(messages)}")

        # --- 4. Stream Response and Handle Persistence ---
        selected_key = random.choice(GENERATE_API_KEYS)
        client_generate = AsyncGroq(api_key=selected_key)

        stream = await client_generate.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            max_completion_tokens=4000,
            temperature=0.7,
            stream=True,
        )

        async def generate_stream():
            full_reply = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                full_reply += delta
                yield delta

            # --- Persistence and Background Tasks after stream ends ---
            reply_content = full_reply.strip()

            # a) Process LLM commands (Goal/Task updates)
            await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)

            # b) Clean reply content for storage
            lines = reply_content.split("\n")
            clean_lines = [line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())]
            reply_content_clean = "\n".join(clean_lines).strip()

            # c) Generate embeddings and save to chat history
            user_embedding = await generate_text_embedding(user_message)
            assistant_embedding = await generate_text_embedding(reply_content_clean)

            new_messages = [
                {"role": "user", "content": user_message, **({"embedding": user_embedding} if user_embedding else {})},
                {"role": "assistant", "content": reply_content_clean, **({"embedding": assistant_embedding} if assistant_embedding else {})},
            ]

            update_fields = {
                "$push": {"messages": {"$each": new_messages}},
                "$set": {"last_updated": datetime.now(timezone.utc)},
            }
            if chat_entry:
                await chats_collection.update_one({"_id": chat_entry["_id"]}, update_fields)
                updated_messages = chat_entry.get("messages", []) + new_messages
            else:
                new_chat_entry = {"user_id": user_id, "session_id": session_id, "messages": new_messages, "last_updated": datetime.now(timezone.utc)}
                await chats_collection.insert_one(new_chat_entry)
                updated_messages = new_messages

            # d) Update Long-Term Memory (in background if chat is long)
            if len(updated_messages) >= 10:
                background_tasks.add_task(
                    store_long_term_memory, user_id, session_id, updated_messages[-10:]
                )

            # e) Update RAG usage count for files actually used in this query
            for filename in used_filenames:
                await uploads_collection.update_many(
                    {"user_id": user_id, "session_id": session_id, "filename": filename},
                    {"$inc": {"query_count": 1}},
                )

        return StreamingResponse(generate_stream(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error processing your request: {e}")


@router.post("/regenerate", response_model=GenerateResponse)
async def regenerate_response_endpoint(request: RegenerateRequest, background_tasks: BackgroundTasks):
    """
    Regenerates the last assistant response based on the last user message.
    Clean, safe rewritten version (Option B).
    """
    try:
        user_id, session_id, filenames = request.user_id, request.session_id, request.filenames

        # ---- Fetch chat entry ----
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        if not chat_entry or not chat_entry.get("messages"):
            raise HTTPException(status_code=400, detail="No chat history found for regeneration.")

        messages_list = chat_entry["messages"]

        # ---- Find last user message ----
        last_user = None
        for msg in reversed(messages_list):
            if msg.get("role") == "user":
                last_user = msg
                break

        if not last_user:
            raise HTTPException(status_code=400, detail="No user message found to regenerate from.")

        user_message = last_user["content"]

        # ---- Remove last assistant response (if exists) ----
        last_assistant_idx = None
        for i in range(len(messages_list) - 1, -1, -1):
            if messages_list[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is not None:
            await chats_collection.update_one(
                {"_id": chat_entry["_id"]},
                {"$unset": {f"messages.{last_assistant_idx}": ""}},
            )
            await chats_collection.update_one(
                {"_id": chat_entry["_id"]},
                {"$pull": {"messages": None}},
            )
            messages_list.pop(last_assistant_idx)

        # ---- Prepare context ----
        current_date = get_current_datetime()

        long_term_memory = await memory_collection.find_one({"user_id": user_id})
        long_term_memory_summary = long_term_memory.get("summary", "") if long_term_memory else ""

        # ---- System prompt ----
        system_prompt = (
            "You are Stelle, an intelligent assistant. "
            "Regenerate the assistant response based ONLY on the last user message "
            "and context provided. Keep goal/task consistency.\n"
            f"Current date/time: {current_date}\n"
        )

        # ---- Build messages for model ----
        final_messages = [{"role": "system", "content": system_prompt}]

        if long_term_memory_summary:
            final_messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory_summary}"})

        # Add the last few cleaned messages (history)
        cleaned_history = filter_think_messages(messages_list[-4:])
        for m in cleaned_history:
            clean_content = re.sub(r"<think>.*?</think>", "", m["content"], flags=re.DOTALL).strip()
            clean_content = clean_content[:800] + "…" if len(clean_content) > 800 else clean_content
            final_messages.append({"role": m["role"], "content": clean_content})

        # Add the user message that triggered regeneration
        final_messages.append({"role": "user", "content": user_message})

        # ---- Generate new response ----
        selected_key = random.choice(GENERATE_API_KEYS)
        client_generate = AsyncGroq(api_key=selected_key)

        stream = await client_generate.chat.completions.create(
            messages=final_messages,
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            max_completion_tokens=4000,
            temperature=0.7,
            stream=True,
        )

        async def regeneration_stream():
            full_reply = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_reply += delta
                yield delta

            reply_content = full_reply.strip()

            # ---- Post-processing ----
            await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)

            # Remove goal/task commands for storage
            lines = reply_content.split("\n")
            clean_lines = [L for L in lines if not re.match(r"\[.*?: .*?\]", L.strip())]
            cleaned_reply = "\n".join(clean_lines).strip()

            # ---- Save regenerated reply ----
            assistant_embedding = await generate_text_embedding(cleaned_reply)

            new_msg = {
                "role": "assistant",
                "content": cleaned_reply,
                "embedding": assistant_embedding,
            }

            await chats_collection.update_one(
                {"user_id": user_id, "session_id": session_id},
                {
                    "$push": {"messages": new_msg},
                    "$set": {"last_updated": datetime.now(timezone.utc)},
                },
            )

            # Update long-term memory after enough messages
            updated_chat = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
            if updated_chat and len(updated_chat.get("messages", [])) >= 10:
                background_tasks.add_task(store_long_term_memory, user_id, session_id, updated_chat["messages"][-10:])

            # Cleanup RAG chunks
            await update_rag_usage_and_cleanup(user_id, session_id)

        return StreamingResponse(regeneration_stream(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in /regenerate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error during regeneration: {e}")


# --- NLP/Voice Endpoint (WebSocket) ---


@router.websocket("/nlp")
async def nlp_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time NLP/Voice chat."""
    await websocket.accept()

    from services.goal_service import update_task_goal_status, schedule_immediate_reminder
    from services.common_utils import get_current_datetime, filter_think_messages

    try:
        while True:
            data = await websocket.receive_text()
            nlp_data = json.loads(data)
            req = NLPRequest(**nlp_data)
            user_id, session_id, user_message, filenames = req.user_id, req.session_id, req.message, req.filenames

            if not user_id or not session_id or not user_message:
                await websocket.send_json({"error": "Invalid request parameters"})
                continue

            current_date = get_current_datetime()

            # --- Gather Context (Similar to /generate) ---

            active_goals = await goals_collection.find({"user_id": user_id, "status": {"$in": ["active", "in progress"]}}).to_list(None)
            goals_context = ""
            if active_goals:
                goals_context = "User's current goals and tasks:\n"
                for goal in active_goals:
                    goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal.get('goal_id','N/A')}]\n"
                    for task in goal.get("tasks", []):
                        goals_context += f"  - Task: {task['title']} ({task['status']}) [ID: {task.get('task_id','N/A')}]\n"

            uploaded_files = await uploads_collection.distinct("filename", {"session_id": session_id})
            mentioned_filenames = [fn for fn in uploaded_files if fn.lower() in user_message.lower()]
            hooked_filenames = filenames if filenames else mentioned_filenames

            external_content = ""
            url_match = re.search(r"https?://[^\s]+", user_message)
            if url_match:
                url = url_match.group(0)
                if "youtube.com" in url or "youtu.be" in url:
                    summary = await query_internet_via_groq(f"Summarize the content of the YouTube video at {url}")
                    external_content = await detailed_explanation(summary)
                else:
                    summary = await query_internet_via_groq(f"Summarize the content of the webpage at {url}")
                    external_content = await content_for_website(summary)

            multimodal_context, used_filenames = await retrieve_multimodal_context(user_message, session_id, hooked_filenames)

            # --- 2. Construct Unified Prompt (NLP-specific tone) ---
            unified_prompt = f"User Query: {user_message}\n"
            if external_content:
                unified_prompt += f"\n[External Content]:\n{external_content}\n"
            if multimodal_context:
                unified_prompt += f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
            unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a conversational, friendly response as if speaking directly to the user."

            research_needed = await classify_prompt(user_message)
            if research_needed == "research" and not multimodal_context:
                # Try DeepSearch first, then fallback to query_internet_via_groq
                try:
                    await websocket.send_json({"status": "researching", "message": "Deep-searching the topic..."})
                    ds_content, ds_sources = await query_deepsearch(user_message)
                    if ds_content and "Error" not in ds_content and "unavailable" not in ds_content:
                        unified_prompt += f"\n\n[DeepSearch Results]:\n{ds_content}\n"
                        if ds_sources:
                            unified_prompt += "\nSources:\n" + "\n".join([f"- {s['title']}: {s['url']}" for s in ds_sources])
                    else:
                        research_results = await query_internet_via_groq(user_message)
                        if research_results and research_results != "Error accessing internet information.":
                            unified_prompt += f"\n\n[Additional Research]:\n{research_results}"
                except Exception as e:
                    logger.warning(f"DeepSearch failed in /nlp, falling back: {e}")
                    research_results = await query_internet_via_groq(user_message)
                    if research_results and research_results != "Error accessing internet information.":
                        unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

            # Visualization integration for multimodal content in websocket mode
            if multimodal_context and any(k in user_message.lower() for k in ("visual", "visualize", "analyze", "analysis", "insight")):
                try:
                    viz = await visualize_content(multimodal_context)
                    unified_prompt += (
                        f"\n\n[Visualization Summary]:\nSummary: {viz.get('summary','')}\n"
                        f"Themes: {', '.join(viz.get('themes',[]))}\n"
                    )
                except Exception as e:
                    logger.warning(f"visualize_content failed in /nlp: {e}")

            chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
            past_messages = chat_entry.get("messages", []) if chat_entry else []
            chat_history = filter_think_messages(past_messages[-4:])

            long_term_memory = await memory_collection.find_one({"user_id": user_id})
            long_term_memory_summary = long_term_memory.get("summary", "") if long_term_memory else ""

            system_prompt = (
                "You are Stelle, a friendly, empathetic AI companion designed for natural, one-on-one voice conversations. "
                "Respond in a casual, engaging manner as if speaking directly to the user, avoiding formal text-like responses.,dont give response in markdowns make it like you and user doing conversation"
                "Use a warm tone, show interest in the user's input, and adapt to their context. "
                "The system prompt also contains goal-management commands. [System prompt content omitted for brevity].\n"
                f"Current date/time: {current_date}\n"
            )

            messages = [{"role": "system", "content": system_prompt}]
            if long_term_memory_summary:
                messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory_summary}"})
            if goals_context:
                messages.append({"role": "system", "content": goals_context})
            for msg in chat_history:
                cleaned = re.sub(r"<think>.*?</think>", "", msg.get("content", ""), flags=re.DOTALL).strip()
                if cleaned:
                    cleaned = cleaned[:800] + "…" if len(cleaned) > 800 else cleaned
                    messages.append({"role": msg["role"], "content": cleaned})

            messages.append({"role": "user", "content": unified_prompt})

            # --- 4. Stream Response (NLP) ---
            selected_key = random.choice(GENERATE_API_KEYS)
            client_generate = AsyncGroq(api_key=selected_key)

            stream = await client_generate.chat.completions.create(
                messages=messages,
                model="deepseek-r1-distill-llama-70b",
                max_completion_tokens=4000,
                temperature=0.7,
                stream=True,
                reasoning_format="hidden",
            )

            full_reply = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                full_reply += delta
                await websocket.send_json({"status": "streaming", "message": delta})

            reply_content = full_reply.strip()

            await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)

            remind_match = re.search(r"remind me (.+)", user_message, re.IGNORECASE)
            if remind_match:
                reminder_text = remind_match.group(1).strip()
                await schedule_immediate_reminder(user_id, reminder_text)

            lines = reply_content.split("\n")
            clean_lines = [line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())]
            reply_content_clean = "\n".join(clean_lines).strip()

            user_embedding = await generate_text_embedding(user_message)
            assistant_embedding = await generate_text_embedding(reply_content_clean)

            new_messages = [
                {"role": "user", "content": user_message, "embedding": user_embedding},
                {"role": "assistant", "content": reply_content_clean, "embedding": assistant_embedding},
            ]

            if chat_entry:
                await chats_collection.update_one({"_id": chat_entry["_id"]}, {"$push": {"messages": {"$each": new_messages}}})
                updated_messages = chat_entry.get("messages", []) + new_messages
            else:
                new_chat_entry = {"user_id": user_id, "session_id": session_id, "messages": new_messages, "last_updated": datetime.now(timezone.utc)}
                await chats_collection.insert_one(new_chat_entry)
                updated_messages = new_messages

            if len(updated_messages) >= 10:
                asyncio.create_task(store_long_term_memory(user_id, session_id, updated_messages[-10:]))

            for filename in used_filenames:
                await uploads_collection.update_many({"user_id": user_id, "session_id": session_id, "filename": filename}, {"$inc": {"query_count": 1}})

            await websocket.send_json({"status": "complete", "message": reply_content_clean})

    except Exception as e:
        logger.error(f"Error in /nlp endpoint: {e}")
        try:
            await websocket.send_json({"error": f"Internal error processing your request: {e}"})
        except Exception:
            logger.error("Failed to send error over websocket.")
    finally:
        await websocket.close()


# --- AI Assist Endpoints ---

@router.post("/aiassist")
async def ai_assist_endpoint(input_data: dict):
    """
    Generates captions + keywords + hashtags for selected platforms.
    """
    from services.post_generator_service import (
        generate_keywords_post,
        fetch_platform_hashtags,
        generate_caption_post
    )

    query = input_data.get("query")
    platforms = input_data.get("platforms", ["instagram"])

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field.")

    # 1️⃣ Generate keywords
    seed_keywords = await generate_keywords_post(None, query)

    # 2️⃣ Generate captions & hashtags (your updated function returns both)
    result = await generate_caption_post(query, seed_keywords, platforms)

    return {
        "query": query,
        "platforms": platforms,
        "keywords": seed_keywords,
        "captions": result["captions"],
        "hashtags": result["platform_hashtags"]
    }

# Store deepsearch requests temporarily
deepsearch_jobs = {}

@router.post("/start_deepsearch")
async def start_deepsearch(request: Request):
    """
    FRONTEND sends:
    {
        "user_id": "...",
        "session_id": "...",
        "prompt": "...",
        "filenames": []
    }
    Returns: { "query_id": "uuid" }
    """
    data = await request.json()
    prompt = data.get("prompt")

    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    query_id = str(uuid.uuid4())
    deepsearch_jobs[query_id] = data   # save job for websocket

    return {"query_id": query_id}


# WebSocket DeepSearch

@router.websocket("/ws/deepsearch/{query_id}")
async def ws_deepsearch(websocket: WebSocket, query_id: str):
    await websocket.accept()

    job = deepsearch_jobs.pop(query_id, None)
    if not job:
        await websocket.send_json({
            "step": "error",
            "message": "Invalid query_id"
        })
        await websocket.close()
        return

    prompt = job["prompt"]

    try:
        # ----------------------------
        # 🧭 PHASE UPDATES (UX ONLY)
        # ----------------------------
        await websocket.send_json({
            "step": "phase",
            "message": "Searching sources..."
        })
        await asyncio.sleep(0.4)

        await websocket.send_json({
            "step": "phase",
            "message": "Reading articles..."
        })
        await asyncio.sleep(0.4)

        await websocket.send_json({
            "step": "phase",
            "message": "Analyzing data..."
        })
        await asyncio.sleep(0.4)

        await websocket.send_json({
            "step": "phase",
            "message": "Drafting answer..."
        })

        # ----------------------------
        # 🤖 LLM STREAMING STARTS
        # ----------------------------
        client = AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))

        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. "
                        "Give a clear, structured, factual answer."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_completion_tokens=1500,
            stream=True,
        )

        full_answer = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_answer += delta

                # 🔥 LIVE STREAMING TO FRONTEND
                await websocket.send_json({
                    "step": "stream",
                    "delta": delta
                })

        # ----------------------------
        # ✅ FINAL MESSAGE
        # ----------------------------
        await websocket.send_json({
            "step": "done",
            "result": full_answer
        })

        # ----------------------------
        # 💾 STORE DEEPSEARCH IN CHAT HISTORY
        # ----------------------------
        user_id = job.get("user_id")
        session_id = job.get("session_id")

        if user_id and session_id and full_answer.strip():
            chat_entry = await chats_collection.find_one({
                "user_id": user_id,
                "session_id": session_id
            })

            deepsearch_message = {
                "role": "assistant",
                "content": full_answer.strip(),
                "type": "deepsearch",
                "timestamp": datetime.now(timezone.utc)
            }

            if chat_entry:
                await chats_collection.update_one(
                    {"_id": chat_entry["_id"]},
                    {
                        "$push": {"messages": deepsearch_message},
                        "$set": {"last_updated": datetime.now(timezone.utc)}
                    }
                )
            else:
                await chats_collection.insert_one({
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": [deepsearch_message],
                    "last_updated": datetime.now(timezone.utc)
                })


    except WebSocketDisconnect:
        logger.info("DeepSearch client disconnected")

    except Exception as e:
        logger.error(f"DeepSearch WS error: {e}")
        await websocket.send_json({
            "step": "error",
            "message": str(e)
        })

    finally:
        await websocket.close()

        
# -------------------------
# SIMPLE CHAT ENDPOINT
# -------------------------

@router.get("/chat-history")
async def get_chat_history(
    user_id: str = Query(...),
    session_id: str = Query(...)
):
    """
    Backend endpoint to fetch chat history for a user session.
    Read-only. Used for debugging / compatibility.
    """
    try:
        chat_entry = await chats_collection.find_one(
            {"user_id": user_id, "session_id": session_id},
            {"_id": 0, "messages": 1}
        )

        if chat_entry and chat_entry.get("messages"):
            return {
                "messages": filter_think_messages(chat_entry["messages"])
            }

        return {"messages": []}

    except Exception as e:
        logger.error(
            f"Chat history retrieval error: user_id={user_id}, session_id={session_id}, error={e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Error retrieving chat history"
        )


@router.post("/chat")
async def chat_endpoint(request: Request):
    """
    Frontend uses /chat.
    Internally we forward the request to /generate.
    No deepsearch.
    No visualize.
    Only normal chat.
    """
    return await generate_response_endpoint(request, BackgroundTasks())
