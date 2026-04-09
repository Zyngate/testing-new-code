# stelle_backend/routes/chat_routes.py
import asyncio
import json
import re
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Union, Tuple, Optional, cast
from groq import Groq, AsyncGroq
import uuid
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query, WebSocket,WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np

from models.common_models import (
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
    generate_thinking_steps
)
from services.goal_service import update_task_goal_status, schedule_immediate_reminder
from services.common_utils import get_current_datetime, filter_think_messages, convert_object_ids,sanitize_chat_history
from config import logger, GENERATE_API_KEYS
from database import doc_index, code_index, file_doc_memory_map, code_memory_map

router = APIRouter(tags=["Chat"])

visualize_queries = {}
visualize_jobs = {}
deepsearch_queries = {}


def _streaming_headers() -> Dict[str, str]:
    """Headers that help proxies/clients avoid buffering streamed chunks."""
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Content-Type-Options": "nosniff",
    }


def _should_use_sse(request: Request, payload: Optional[Dict[str, Any]] = None) -> bool:
    """Use SSE when explicitly requested by headers, query params, or request body flags."""
    accept = (request.headers.get("accept") or "").lower()
    if "text/event-stream" in accept:
        return True

    query_stream = (request.query_params.get("stream") or request.query_params.get("stream_mode") or "").lower().strip()
    if query_stream in {"sse", "event-stream"}:
        return True

    header_stream = (request.headers.get("x-stream-mode") or "").lower().strip()
    if header_stream in {"sse", "event-stream"}:
        return True

    if payload and isinstance(payload, dict):
        body_mode = str(payload.get("stream_mode") or payload.get("streamMode") or "").lower().strip()
        if body_mode in {"sse", "event-stream"}:
            return True
        if payload.get("sse") is True:
            return True

    return False


def _stream_media_type(use_sse: bool) -> str:
    return "text/plain"


def _encode_stream_chunk(text: str, use_sse: bool) -> str:
    if not use_sse:
        return text
    if not text:
        return ""
    sse_safe = text.replace("\n", "\ndata: ")
    return f"data: {sse_safe}\n\n"


async def _yield_word_chunks(text: str, use_sse: bool, delay_seconds: float = 0.01):
    """Yield text in small word-level pieces to make streaming visibly incremental in UI."""
    if not text:
        return
    parts = re.findall(r"\S+\s*|\n", text)
    for part in parts:
        if not part:
            continue
        yield _encode_stream_chunk(part, use_sse)
        await asyncio.sleep(delay_seconds)

def is_small_talk(message: str) -> bool:
    msg = message.lower().strip()
    return msg in ["hi", "hello", "hey", "yo"]


def is_simple_query(message: str) -> bool:
    msg = message.lower().strip()
    patterns = [
        r"^(what(?:'s| is)?\s+)?(the\s+)?time(\s+now)?\??$",
        r"^(what(?:'s| is)?\s+)?(the\s+)?date(\s+today|\s+now)?\??$",
        r"^(today'?s\s+date)\??$",
        r"^(current\s+time|current\s+date)\??$",
    ]
    return any(re.match(p, msg) for p in patterns)


def _clean_chat_response_text(text: str) -> str:
    """Remove internal control artifacts from model output shown to users."""
    if not text:
        return ""

    cleaned = re.sub(r"<plan>.*?</plan>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:[\w+-]+)?\n?", "", cleaned)
    cleaned = cleaned.replace("```", "")

    clean_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if re.match(r"^\[[A-Z_]+\s*:\s*.*\]$", stripped):
            continue
        clean_lines.append(line)

    cleaned = "\n".join(clean_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _extract_text_from_research_result(result: Any) -> str:
    """Normalize query_internet_via_groq output to plain text."""
    if isinstance(result, tuple) and result:
        head = result[0]
        return head if isinstance(head, str) else ""
    return result if isinstance(result, str) else ""

# -------------------------
# Helper: normalize platform names and detect chosen platform field
# -------------------------
POSSIBLE_PLATFORM_FIELDS = ["platform", "default_platform", "selected_platform", "social_platform"]

def build_dynamic_approach_line(prompt: str) -> str:
    p = prompt.lower()

    # Beginner / learning intent
    if any(k in p for k in ["what is", "basics", "beginner", "introduction", "learn"]):
        return (
            "I’ll start from the fundamentals, explain the core ideas clearly, "
            "and build things up step by step so it’s easy to follow."
        )

    # Growth / marketing intent
    if any(k in p for k in ["grow", "growth", "followers", "marketing", "branding"]):
        return (
            "I’ll first clarify what actually drives results here, then break down "
            "the strategies that work today, and finally show how to apply them to your case."
        )

    # Execution / how-to intent
    if any(k in p for k in ["how to", "steps", "implement", "process"]):
        return (
            "I’ll walk through this practically — focusing on what to do, why it matters, "
            "and how to execute it without overcomplicating things."
        )

    # Comparison / decision intent
    if any(k in p for k in ["vs", "difference", "compare", "better"]):
        return (
            "I’ll compare the options clearly, highlight the trade-offs, "
            "and help you decide based on real-world use cases."
        )

    # Strategy / advanced intent
    if any(k in p for k in ["strategy", "framework", "optimize", "scale"]):
        return (
            "I’ll approach this strategically — connecting principles, frameworks, "
            "and practical decisions so it makes sense at a higher level."
        )

    # Fallback (safe)
    return (
        "I’ll approach this in a clear, structured way — giving you context first, "
        "then insights, and ending with practical takeaways."
    )


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
        # ✅ STEP 1: Extract ONLY internal plan
        plan_match = re.search(r"<plan>(.*?)</plan>", reply_content, re.DOTALL)
        plan_content = plan_match.group(1) if plan_match else ""

        # ⛔ No plan → no DB mutation
        if not plan_content.strip():
            return

        new_goals_map = {}

        # ✅ STEP 2: Parse ONLY from plan_content
        goal_set_matches = re.findall(r"\[GOAL_SET: (.*?)\]", plan_content)
        for goal_phrase in goal_set_matches:
            await update_task_goal_status(
                user_id, session_id,
                goal_phrase,
                "set_goal",
                None, None,
                new_goals_map
            )

        task_matches = re.findall(r"\[TASK: (.*?)\]", plan_content)
        for task_desc in task_matches:
            if new_goals_map:
                goal_id_placeholder = next(iter(new_goals_map.keys()))
                goal_id = new_goals_map.get(goal_id_placeholder, goal_id_placeholder)
                await update_task_goal_status(
                    user_id, session_id,
                    task_desc,
                    "add_task",
                    goal_id, None,
                    new_goals_map
                )

        task_add_matches = re.findall(r"\[TASK_ADD:\s*(.*?):\s*(.*?)\]", plan_content)
        for goal_id_str, task_desc in task_add_matches:
            await update_task_goal_status(
                user_id, session_id,
                task_desc,
                "add_task",
                goal_id_str, None,
                new_goals_map
            )

        commands = {
            "GOAL_DELETE": re.findall(r"\[GOAL_DELETE: (.*?)\]", plan_content),
            "TASK_DELETE": re.findall(r"\[TASK_DELETE: (.*?)\]", plan_content),
            "TASK_MODIFY": re.findall(r"\[TASK_MODIFY:\s*(.*?):\s*(.*?)\]", plan_content),
            "GOAL_START": re.findall(r"\[GOAL_START: (.*?)\]", plan_content),
            "TASK_START": re.findall(r"\[TASK_START: (.*?)\]", plan_content),
            "GOAL_COMPLETE": re.findall(r"\[GOAL_COMPLETE: (.*?)\]", plan_content),
            "TASK_COMPLETE": re.findall(r"\[TASK_COMPLETE: (.*?)\]", plan_content),
        }

        for command, matches in commands.items():
            if command == "TASK_MODIFY":
                for tid, new_desc in matches:
                    await update_task_goal_status(
                        user_id, session_id,
                        new_desc,
                        command,
                        None, tid,
                        new_goals_map
                    )
            elif command in ["GOAL_START", "GOAL_COMPLETE", "GOAL_DELETE"]:
                for gid in matches:
                    await update_task_goal_status(
                        user_id, session_id,
                        None,
                        command,
                        gid, None,
                        new_goals_map
                    )
            elif command in ["TASK_START", "TASK_COMPLETE", "TASK_DELETE"]:
                for tid in matches:
                    await update_task_goal_status(
                        user_id, session_id,
                        None,
                        command,
                        None, tid,
                        new_goals_map
                    )

        task_deadline_matches = re.findall(r"\[TASK_DEADLINE:\s*(.*?):\s*(.*?)\]", plan_content)
        for tid, deadline_str in task_deadline_matches:
            await update_task_goal_status(
                user_id, session_id,
                deadline_str,
                "TASK_DEADLINE",
                None, tid,
                new_goals_map
            )

        task_progress_matches = re.findall(r"\[TASK_PROGRESS:\s*(.*?):\s*(.*?)\]", plan_content)
        for tid, progress_desc in task_progress_matches:
            await update_task_goal_status(
                user_id, session_id,
                progress_desc,
                "TASK_PROGRESS",
                None, tid,
                new_goals_map
            )

        await update_rag_usage_and_cleanup(user_id, session_id)

    except Exception as e:
        logger.error(f"Error in handle_goal_updates_and_cleanup: {e}")



# ---------------------------------------------------------------------------
# Persistence helpers (called via asyncio.create_task — outside stream scope)
# ---------------------------------------------------------------------------

async def _persist_turn(user_id: str, session_id: str, user_message: str, reply: str):
    """Persist a single user/assistant turn to chat history."""
    try:
        user_embedding = await generate_text_embedding(user_message)
        assistant_embedding = await generate_text_embedding(reply)
        new_messages = [
            {"role": "user", "content": user_message, **(({"embedding": user_embedding}) if user_embedding else {})},
            {"role": "assistant", "content": reply, **(({"embedding": assistant_embedding}) if assistant_embedding else {})},
        ]
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        if chat_entry:
            await chats_collection.update_one(
                {"_id": chat_entry["_id"]},
                {"$push": {"messages": {"$each": new_messages}}, "$set": {"last_updated": datetime.now(timezone.utc)}},
            )
        else:
            await chats_collection.insert_one({
                "user_id": user_id, "session_id": session_id,
                "messages": new_messages, "last_updated": datetime.now(timezone.utc),
            })
    except Exception as e:
        logger.error(f"_persist_turn error: {e}")


async def _post_stream_persist(
    user_id: str,
    session_id: str,
    user_message: str,
    reply_content: str,
    reply_content_clean: str,
    chat_entry: Optional[dict],
    used_filenames: set,
):
    """All DB work that happens after the stream finishes."""
    try:
        # 1. Goal/task command parsing
        await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)

        # 2. Embeddings
        user_embedding = await generate_text_embedding(user_message)
        assistant_embedding = await generate_text_embedding(reply_content_clean)

        new_messages = [
            {"role": "user", "content": user_message, **(({"embedding": user_embedding}) if user_embedding else {})},
            {"role": "assistant", "content": reply_content_clean, **(({"embedding": assistant_embedding}) if assistant_embedding else {})},
        ]

        # 3. Persist to DB
        if chat_entry:
            await chats_collection.update_one(
                {"_id": chat_entry["_id"]},
                {"$push": {"messages": {"$each": new_messages}}, "$set": {"last_updated": datetime.now(timezone.utc)}},
            )
            updated_messages = chat_entry.get("messages", []) + new_messages
        else:
            await chats_collection.insert_one({
                "user_id": user_id, "session_id": session_id,
                "messages": new_messages, "last_updated": datetime.now(timezone.utc),
            })
            updated_messages = new_messages

        # 4. Long-term memory
        if len(updated_messages) >= 10:
            asyncio.create_task(store_long_term_memory(user_id, session_id, updated_messages[-10:]))

        # 5. RAG usage count
        for filename in used_filenames:
            await uploads_collection.update_many(
                {"user_id": user_id, "session_id": session_id, "filename": filename},
                {"$inc": {"query_count": 1}},
            )
    except Exception as e:
        logger.error(f"_post_stream_persist error: {e}")


# --- Endpoints ---


@router.post("/generate", response_model=GenerateResponse)
async def generate_response_endpoint(request: Request, background_tasks: BackgroundTasks):
    try:
        current_date = get_current_datetime()
        data = await request.json()

        # Accept both snake_case and camelCase payload styles used by different frontend builds.
        user_id = str((data.get("user_id") or data.get("userId") or "")).strip()
        session_id = str((data.get("session_id") or data.get("sessionId") or "")).strip()
        user_message = str((data.get("prompt") or data.get("message") or "")).strip()

        filenames_raw = data.get("filenames")
        if filenames_raw is None:
            filenames_raw = data.get("files")
        if isinstance(filenames_raw, str):
            filenames = [f.strip() for f in re.split(r"[,\n]+", filenames_raw) if f.strip()]
        elif isinstance(filenames_raw, list):
            filenames = [str(f).strip() for f in filenames_raw if str(f).strip()]
        else:
            filenames = []

        if not user_id or not session_id or not user_message:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid request parameters. Required fields: "
                    "user_id/session_id/prompt (or userId/sessionId/message)."
                ),
            )

        # /chat forwards into this handler and should stay lightweight for fast token streaming.
        is_basic_chat_mode = request.url.path.rstrip("/").endswith("/chat")
        use_sse = _should_use_sse(request, data)

        # --- Short-circuit: small talk ---
        if is_small_talk(user_message):
            reply = "Hello. How can I assist you?"
            asyncio.create_task(_persist_turn(user_id, session_id, user_message, reply))
            async def _small_talk_stream():
                if use_sse:
                    yield ": stream-open\n\n"
                yield _encode_stream_chunk(reply, use_sse)
            return StreamingResponse(_small_talk_stream(), media_type=_stream_media_type(use_sse), headers=_streaming_headers())

        # --- Short-circuit: simple date/time query ---
        if is_simple_query(user_message):
            lower_msg = user_message.lower()
            if "time" in lower_msg:
                reply = f"Current time is {datetime.now().strftime('%I:%M %p')} on {datetime.now().strftime('%B %d, %Y')}."
            else:
                reply = f"Today is {datetime.now().strftime('%B %d, %Y')}."
            asyncio.create_task(_persist_turn(user_id, session_id, user_message, reply))
            async def _simple_query_stream():
                if use_sse:
                    yield ": stream-open\n\n"
                yield _encode_stream_chunk(reply, use_sse)
            return StreamingResponse(_simple_query_stream(), media_type=_stream_media_type(use_sse), headers=_streaming_headers())

        # --- 1. Gather Context (Goals, Files, URLs, Memory) ---
        active_goals = await goals_collection.find({"user_id": user_id, "status": {"$in": ["active", "in progress"]}}).to_list(None)
        goals_context = ""
        if active_goals:
            goals_context = "User's current goals and tasks:\n"
            for goal in active_goals:
                goals_context += f"- Goal: {goal['title']} ({goal['status']})\n"
                for task in goal.get("tasks", []):
                    goals_context += f"  - Task: {task['title']} ({task['status']})\n"

        uploaded_files = await uploads_collection.distinct("filename", {"session_id": session_id})
        mentioned_filenames = [fn for fn in uploaded_files if fn.lower() in user_message.lower()]
        hooked_filenames = filenames if filenames else mentioned_filenames

        external_content = ""
        url_match = re.search(r"https?://[^\s]+", user_message)
        if url_match:
            url = url_match.group(0)
            if "youtube.com" in url or "youtu.be" in url:
                summary_raw = await query_internet_via_groq(f"Summarize the content of the YouTube video at {url}")
                summary = _extract_text_from_research_result(summary_raw)
                if summary:
                    external_content = await detailed_explanation(summary)
            else:
                summary_raw = await query_internet_via_groq(f"Summarize the content of the webpage at {url}")
                summary = _extract_text_from_research_result(summary_raw)
                if summary:
                    external_content = await content_for_website(summary)

        multimodal_context, used_filenames = await retrieve_multimodal_context(user_message, session_id, hooked_filenames)

        # --- 2. Construct Unified Prompt ---
        unified_prompt = user_message
        if external_content:
            unified_prompt += f"\n[External Content]:\n{external_content}\n"
        if multimodal_context:
            unified_prompt += f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
        unified_prompt += "\n\nRespond appropriately based on the query."

        research_needed = "none"
        if not is_basic_chat_mode:
            research_needed = await classify_prompt(user_message)

        if (not is_basic_chat_mode) and research_needed == "research" and not multimodal_context:
            try:
                ds_content, ds_sources = await query_deepsearch(user_message)
                if ds_content and "Error" not in ds_content and "unavailable" not in ds_content:
                    unified_prompt += f"\n\n[DeepSearch Results]:\n{ds_content}\n"
                    if ds_sources:
                        unified_prompt += "\nSources:\n" + "\n".join([f"- {s['title']}: {s['url']}" for s in ds_sources])
                else:
                    research_results_raw = await query_internet_via_groq(user_message)
                    research_results = _extract_text_from_research_result(research_results_raw)
                    if research_results and research_results != "Error accessing internet information.":
                        unified_prompt += f"\n\n[Additional Research]:\n{research_results}"
            except Exception as e:
                logger.warning(f"DeepSearch failed in /generate, falling back: {e}")
                research_results_raw = await query_internet_via_groq(user_message)
                research_results = _extract_text_from_research_result(research_results_raw)
                if research_results and research_results != "Error accessing internet information.":
                    unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

        if (not is_basic_chat_mode) and multimodal_context and any(k in user_message.lower() for k in ("visual", "visualize", "analyze", "analysis", "insight")):
            try:
                viz = await visualize_content(multimodal_context)
                unified_prompt += (
                    f"\n\n[Visualization Summary]:\nSummary: {viz.get('summary','')}\n"
                    f"Themes: {', '.join(viz.get('themes',[]))}\n"
                )
            except Exception as e:
                logger.warning(f"visualize_content failed in /generate: {e}")

        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        past_messages = chat_entry.get("messages", []) if chat_entry else []
        past_messages = sanitize_chat_history(past_messages)
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

        long_term_memory = await memory_collection.find_one({"user_id": user_id})
        long_term_memory_summary = long_term_memory.get("summary", "") if long_term_memory else ""

        system_prompt = (
    "You are Stelle, an advanced AI assistant — intelligent, articulate, and direct.\n\n"
    "Core behavior:\n"
    "- Answer any question on any topic: technology, science, business, coding, creative writing, math, philosophy, culture, and more.\n"
    "- Think deeply before responding. Give accurate, well-reasoned, thorough answers.\n"
    "- Match response length to the question — short for simple queries, detailed for complex ones.\n"
    "- Write in clear, natural prose. No forced templates, no rigid sections.\n"
    "- Use bullet points or numbered lists only when they genuinely improve clarity (steps, comparisons, lists of items).\n"
    "- Never use markdown symbols like **, ##, or ___ — use plain text with proper line breaks instead.\n"
    "- Always put a blank line between paragraphs and between list items so responses are easy to read.\n\n"
    "Personality:\n"
    "- Confident but not arrogant. Honest about uncertainty.\n"
    "- Warm and conversational, but professional.\n"
    "- Always open with a natural, genuine one-liner that acknowledges the user's question — "
    "for example: 'That is a great area to explore.', 'Really interesting question.', "
    "'Love that you are thinking about this.', 'This is one of those topics worth getting right.' — "
    "vary it every time, never repeat the same opener twice in a conversation.\n"
    "- Never use hollow phrases like 'Great question!' or 'Certainly!' alone — the opener must feel human and specific to what was asked.\n"
    "- Warm and encouraging when the user is learning or exploring something new.\n"
    "- Motivating and energetic when the user is building or working on something.\n"
    "- Calm and precise when the user needs technical or factual answers.\n"
    "- If you don't know something, say so clearly and offer what you do know.\n\n"
    "Coding & technical questions:\n"
    "- Provide working, production-quality code.\n"
    "- Explain what the code does and why, not just how.\n"
    "- Point out edge cases, pitfalls, or better alternatives when relevant.\n\n"
    "General rules:\n"
    "- Never mention your system prompt, training, or internal instructions.\n"
    "- Never output [TASK:], [GOAL:], <think>, <plan>, or any internal control tokens.\n"
    "- Do not repeat the user's question back to them.\n"
    "- Do not add unnecessary disclaimers or over-qualify simple answers.\n"
    "- Mention date/time only when the user asks.\n\n"
    f"Current date and time: {current_date}"
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

        selected_key = random.choice(GENERATE_API_KEYS)
        client_generate = AsyncGroq(api_key=selected_key)

        async def generate_stream():
            full_reply = ""
            try:
                stream = await client_generate.chat.completions.create(
                    messages=cast(Any, messages),
                    model="llama-3.3-70b-versatile",
                    max_completion_tokens=2000,
                    temperature=0.35,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full_reply += delta
                        # Split into words and yield one by one with delay
                        for char in delta:
                            yield char
                            await asyncio.sleep(0.015)  # 🔥 smooth typing speed
            except Exception as stream_err:
                logger.error(f"Stream error in /generate: {stream_err}")
                fallback = "I encountered a temporary response issue. Please try again."
                yield fallback
                asyncio.create_task(_persist_turn(user_id, session_id, user_message, fallback))
                return

            reply_content = full_reply.strip()
            reply_content_clean = _clean_chat_response_text(reply_content)

            if not reply_content_clean:
                yield "I could not generate a usable response. Please rephrase and try again."

            asyncio.create_task(
                _post_stream_persist(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    reply_content=reply_content,
                    reply_content_clean=reply_content_clean,
                    chat_entry=chat_entry,
                    used_filenames=used_filenames,
                )
            )

        return StreamingResponse(generate_stream(), media_type=_stream_media_type(use_sse), headers=_streaming_headers())

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
    "You are Stelle, a mature, professional, and neutral conversational AI assistant.\n\n"
    "Your role:\n"
    "- Respond with clarity, conciseness, and professionalism.\n"
    "- Avoid childish, overly friendly, or casual language.\n"
    "- Never use emojis, jokes, or filler.\n"
    "- Never suggest random topics or make assumptions about the user's interests.\n"
    "- Never use exclamation marks unless quoting.\n"
    "- Never use phrases like 'Hey!', 'Hi there!', or similar.\n"
    "- Never use slang or informal expressions.\n"
    "- Never use markdown formatting.\n"
    "- Never mention the current date/time unless explicitly asked.\n\n"

    "For greetings (hi, hello, hey, etc.):\n"
    "- Respond with a brief, neutral greeting and ask how you can assist.\n"
    "- Example: 'Hello. How can I assist you?'\n\n"

    "When answering:\n"
    "- Use a neutral, informative, and direct tone.\n"
    "- For all substantive queries, follow this exact flow:\n"
    "  1) First line with empathy/appreciation/motivation relevant to user context.\n"
    "  2) Brief introduction describing the query focus.\n"
    "  3) Deep Dive: heading must be exactly 'Deep Dive:' followed by at least 4 bullets.\n"
    "     Every bullet must begin with '- ' and contain practical value.\n"
    "  4) Short conclusion with key takeaway.\n"
    "  5) End with one relevant next-step question.\n"
    "- Use section labels exactly: 'Introduction:', 'Deep Dive:', 'Conclusion:'.\n"
    "- Allow bullets only in 'Deep Dive:'. No bullets are allowed in the empathy line, Introduction, Conclusion, or follow-up question.\n"
    "- 'Conclusion:' must be paragraph text on its own lines, not appended to any bullet.\n"
    "- For substantive informational answers, target about 1800-2200 characters including spaces unless user asks for a different length.\n"
    "- Add substance and specificity, not filler.\n"
    "- For simple queries, keep the answer short but include a warm opener and one follow-up question.\n"
    "- Avoid unnecessary elaboration or repetition.\n"
    "- Never sound like a coach, consultant, or motivational speaker.\n"
    "- Never use childish, playful, or overly friendly language.\n"
    "- Never ask follow-up questions unless the user requests more information.\n"
    "- Never use phrases like 'I'm here to help!' or 'Let me know if you need anything else.'\n\n"

    "If the user asks something simple, keep the answer simple and direct.\n"
    "If the user asks something deep, provide a thoughtful, well-structured answer, but remain concise and neutral.\n\n"

    "Never mention internal systems, plans, tasks, or tools.\n"
    "If a response sounds generic or like a textbook answer, rewrite it to sound more human and mature.\n\n"
    f"Current date and time: {current_date}"
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
            messages=cast(Any, final_messages),
            model="llama-3.3-70b-versatile",
            max_completion_tokens=4000,
            temperature=0.7,
            stream=True,
        )

        async def regeneration_stream():
            full_reply = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_reply += delta
                visible_delta = re.sub(
                    r"<plan>.*?</plan>",
                    "",
                    delta,
                    flags=re.DOTALL
                )
                if visible_delta.strip():
                    async for word_chunk in _yield_word_chunks(visible_delta, False):
                        yield word_chunk


            reply_content = full_reply.strip()

            # ---- Post-processing ----
            await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)

            # Remove goal/task commands for storage
            lines = reply_content.split("\n")
            clean_lines = [L for L in lines if not re.match(r"\[.*?: .*?\]", L.strip())]
            cleaned_reply = re.sub(
                r"<plan>.*?</plan>",
                "",
                "\n".join(clean_lines),
                flags=re.DOTALL
            ).strip()


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

        return StreamingResponse(regeneration_stream(), media_type="text/plain", headers=_streaming_headers())

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
                    goals_context += f"- Goal: {goal['title']} ({goal['status']})\n"
                    for task in goal.get("tasks", []):
                        goals_context += f"  - Task: {task['title']} ({task['status']})\n"

            uploaded_files = await uploads_collection.distinct("filename", {"session_id": session_id})
            mentioned_filenames = [fn for fn in uploaded_files if fn.lower() in user_message.lower()]
            hooked_filenames = filenames if filenames else mentioned_filenames

            external_content = ""
            url_match = re.search(r"https?://[^\s]+", user_message)
            if url_match:
                url = url_match.group(0)
                if "youtube.com" in url or "youtu.be" in url:
                    summary_raw = await query_internet_via_groq(f"Summarize the content of the YouTube video at {url}")
                    summary = _extract_text_from_research_result(summary_raw)
                    if summary:
                        external_content = await detailed_explanation(summary)
                else:
                    summary_raw = await query_internet_via_groq(f"Summarize the content of the webpage at {url}")
                    summary = _extract_text_from_research_result(summary_raw)
                    if summary:
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
                        research_results_raw = await query_internet_via_groq(user_message)
                        research_results = _extract_text_from_research_result(research_results_raw)
                        if research_results and research_results != "Error accessing internet information.":
                            unified_prompt += f"\n\n[Additional Research]:\n{research_results}"
                except Exception as e:
                    logger.warning(f"DeepSearch failed in /nlp, falling back: {e}")
                    research_results_raw = await query_internet_via_groq(user_message)
                    research_results = _extract_text_from_research_result(research_results_raw)
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
                messages=cast(Any, messages),
                model="deepseek-r1-distill-llama-70b",
                max_completion_tokens=4000,
                temperature=0.7,
                stream=True,
                reasoning_format="hidden",
            )

            full_reply = ""
            
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_reply += delta
                visible_delta = re.sub(
                    r"<plan>.*?</plan>",
                    "",
                    delta,
                    flags=re.DOTALL
                )
                if visible_delta.strip():
                    await websocket.send_json({
                        "status": "streaming",
                        "message": visible_delta
                     })
            reply_content = full_reply.strip()

            await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)

            remind_match = re.search(r"remind me (.+)", user_message, re.IGNORECASE)
            if remind_match:
                reminder_text = remind_match.group(1).strip()
                await schedule_immediate_reminder(user_id, reminder_text)

            lines = reply_content.split("\n")
            clean_lines = [line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())]
            reply_content_clean = re.sub(
                r"<plan>.*?</plan>",
                "",
                "\n".join(clean_lines),
                flags=re.DOTALL
            ).strip()


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
    keyword_client = AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))
    seed_keywords = await generate_keywords_post(keyword_client, query)

    # 2️⃣ Generate captions & hashtags (your updated function returns both)
    result = await generate_caption_post(query, seed_keywords, platforms)

    return {
        "query": query,
        "platforms_requested": platforms,
        "keywords": seed_keywords,
        "platforms": result.get("platforms", {})
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
        # --------------------------------
        # 🧠 HUMAN-LIKE THINKING (VISIBLE)
        # --------------------------------
        thinking_steps = await generate_thinking_steps(prompt)
        for thought in thinking_steps:
            await websocket.send_json({
                "step": "thinking",
                "message": thought
            })
            await asyncio.sleep(0.5)

        # --------------------------------
        # 🧭 USER-AWARE APPROACH (UX ONLY)
        # --------------------------------
        approach_line = build_dynamic_approach_line(prompt)
        await websocket.send_json({
            "step": "approach",
            "title": "How I’ll approach this",
            "message": approach_line
        })
        await asyncio.sleep(0.6)

        # --------------------------------
        # 🧭 PHASE UPDATES (UX ONLY)
        # --------------------------------
        for phase in [
            "Searching sources...",
            "Reading articles...",
            "Analyzing data...",
            "Drafting answer..."
        ]:
            await websocket.send_json({
                "step": "phase",
                "message": phase
            })
            await asyncio.sleep(0.4)

        # --------------------------------
        # 🤖 LLM STREAMING STARTS
        # --------------------------------
        client = AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))

        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=cast(Any, [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. "
                        "Explain in a clear, structured, human-like way. "
                        "Do not mention internal reasoning explicitly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]),
            temperature=0.4,
            max_completion_tokens=1500,
            stream=True,
        )

        full_answer = ""

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_answer += delta
                await websocket.send_json({
                    "step": "stream",
                    "delta": delta
                })
        # --------------------------------
        # ✅ FINAL MESSAGE
        # --------------------------------
        await websocket.send_json({
            "step": "done",
            "result": full_answer
        })

        # --------------------------------
        # 💾 STORE DEEPSEARCH IN CHAT HISTORY
        # --------------------------------
        user_id = job.get("user_id")
        session_id = job.get("session_id")

        if user_id and session_id and full_answer.strip():
            chat_entry = await chats_collection.find_one({
                "user_id": user_id,
                "session_id": session_id
            })

            messages_to_store = [
                {
                    "role": "user",
                    "content": prompt.strip(),
                    "type": "deepsearch",
                    "timestamp": datetime.now(timezone.utc)
                },
                {
                    "role": "assistant",
                    "content": full_answer.strip(),
                    "type": "deepsearch",
                    "timestamp": datetime.now(timezone.utc)
                }
            ]

            if chat_entry:
                await chats_collection.update_one(
                    {"_id": chat_entry["_id"]},
                    {
                        "$push": {"messages": {"$each": messages_to_store}},
                        "$set": {"last_updated": datetime.now(timezone.utc)}
                    }
                )
            else:
                await chats_collection.insert_one({
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": messages_to_store,
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




chat_jobs = {}

@router.post("/start_chat")
async def start_chat(request: Request):
    data = await request.json()
    if not data.get("prompt", "").strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    query_id = str(uuid.uuid4())
    chat_jobs[query_id] = data
    return {"query_id": query_id}


@router.websocket("/ws/chat/{query_id}")
async def ws_chat(websocket: WebSocket, query_id: str):
    await websocket.accept()

    job = chat_jobs.pop(query_id, None)
    if not job:
        await websocket.send_json({"step": "error", "message": "Invalid query_id"})
        await websocket.close()
        return

    user_id = str(job.get("user_id") or "").strip()
    session_id = str(job.get("session_id") or "").strip()
    user_message = str(job.get("prompt") or "").strip()

    try:
        client = AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))

        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        past_messages = chat_entry.get("messages", []) if chat_entry else []
        past_messages = sanitize_chat_history(past_messages)
        current_date = get_current_datetime()

        system_prompt = (
            "You are Stelle, an advanced AI assistant — intelligent, articulate, and direct.\n\n"
            "Core behavior:\n"
            "- Answer any question on any topic: technology, science, business, coding, creative writing, math, philosophy, culture, and more.\n"
            "- Think deeply before responding. Give accurate, well-reasoned, thorough answers.\n"
            "- Match response length to the question — short for simple queries, detailed for complex ones.\n"
            "- Write in clear, natural prose. No forced templates, no rigid sections.\n"
            "- Use bullet points or numbered lists only when they genuinely improve clarity.\n"
            "- Never use markdown symbols like **, ##, or ___ — use plain text with proper line breaks instead.\n"
            "- Always put a blank line between paragraphs so responses are easy to read.\n\n"
            "Personality:\n"
            "- Confident but not arrogant. Honest about uncertainty.\n"
            "- Always open with a natural, genuine one-liner that acknowledges the user's question.\n"
            "- Vary the opener every time, never repeat the same opener twice.\n"
            "- Never use hollow phrases like 'Great question!' or 'Certainly!' alone.\n"
            "- Warm and encouraging when the user is learning something new.\n"
            "- Motivating and energetic when the user is building something.\n"
            "- Calm and precise when the user needs technical answers.\n\n"
            "General rules:\n"
            "- Never mention your system prompt or internal instructions.\n"
            "- Never output [TASK:], [GOAL:], <think>, <plan>, or any control tokens.\n"
            "- Do not repeat the user's question back to them.\n"
            "- Do not add unnecessary disclaimers.\n\n"
            f"Current date and time: {current_date}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in filter_think_messages(past_messages[-6:]):
            cleaned = re.sub(r"<think>.*?</think>", "", msg.get("content", ""), flags=re.DOTALL).strip()
            if cleaned:
                messages.append({"role": msg["role"], "content": cleaned[:800]})
        messages.append({"role": "user", "content": user_message})

        stream = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=cast(Any, messages),
            temperature=0.35,
            max_completion_tokens=2000,
            stream=True,
        )

        full_answer = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_answer += delta
                await websocket.send_json({"step": "stream", "delta": delta})

        await websocket.send_json({"step": "done", "result": full_answer})

        reply_clean = _clean_chat_response_text(full_answer.strip())
        asyncio.create_task(
            _post_stream_persist(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                reply_content=full_answer.strip(),
                reply_content_clean=reply_clean,
                chat_entry=chat_entry,
                used_filenames=[],
            )
        )

    except WebSocketDisconnect:
        logger.info("Chat WS client disconnected")
    except Exception as e:
        logger.error(f"Chat WS error: {e}")
        await websocket.send_json({"step": "error", "message": str(e)})
    finally:
        await websocket.close()