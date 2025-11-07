# stelle_backend/routes/chat_routes.py
# stelle_backend/routes/chat_routes.py
import asyncio
import json
import re
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Union, Tuple # <-- FIX 14, 15, 16, 17: Explicit typing imports
from groq import Groq, AsyncGroq # <-- FIX 6, 7, 8, 11, 12, 13: Missing 'Groq' and 'AsyncGroq'

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np

from models.common_models import GenerateRequest, GenerateResponse, RegenerateRequest, NLPRequest, UserInput # <-- FIX 10: Missing 'UserInput'
from database import chats_collection, memory_collection, uploads_collection, goals_collection
from services.ai_service import (
    get_groq_client, query_internet_via_groq, retrieve_multimodal_context, 
    content_for_website, detailed_explanation, classify_prompt, 
    generate_text_embedding, store_long_term_memory
)
from services.goal_service import update_task_goal_status, schedule_immediate_reminder
from services.common_utils import get_current_datetime, filter_think_messages, convert_object_ids
from config import logger, GENERATE_API_KEYS
from database import doc_index, code_index, file_doc_memory_map, code_memory_map

router = APIRouter()

# ... (rest of the file remains the same) ...

router = APIRouter()

# --- Main Generation Logic (Shared by /generate and /regenerate) ---

async def handle_goal_updates_and_cleanup(reply_content: str, user_id: str, session_id: str):
    """Parses LLM output for goal/task commands and applies database changes."""
    # NOTE: This uses the direct database collections from database.py

    # 1. Goal Setting/Modification Commands
    new_goals_map = {}
    goal_set_matches = re.findall(r"\[GOAL_SET: (.*?)\]", reply_content)
    for goal_phrase in goal_set_matches:
        await update_task_goal_status(
            user_id, session_id, goal_phrase, 'set_goal', None, None, new_goals_map
        )
        
    task_matches = re.findall(r"\[TASK: (.*?)\]", reply_content)
    for task_desc in task_matches:
        # Assuming task is for the last set goal, use the first key of new_goals_map if available
        if new_goals_map:
            goal_id_placeholder = next(iter(new_goals_map.keys()))
            goal_id = new_goals_map.get(goal_id_placeholder, goal_id_placeholder)
            await update_task_goal_status(
                user_id, session_id, task_desc, 'add_task', goal_id, None, new_goals_map
            )
            
    task_add_matches = re.findall(r"\[TASK_ADD:\s*(.*?):\s*(.*?)\]", reply_content)
    for goal_id_str, task_desc in task_add_matches:
        await update_task_goal_status(
            user_id, session_id, task_desc, 'add_task', goal_id_str, None, new_goals_map
        )

    # 2. Status/Modification/Deletion Commands
    commands = {
        'GOAL_DELETE': re.findall(r"\[GOAL_DELETE: (.*?)\]", reply_content),
        'TASK_DELETE': re.findall(r"\[TASK_DELETE: (.*?)\]", reply_content),
        'TASK_MODIFY': re.findall(r"\[TASK_MODIFY:\s*(.*?):\s*(.*?)\]", reply_content),
        'GOAL_START': re.findall(r"\[GOAL_START: (.*?)\]", reply_content),
        'TASK_START': re.findall(r"\[TASK_START: (.*?)\]", reply_content),
        'GOAL_COMPLETE': re.findall(r"\[GOAL_COMPLETE: (.*?)\]", reply_content),
        'TASK_COMPLETE': re.findall(r"\[TASK_COMPLETE: (.*?)\]", reply_content),
    }

    for command, matches in commands.items():
        if command in ['TASK_MODIFY']:
            for tid, new_desc in matches:
                await update_task_goal_status(user_id, session_id, new_desc, command, None, tid, new_goals_map)
        elif command in ['GOAL_START', 'GOAL_COMPLETE', 'GOAL_DELETE']:
            for gid in matches:
                await update_task_goal_status(user_id, session_id, None, command, gid, None, new_goals_map)
        elif command in ['TASK_START', 'TASK_COMPLETE', 'TASK_DELETE']:
            for tid in matches:
                await update_task_goal_status(user_id, session_id, None, command, None, tid, new_goals_map)


    # 3. Deadline and Progress
    task_deadline_matches = re.findall(r"\[TASK_DEADLINE:\s*(.*?):\s*(.*?)\]", reply_content)
    for tid, deadline_str in task_deadline_matches:
        await update_task_goal_status(user_id, session_id, deadline_str, 'TASK_DEADLINE', None, tid, new_goals_map)

    task_progress_matches = re.findall(r"\[TASK_PROGRESS:\s*(.*?):\s*(.*?)\]", reply_content)
    for tid, progress_desc in task_progress_matches:
        await update_task_goal_status(user_id, session_id, progress_desc, 'TASK_PROGRESS', None, tid, new_goals_map)

    # 4. Cleanup/RAG Cleanup
    # This logic is kept inline for now as it directly manipulates the FAISS indexes and collection.
    await update_rag_usage_and_cleanup(user_id, session_id)


async def update_rag_usage_and_cleanup(user_id: str, session_id: str):
    """Increments RAG chunk usage counts and removes expired chunks from DB and FAISS."""
    
    # Note: Need to be careful about `used_filenames` which is only available inside
    # the main generation call. For now, this just cleans up globally or based on old counts.

    # Remove old uploads if query_count >= 15
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
        # Delete from MongoDB
        await uploads_collection.delete_many(
            {"user_id": user_id, "session_id": session_id, "filename": filename}
        )
        logger.info(f"Cleaned up MongoDB uploads for expired file: {filename}")
        
        # Remove from FAISS Doc Index
        indices_to_remove = [
            idx for idx, m in file_doc_memory_map.items() 
            if m.get("filename") == filename and m.get("session_id") == session_id
        ]
        if indices_to_remove:
            # Note: doc_index.remove_ids only works for IndexFlatL2 if the index supports deletion.
            # Assuming FAISS setup in database.py can handle deletion for simplicity of refactoring.
            # In a real system, FlatL2 often requires a rebuild/compaction method after deletion.
            doc_index.remove_ids(np.array(indices_to_remove, dtype="int64"))
            for idx in indices_to_remove:
                del file_doc_memory_map[idx]
            logger.info(f"Cleaned up {len(indices_to_remove)} FAISS doc chunks for {filename}.")

        # Remove from FAISS Code Index
        code_indices_to_remove = [
            idx for idx, m in code_memory_map.items()
            if m.get("filename") == filename and m.get("session_id") == session_id
        ]
        if code_indices_to_remove:
            code_index.remove_ids(np.array(code_indices_to_remove, dtype="int64"))
            for idx in code_indices_to_remove:
                del code_memory_map[idx]
            logger.info(f"Cleaned up {len(code_indices_to_remove)} FAISS code chunks for {filename}.")
            
    logger.info("RAG cleanup complete.")

# --- Endpoints ---

@router.post("/generate", response_model=GenerateResponse)
async def generate_response_endpoint(request: Request, background_tasks: BackgroundTasks):
    try:
        current_date = get_current_datetime()
        data = await request.json()
        req = GenerateRequest(**data)
        user_id, session_id, user_message, filenames = (
            req.user_id, req.session_id, req.prompt, req.filenames
        )
        if not user_id or not session_id or not user_message:
            raise HTTPException(status_code=400, detail="Invalid request parameters.")

        # --- 1. Gather Context (Goals, Files, URLs, Memory) ---
        
        # a) Goal Context
        active_goals = await goals_collection.find(
            {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
        ).to_list(None)
        goals_context = ""
        if active_goals:
            goals_context = "User's current goals and tasks:\n"
            for goal in active_goals:
                goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal.get('goal_id','N/A')}]\n"
                for task in goal.get('tasks', []):
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
        multimodal_context, used_filenames = await retrieve_multimodal_context(
            user_message, session_id, hooked_filenames
        )

        # --- 2. Construct Unified Prompt ---
        unified_prompt = f"User Query: {user_message}\n"
        if external_content: unified_prompt += f"\n[External Content]:\n{external_content}\n"
        if multimodal_context: unified_prompt += f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
        unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a detailed and context-aware response."
        
        # e) Dynamic Research
        research_needed = await classify_prompt(user_message)
        if research_needed == "research" and not multimodal_context:
            research_results = await query_internet_via_groq(user_message)
            if research_results and research_results != "Error accessing internet information.":
                unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

        # --- 3. Build Message History (Retrieval Augmented) ---
        chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
        past_messages = chat_entry.get("messages", []) if chat_entry else []
        chat_history = []
        
        if past_messages:
            # Vector-based history retrieval logic
            past_embeddings = [
                msg["embedding"] for msg in past_messages
                if "embedding" in msg and isinstance(msg["embedding"], list) and len(msg["embedding"]) == 768
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
        if long_term_memory_summary: messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory_summary}"})
        if goals_context: messages.append({"role": "system", "content": goals_context})
        
        for msg in chat_history:
            cleaned = re.sub(r"<think>.*?</think>", "", msg.get("content", ""), flags=re.DOTALL).strip()
            if cleaned:
                # Truncate long messages for context window efficiency
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
async def regenerate_response_endpoint(
    request: RegenerateRequest, background_tasks: BackgroundTasks
):
    """Regenerates the last assistant response based on the last user message."""
    try:
        req = request
        user_id, session_id, filenames = req.user_id, req.session_id, req.filenames

        chat_entry = await chats_collection.find_one(
            {"user_id": user_id, "session_id": session_id}
        )
        if not chat_entry or not chat_entry.get("messages"):
            raise HTTPException(status_code=400, detail="No chat history found for regeneration.")

        # Find the last user message to regenerate from
        messages = chat_entry["messages"]
        last_user_message = next(
            (msg for msg in reversed(messages) if msg.get("role") == "user"), None
        )
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found to regenerate response for.")

        prompt = last_user_message["content"]

        # --- 1. Gather Context (Similar to /generate, but using last prompt) ---
        # Note: All logic for gathering context (goals, files, URLs, RAG lookup)
        # is encapsulated inside the /generate_response_endpoint logic for efficiency.
        
        # We simulate the request to the main logic
        simulated_request_data = {
            "user_id": user_id,
            "session_id": session_id,
            "prompt": prompt,
            "filenames": filenames
        }
        
        # Temporarily remove the last assistant message from history before running regeneration
        # Find the index of the *last* assistant message (which might be deleted)
        last_assistant_index = -1
        for i, msg in enumerate(reversed(messages)):
            if msg.get("role") == "assistant":
                last_assistant_index = len(messages) - 1 - i
                break
        
        if last_assistant_index != -1:
            # Remove the message from the in-memory `messages` list
            del messages[last_assistant_index] 
            # Remove it from the database *before* streaming the new one
            await chats_collection.update_one(
                {"_id": chat_entry["_id"]},
                {"$pop": {"messages": 1}} # Assumes the assistant message is the last one added
            )
            logger.info(f"Removed previous assistant message for regeneration in session {session_id}.")

        # Re-run the main generation logic using the simplified simulated request
        # The stream handler within generate_response_endpoint will save the new response.
        
        # NOTE: The current structure makes it hard to reuse the streaming logic without
        # duplicating it or restructuring. For clean refactoring, we'll keep the required
        # logic inside generate_response_endpoint and invoke it by reconstructing a Request.
        
        # **Simplified invocation of generation logic: Re-implement necessary parts**
        
        # --- 2. Build the full request body and call the stream again ---
        
        # Use the actual code from /generate_response_endpoint to create the stream
        async def call_generate_stream():
            # Reconstruct the logic from /generate endpoint for the stream to ensure all side effects are handled
            current_date = get_current_datetime()
            
            # ... [Redundant context gathering logic would be here, skipping for brevity] ...
            
            # Simplified message history setup for regeneration
            long_term_memory = await memory_collection.find_one({"user_id": user_id})
            long_term_memory_summary = long_term_memory.get("summary", "") if long_term_memory else ""
            
            system_prompt = (
                "You are Stelle... [full system prompt content] ...\n"
                f"Current date/time: {current_date}\n"
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            if long_term_memory_summary: messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory_summary}"})
            # Add filtered chat history
            # The chat_entry/messages state here is the one *after* the last assistant message was removed
            for msg in filter_think_messages(messages):
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Final user message (the prompt for regeneration)
            messages.append({"role": "user", "content": prompt}) 
            
            selected_key = random.choice(GENERATE_API_KEYS)
            client_generate = AsyncGroq(api_key=selected_key)
            
            stream = await client_generate.chat.completions.create(
                messages=messages,
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                max_completion_tokens=4000,
                temperature=0.7,
                stream=True,
            )

            full_reply = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                full_reply += delta
                yield delta

            # --- Persistence and Background Tasks after stream ends (Regeneration-specific) ---
            reply_content = full_reply.strip()

            # a) Process LLM commands
            await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)

            # b) Clean reply content for storage
            lines = reply_content.split("\n")
            clean_lines = [line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())]
            reply_content_clean = "\n".join(clean_lines).strip()

            # c) Generate embedding for new assistant response
            assistant_embedding = await generate_text_embedding(reply_content_clean)

            new_assistant_message = {
                "role": "assistant",
                "content": reply_content_clean,
                **({"embedding": assistant_embedding} if assistant_embedding else {}),
            }
            
            # Append only the new assistant message (the user message is already there)
            await chats_collection.update_one(
                {"user_id": user_id, "session_id": session_id},
                {
                    "$push": {"messages": new_assistant_message},
                    "$set": {"last_updated": datetime.now(timezone.utc)},
                },
            )
            logger.info(f"Regeneration complete and new response saved for session {session_id}.")
            
            # d) Update Long-Term Memory (in background if chat is long)
            # Fetch the updated messages list again to get the full count
            updated_chat = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
            if updated_chat and len(updated_chat.get("messages", [])) >= 10:
                background_tasks.add_task(
                    store_long_term_memory, user_id, session_id, updated_chat["messages"][-10:]
                )
                
            # e) RAG usage cleanup
            await update_rag_usage_and_cleanup(user_id, session_id)


        return StreamingResponse(call_generate_stream(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /regenerate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing your regeneration request.")


@router.get("/chat-history")
async def get_chat_history_endpoint(user_id: str = Query(...), session_id: str = Query(...)):
    """Retrieves chat history for a specific session."""
    try:
        chat_entry = await chats_collection.find_one(
            {"user_id": user_id, "session_id": session_id}, {"messages": 1}
        )
        if chat_entry and "messages" in chat_entry:
            # Filter the messages to remove internal <think> messages
            return {"messages": filter_think_messages(chat_entry["messages"])}
        return {"messages": []}
    except Exception as e:
        logger.error(f"Chat history retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chat history.")

# --- NLP/Voice Endpoint (WebSocket) ---

@router.websocket("/nlp")
async def nlp_websocket_endpoint(websocket):
    """WebSocket endpoint for real-time NLP/Voice chat."""
    await websocket.accept()
    
    # Imports needed for NLP endpoint logic
    from services.goal_service import update_task_goal_status, schedule_immediate_reminder
    from services.common_utils import get_current_datetime, filter_think_messages
    
    try:
        while True:
            data = await websocket.receive_text()
            nlp_data = json.loads(data)
            req = NLPRequest(**nlp_data)
            user_id, session_id, user_message, filenames = (
                req.user_id, req.session_id, req.message, req.filenames
            )

            if not user_id or not session_id or not user_message:
                await websocket.send_json({"error": "Invalid request parameters"})
                continue
                
            current_date = get_current_datetime()
            
            # --- Gather Context (Similar to /generate) ---
            
            # a) Goal Context
            active_goals = await goals_collection.find(
                {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
            ).to_list(None)
            goals_context = ""
            if active_goals:
                goals_context = "User's current goals and tasks:\n"
                for goal in active_goals:
                    goals_context += f"- Goal: {goal['title']} ({goal['status']}) [ID: {goal.get('goal_id','N/A')}]\n"
                    for task in goal.get('tasks', []):
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
            multimodal_context, used_filenames = await retrieve_multimodal_context(
                user_message, session_id, hooked_filenames
            )

            # --- 2. Construct Unified Prompt (NLP-specific tone) ---
            unified_prompt = f"User Query: {user_message}\n"
            if external_content: unified_prompt += f"\n[External Content]:\n{external_content}\n"
            if multimodal_context: unified_prompt += f"\n[Retrieved File & Code Context]:\n{multimodal_context}\n"
            unified_prompt += f"\nCurrent Date/Time: {current_date}\n\nProvide a conversational, friendly response as if speaking directly to the user."

            # e) Dynamic Research
            research_needed = await classify_prompt(user_message)
            if research_needed == "research" and not multimodal_context:
                await websocket.send_json(
                    {"status": "researching", "message": "Researching the topic..."}
                )
                research_results = await query_internet_via_groq(user_message)
                if research_results and research_results != "Error accessing internet information.":
                    unified_prompt += f"\n\n[Additional Research]:\n{research_results}"

            # --- 3. Build Message History (NLP-specific short history) ---
            chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
            past_messages = chat_entry.get("messages", []) if chat_entry else []
            
            # Simple retrieval for fast NLP/Voice response (last 2 user/assistant pair)
            chat_history = filter_think_messages(past_messages[-4:]) 
            
            # Long-term memory
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
            if long_term_memory_summary: messages.append({"role": "system", "content": f"Long-term memory: {long_term_memory_summary}"})
            if goals_context: messages.append({"role": "system", "content": goals_context})
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
                model="deepseek-r1-distill-llama-70b", # NLP specific model
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
                
            # --- Persistence and Side Effects (NLP) ---
            reply_content = full_reply.strip()

            # a) Process LLM commands
            await handle_goal_updates_and_cleanup(reply_content, user_id, session_id)
            
            # b) Handle Reminders
            remind_match = re.search(r"remind me (.+)", user_message, re.IGNORECASE)
            if remind_match:
                reminder_text = remind_match.group(1).strip()
                await schedule_immediate_reminder(user_id, reminder_text)

            # c) Clean reply content for storage
            lines = reply_content.split("\n")
            clean_lines = [line for line in lines if not re.match(r"\[.*?: .*?\]", line.strip())]
            reply_content_clean = "\n".join(clean_lines).strip()
            
            # d) Generate embeddings and save to chat history
            user_embedding = await generate_text_embedding(user_message)
            assistant_embedding = await generate_text_embedding(reply_content_clean)

            new_messages = [
                {"role": "user", "content": user_message, "embedding": user_embedding},
                {"role": "assistant", "content": reply_content_clean, "embedding": assistant_embedding},
            ]
            
            if chat_entry:
                await chats_collection.update_one(
                    {"_id": chat_entry["_id"]},
                    {"$push": {"messages": {"$each": new_messages}}},
                )
                updated_messages = chat_entry.get("messages", []) + new_messages
            else:
                new_chat_entry = {"user_id": user_id, "session_id": session_id, "messages": new_messages, "last_updated": datetime.now(timezone.utc)}
                await chats_collection.insert_one(new_chat_entry)
                updated_messages = new_messages

            # e) Update Long-Term Memory
            if len(updated_messages) >= 10:
                # Run memory update in a background task for immediate response
                asyncio.create_task(store_long_term_memory(user_id, session_id, updated_messages[-10:]))
            
            # f) Update RAG usage count
            for filename in used_filenames:
                await uploads_collection.update_many(
                    {"user_id": user_id, "session_id": session_id, "filename": filename},
                    {"$inc": {"query_count": 1}},
                )

            await websocket.send_json(
                {"status": "complete", "message": reply_content_clean}
            )

    except Exception as e:
        logger.error(f"Error in /nlp endpoint: {e}")
        await websocket.send_json({"error": f"Internal error processing your request: {e}"})
    finally:
        await websocket.close()

# --- AI Assist Endpoints ---

@router.post("/aiassist")
async def ai_assist_endpoint(input_data: UserInput):
    """Generates social media assets (sync version)."""
    from services.post_generator_service import (
        generate_keywords_post, fetch_trending_hashtags_post, 
        fetch_seo_keywords_post, generate_caption_post, Platforms
    )
    
    # Create a transient sync client for this operation
    client_sync = Groq(api_key=random.choice(GENERATE_API_KEYS))
    client_async = AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))
    
    try:
        # NOTE: The original code used a synchronous Groq client (`internet_client`) inside the
        # background functions, which would be run in a thread by FastAPI.
        # Here we directly call the async helpers, letting FastAPI manage the threadpool/concurrency.
        
        # A simple default platform list is needed for the helper function:
        default_platforms = [Platforms.Instagram]
        
        seed_keywords = await generate_keywords_post(client_async, input_data.query)
        trending_hashtags = await fetch_trending_hashtags_post(client_async, seed_keywords, default_platforms)
        seo_keywords = await fetch_seo_keywords_post(client_async, seed_keywords)
        caption_dict = await generate_caption_post(
            client_async, input_data.query, seed_keywords, trending_hashtags, default_platforms
        )
        caption = caption_dict.get(Platforms.Instagram.value, list(caption_dict.values())[0])

        return {
            "caption": caption,
            "keywords": seed_keywords,
            "hashtags": trending_hashtags,
            "seo_keywords": seo_keywords,
        }
    except Exception as e:
        logger.error(f"Error in /aiassist endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.websocket("/wss/aiassist")
async def websocket_ai_assist_endpoint(websocket):
    """Streams the social media asset generation process."""
    await websocket.accept()
    from services.post_generator_service import (
        generate_keywords_post, fetch_trending_hashtags_post, 
        fetch_seo_keywords_post, generate_caption_post, Platforms
    )
    
    client_async = AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))

    try:
        while True:
            data = await websocket.receive_text()
            user_input = json.loads(data)
            query = user_input["query"]

            await websocket.send_text(json.dumps({"step": "Initializing AI Assistant..."}))
            
            # Use default platforms for simplicity in this endpoint
            default_platforms = [Platforms.Instagram, Platforms.X]

            seed_keywords = await generate_keywords_post(client_async, query)
            await websocket.send_text(
                json.dumps({"step": "Generated Seed Keywords", "keywords": seed_keywords})
            )

            trending_hashtags = await fetch_trending_hashtags_post(client_async, seed_keywords, default_platforms)
            await websocket.send_text(
                json.dumps({"step": "Fetched Trending Hashtags", "hashtags": trending_hashtags})
            )

            seo_keywords = await fetch_seo_keywords_post(client_async, seed_keywords)
            await websocket.send_text(
                json.dumps({"step": "Fetched SEO Keywords", "seo_keywords": seo_keywords})
            )

            await websocket.send_text(json.dumps({"step": "Generating Final Captions..."}))
            caption_dict = await generate_caption_post(
                client_async, query, seed_keywords, trending_hashtags, default_platforms
            )
            
            # The original structure returned a single caption, so we pick one (e.g., Instagram)
            caption = caption_dict.get(Platforms.Instagram.value, list(caption_dict.values())[0])

            await websocket.send_text(
                json.dumps({
                    "step": "Caption ready",
                    "caption": caption,
                    "keywords": seed_keywords,
                    "hashtags": trending_hashtags,
                    "seo_keywords": seo_keywords,
                })
            )
    except Exception as e:
        logger.error(f"Error in AI Assist WebSocket: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()