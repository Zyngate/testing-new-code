# stelle_backend/routes/misc_routes.py
import json
import uuid
import asyncio
from typing import List, Dict, Any, Union

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from models.common_models import ResearchQuery, DeepSearchRequest
from services.ai_service import (
    generate_subqueries, query_internet_via_groq, synthesize_result, 
    clarify_query, clarify_response, generate_keywords, understanding_query
)
from services.common_utils import logger, filter_think_messages, websocket_queries, deepsearch_queries
from database import chats_collection
from services.ai_service import (
    retrieve_multimodal_context, generate_text_embedding, store_long_term_memory
)
from datetime import datetime, timezone
import numpy as np

router = APIRouter()

# --- Deep Search & Research Utility Functions ---

async def deep_search_process(query_data: Dict[str, Any], websocket: WebSocket):
    """The main WebSocket-streaming logic for the deep search pipeline."""
    user_id = query_data.get("user_id", "unknown")
    session_id = query_data.get("session_id", "unknown")
    main_query = query_data.get("prompt", "No query provided")
    filenames = query_data.get("filenames", [])

    try:
        await websocket.send_json({"step": "start", "message": "Starting deep search..."})

        # 1. Fetch memory summary (for context)
        # Note: This is an optional step, kept outside the loop for simplicity.
        # It's primarily used for context, but not streamed as a distinct step in the original code.

        # 2. Clarify query and pre-response
        clarified_query = await clarify_query(main_query)
        clarifyd_response = await clarify_response(main_query)
        await websocket.send_json({
            "step": "clarified_query",
            "message": f"Clarified response: {clarifyd_response}",
        })

        # 3. Generate keywords and understanding
        keywords = await generate_keywords(clarified_query)
        understanding = await understanding_query(clarified_query)
        await websocket.send_json({
            "step": "keywords_generated",
            "message": f"Processing request: {understanding}",
        })

        all_responses = []
        all_sources = []

        # 4. Execute search for each keyword
        for keyword in keywords:
            try:
                response, sources = await query_internet_via_groq(
                    f"Provide information on {keyword}", return_sources=True
                )
                if response and response != "Error accessing internet information.":
                    all_responses.append(response)
                    all_sources.extend(sources)
                    await websocket.send_json({
                        "step": "response_received",
                        "keyword": keyword,
                        "response": response[:200] + "..." if len(response) > 200 else response,
                        "sources": [source.get("url", "N/A") for source in sources],
                    })
                else:
                    all_responses.append(f"No real data found for {keyword}, using fallback info.")
            except:
                all_responses.append(f"Failed to fetch info for {keyword}, using fallback info.")

        if not all_responses:
            all_responses = [f"No information retrieved. Using dummy response for query: {main_query}"]

        # 5. Synthesize final answer (using the synthesizer function)
        final_answer = await synthesize_result(main_query, all_responses)

        # 6. Final streaming response
        unique_sources = list({source.get("url", "N/A"): source for source in all_sources}.keys())

        await websocket.send_json({
            "step": "final_answer",
            "message": final_answer,
            "sources": unique_sources,
        })
        await websocket.send_json({"step": "end", "message": "Deep search complete!"})

        return final_answer

    except Exception as e:
        logger.error(f"Deep search error: {e}")
        await websocket.send_json({"step": "error", "message": "An unexpected error occurred."})
        return f"Error during deep search. Returning fallback answer for query: {main_query}"


async def research_process(main_query: str, websocket: WebSocket):
    """The main WebSocket-streaming logic for the general research pipeline."""
    try:
        await websocket.send_json({"step": "start", "message": "Starting research..."})
        await websocket.send_json(
            {"step": "generating_subqueries", "message": "Generating subqueries..."}
        )
        
        # 1. Generate Subqueries
        subqueries = await generate_subqueries(main_query)
        if not subqueries:
            await websocket.send_json({"step": "error", "message": "Failed to generate subqueries."})
            return
        await websocket.send_json({"step": "subqueries_generated", "subqueries": subqueries})
        
        # 2. Execute Queries
        all_contents = []
        for subquery in subqueries:
            await websocket.send_json(
                {"step": "querying", "subquery": subquery, "message": f"Querying for '{subquery}'..."}
            )
            content = await query_internet_via_groq(subquery)
            if content and content != "Error accessing internet information.":
                all_contents.append(content)
                await websocket.send_json(
                    {
                        "step": "response_received",
                        "subquery": subquery,
                        "response": (content[:200] + "..." if len(content) > 200 else content),
                    }
                )
                
        if not all_contents:
            await websocket.send_json({"step": "no_content", "message": "No content retrieved from any subquery."})
            return
            
        # 3. Synthesize Result
        await websocket.send_json({"step": "synthesizing", "message": "Synthesizing final result..."})
        final_result = await synthesize_result(main_query, all_contents)
        
        # 4. Final streaming response
        await websocket.send_json({"step": "final_result", "result": final_result})
        await websocket.send_json({"step": "end", "message": "Research complete."})
        
    except WebSocketDisconnect:
        logger.info("Client disconnected during research process.")
    except Exception as e:
        logger.error(f"Research process error: {e}")
        await websocket.send_json({"step": "error", "message": f"Unexpected error: {str(e)}"})


# --- Endpoints ---

@router.get("/history")
async def get_history_endpoint(user_id: str = Query(...)):
    """Retrieves all chat sessions for a user."""
    try:
        sessions = await chats_collection.find({"user_id": user_id}).to_list(None)
        history = []
        for session in sessions:
            session_id = session.get("session_id")
            messages = session.get("messages", [])
            time = session.get("last_updated")
            filtered_messages = filter_think_messages(messages)
            first_message = filtered_messages[0]["content"] if filtered_messages else ""
            history.append(
                {"session_id": session_id, "first_message": first_message, "time": time}
            )
        return {"history": history}
    except Exception as e:
        logger.error(f"Error retrieving session history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving session history.")


@router.post("/start_research")
async def start_research_endpoint(query: ResearchQuery):
    """Initializes the general research process and returns a query ID for the WebSocket."""
    query_id = str(uuid.uuid4())
    websocket_queries[query_id] = query.text
    logger.info(f"Received research request: {query.text} (ID: {query_id})")
    return {"query_id": query_id}


@router.websocket("/ws/{query_id}")
async def websocket_endpoint_research(websocket: WebSocket, query_id: str):
    """WebSocket endpoint to stream the general research process steps."""
    await websocket.accept()
    try:
        main_query = websocket_queries.get(query_id)
        if not main_query:
            await websocket.send_json({"step": "error", "message": "Invalid query ID."})
            return
        await research_process(main_query, websocket)
    except WebSocketDisconnect:
        logger.info(f"WebSocket closed for query ID: {query_id}")
    finally:
        websocket_queries.pop(query_id, None)
        await websocket.close()


@router.post("/start_deepsearch")
async def start_deepsearch_endpoint(request: DeepSearchRequest):
    """Initializes the deep search process and returns a query ID for the WebSocket."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    query_id = str(uuid.uuid4())
    deepsearch_queries[query_id] = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "prompt": request.prompt,
        "filenames": request.filenames,
    }
    logger.info(f"Deep search initiated: {query_id}, prompt: {request.prompt}")
    return {"query_id": query_id}


@router.websocket("/ws/deepsearch/{query_id}")
async def deepsearch_websocket_endpoint(websocket: WebSocket, query_id: str):
    """WebSocket endpoint to stream the deep search process steps."""
    await websocket.accept()
    try:
        query_data = deepsearch_queries.get(query_id)
        if not query_data:
            await websocket.send_json({"step": "error", "message": "Invalid query ID"})
            return

        user_id = query_data["user_id"]
        session_id = query_data["session_id"]
        user_message = query_data["prompt"]
        filenames = query_data["filenames"]

        # Check for RAG context first (if files are attached)
        multimodal_context, _ = await retrieve_multimodal_context(
            user_message, session_id, filenames
        )
        # Note: Original logic forced deep search regardless of file presence if a flag was set.
        # Here, we assume Deep Search is a required step, unless RAG already has the answer.
        # Since the flag for `research_needed` is not explicitly calculated here, we proceed with research.
        research_needed = "research"

        if research_needed == "research" and not multimodal_context:
            final_answer = await deep_search_process(query_data, websocket)
            
            # Save Deep Search result to chat history
            user_embedding = await generate_text_embedding(user_message)
            assistant_embedding = await generate_text_embedding(final_answer)
            new_messages = [
                {"role": "user", "content": user_message, "embedding": user_embedding},
                {"role": "assistant", "content": final_answer, "embedding": assistant_embedding},
            ]
            
            chat_entry = await chats_collection.find_one(
                {"user_id": user_id, "session_id": session_id}
            )
            
            # The original logic inside /generate handles the MongoDB update and memory update:
            update_fields = {"$push": {"messages": {"$each": new_messages}}, "$set": {"last_updated": datetime.now(timezone.utc)}}
            if chat_entry:
                await chats_collection.update_one({"_id": chat_entry["_id"]}, update_fields)
                updated_messages = chat_entry.get("messages", []) + new_messages
            else:
                new_chat_entry = {"user_id": user_id, "session_id": session_id, "messages": new_messages, "last_updated": datetime.now(timezone.utc)}
                await chats_collection.insert_one(new_chat_entry)
                updated_messages = new_messages
                
            if len(updated_messages) >= 10:
                asyncio.create_task(store_long_term_memory(user_id, session_id, updated_messages[-10:]))
                
        else:
            await websocket.send_json(
                {"step": "standard_response", "message": "Context found in files. Using standard generation..."}
            )
            # In a real app, you'd trigger a /generate call here. For this structure, we just signal end.
            await websocket.send_json({"step": "end", "message": "Response will be generated via standard chat."})
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {query_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"step": "error", "message": "Server error occurred"})
    finally:
        deepsearch_queries.pop(query_id, None)
        await websocket.close()