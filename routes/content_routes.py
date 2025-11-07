# stelle_backend/routes/content_routes.py
import uuid
import json
import asyncio
import base64
from datetime import datetime, timezone 
from typing import List, Dict, Any # <-- CRITICAL: Dict, Any, List needed for routing fix

from fastapi import APIRouter, HTTPException, Form, File, WebSocket, WebSocketDisconnect, UploadFile # <-- ADDED UploadFile
from fastapi.responses import JSONResponse
from groq import AsyncGroq, Groq 

from models.common_models import Prompt
from database import chats_collection
from services.ai_service import (
    query_internet_via_groq, 
    generate_text_embedding
)
from services.file_service import process_and_index_file
from services.content_service import ( 
    visual_generate_subqueries, 
    visual_synthesize_result,
    generate_html_visualization
) 
from services.common_utils import logger, websocket_queries


router = APIRouter()

# --- Utility Functions for Visualization ---

async def visualization_process(main_query: str, websocket: WebSocket):
    """The main WebSocket-streaming logic for visualization generation."""
    try:
        await websocket.send_json(
            {"step": "start", "message": "Starting visualization research..."}
        )

        await websocket.send_json(
            {"step": "generating_subqueries", "message": "Generating subqueries..."}
        )
        subqueries = await visual_generate_subqueries(main_query)
        if not subqueries:
            await websocket.send_json(
                {"step": "error", "message": "Failed to generate subqueries."}
            )
            return
        await websocket.send_json(
            {"step": "subqueries_generated", "subqueries": subqueries}
        )

        all_contents = []
        for subquery in subqueries:
            await websocket.send_json(
                {
                    "step": "searching",
                    "subquery": subquery,
                    "message": f"Searching for '{subquery}'...",
                }
            )
            content = await query_internet_via_groq(f"Provide information on {subquery}")
            if content and content != "Error accessing internet information.":
                all_contents.append(content)
                await websocket.send_json(
                    {
                        "step": "content_received",
                        "subquery": subquery,
                        "content": (
                            content[:200] + "..." if len(content) > 200 else content
                        ),
                    }
                )

        if not all_contents:
            await websocket.send_json(
                {
                    "step": "no_content",
                    "message": "No content retrieved from any subquery.",
                }
            )
            return

        await websocket.send_json(
            {"step": "synthesizing", "message": "Synthesizing research result..."}
        )
        synthesized = await visual_synthesize_result(main_query, all_contents)
        await websocket.send_json({"step": "synthesized", "result": synthesized})

        await websocket.send_json(
            {"step": "generating_html", "message": "Generating HTML visualization..."}
        )
        html_code = await generate_html_visualization(synthesized)
        await websocket.send_json({"step": "html_generated", "html": html_code})

        await websocket.send_json({"step": "end", "message": "Visualization complete."})
    except WebSocketDisconnect:
        logger.info("Client disconnected during visualization process.")
    except Exception as e:
        logger.error(f"Visualization process error: {e}")
        await websocket.send_json(
            {"step": "error", "message": f"Unexpected error: {str(e)}"}
        )
    finally:
        websocket_queries.pop(websocket.query_params.get("query_id"), None)


# --- Endpoints ---

# FIX: Added response_model=Dict[str, Any] to resolve the FastAPIError
@router.post("/upload", response_model=Dict[str, Any])
async def upload_file_endpoint(
    user_id: str = Form(...),
    session_id: str = Form(...),
    # CRITICAL FIX: Use List[UploadFile] to satisfy type checker and runtime
    files: List[UploadFile] = File(...), 
):
    """Handles file uploads, extracts text/code, generates embeddings, and indexes in FAISS."""
    responses = []
    
    tasks = [
        process_and_index_file(user_id, session_id, file)
        for file in files
    ]
    responses = await asyncio.gather(*tasks)

    return {"results": responses}


@router.post("/start_visualization")
async def start_visualization_endpoint(prompt: Prompt):
    """Initializes the visualization process and returns a query ID for the WebSocket."""
    visual_query_id = str(uuid.uuid4())
    websocket_queries[visual_query_id] = prompt.text
    logger.info(f"Received visualization prompt: {prompt.text} (ID: {visual_query_id})")
    return {"query_id": visual_query_id}


@router.websocket("/ws/visualization/{query_id}")
async def websocket_endpoint_visualization(websocket: WebSocket, query_id: str):
    """WebSocket endpoint to stream the visualization generation steps."""
    await websocket.accept()
    try:
        main_query = websocket_queries.get(query_id)
        if not main_query:
            await websocket.send_json({"step": "error", "message": "Invalid query ID."})
            return
        await visualization_process(main_query, websocket)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for query ID: {query_id}")
    finally:
        await websocket.close()


# FIX: Added response_model=Dict[str, Any] for analyze-images endpoint
@router.post("/analyze-images", response_model=Dict[str, Any])
async def analyze_images_endpoint(
    user_id: str = Form(...),
    session_id: str = Form(...),
    query: str = Form(...),
    # CRITICAL FIX: Use List[UploadFile] here too
    images: List[UploadFile] = File(...),
):
    """Analyzes uploaded images based on a query using a multi-modal model."""
    from services.ai_service import get_groq_client
    
    if not 0 < len(images) <= 3:
        return {"error": "Please upload 1 to 3 images."}

    user_message_text = f"Query regarding images: {query}"
    user_embedding = await generate_text_embedding(user_message_text)

    content = [{"type": "text", "text": query}]
    for image in images:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{image.content_type};base64,{base64_image}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    # Use AsyncGroq client
    client_async = await get_groq_client()
    response = await client_async.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {"role": "system", "content": "Return response in plain english. Do not use LaTeX"},
            {"role": "user", "content": content},
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
    )
    output = response.choices[0].message.content

    assistant_embedding = await generate_text_embedding(output)

    new_messages = [
        {"role": "user", "content": user_message_text, "embedding": user_embedding},
        {"role": "assistant", "content": output, "embedding": assistant_embedding},
    ]

    chat_entry = await chats_collection.find_one(
        {"user_id": user_id, "session_id": session_id}
    )
    
    update_fields = {
        "$push": {"messages": {"$each": new_messages}},
        "$set": {"last_updated": datetime.now(timezone.utc)},
    }
    
    if chat_entry:
        await chats_collection.update_one(
            {"_id": chat_entry["_id"]},
            update_fields
        )
    else:
        await chats_collection.insert_one(
            {
                "user_id": user_id, "session_id": session_id, "messages": new_messages,
                "last_updated": datetime.now(timezone.utc),
            }
        )

    updated_chat_entry = await chats_collection.find_one({"user_id": user_id, "session_id": session_id})
    if updated_chat_entry and len(updated_chat_entry.get("messages", [])) >= 10:
        from services.ai_service import store_long_term_memory
        asyncio.create_task(
            store_long_term_memory(user_id, session_id, updated_chat_entry["messages"][-10:])
        )

    return {"output": output}