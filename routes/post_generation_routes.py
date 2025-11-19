# stelle_backend/routes/post_generation_routes.py

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.ai_service import get_groq_client
from services.post_generator_service import (
    generate_keywords_post,
    generate_caption_post,
    fetch_platform_hashtags,
)
from config import logger

router = APIRouter(tags=["Post Generator"])

# ======================================================
# WEBSOCKET: /wss/generate-post 
# (Matches EXACT frontend logic)
# ======================================================

@router.websocket("/wss/generate-post")
async def websocket_generate_post(websocket: WebSocket, post_option: str):
    await websocket.accept()

    try:
        # 1️⃣ Receive platform list
        raw_platforms = await websocket.receive_text()
        platforms = [p.strip().lower() for p in raw_platforms.split(",") if p.strip()]

        # 2️⃣ Receive content
        content = await websocket.receive_text()

        await websocket.send_json({"status": "processing", "message": "Classifying & extracting keywords..."})

        client_async = await get_groq_client()

        # keywords
        keywords = await generate_keywords_post(client_async, content)

        await websocket.send_json({
            "status": "processing",
            "message": "Generating platform-specific hashtags..."
        })

        # platform hashtags
        platform_hashtags = {}
        for p in platforms:
            platform_hashtags[p] = await fetch_platform_hashtags(None, keywords, p)

        await websocket.send_json({
            "status": "processing",
            "message": "Generating captions..."
        })

        # captions
        captions_output = await generate_caption_post(content, keywords, platforms)

        await websocket.send_json({
            "status": "completed",
            "message": "Post generated successfully!",
            "keywords": keywords,
            "captions": captions_output["captions"],
            "platform_hashtags": captions_output["platform_hashtags"]
        })

    except WebSocketDisconnect:
        logger.warning("Post generator WebSocket disconnected")

    except Exception as e:
        logger.error(f"Post generation error: {e}")
        await websocket.send_json({"status": "error", "message": "Post generation failed."})

    finally:
        await websocket.close()
