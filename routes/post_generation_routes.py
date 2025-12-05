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

async def fetch_seo_keywords_post(*args, **kwargs):
    return []


@router.websocket("/wss/generate-post")
async def websocket_generate_post(websocket: WebSocket, post_option: str = "normal"):
    await websocket.accept()
    try:
        # 1) receive the selected platforms (CSV string)
        raw_platforms = await websocket.receive_text()
        platforms = [p.strip().lower() for p in raw_platforms.split(",") if p.strip()]

        # 2) receive the content (text or description)
        content = await websocket.receive_text()

        await websocket.send_json({"status": "processing", "message": "Generating keywords..."})

        client_async = await get_groq_client()

        # keywords
        keywords = await generate_keywords_post(client_async, content)

        await websocket.send_json({"status": "processing", "message": "Generating platform hashtags..."})

        # platform hashtags (uses query too)
        platform_hashtags = {}
        for p in platforms:
            platform_hashtags[p] = await fetch_platform_hashtags(None, keywords, p, content)

        await websocket.send_json({"status": "processing", "message": "Generating captions..."})

        # captions
        results = await generate_caption_post(content, keywords, platforms)

        await websocket.send_json({
            "status": "completed",
            "message": "Post generated successfully",
            "keywords": keywords,
            "captions": results["captions"],
            "platform_hashtags": results["platform_hashtags"]
        })

    except WebSocketDisconnect:
        logger.info("Post generator socket disconnected by client")
    except Exception as e:
        logger.exception(f"Post generation error: {e}")
        await websocket.send_json({"status": "error", "message": "Post generation failed."})
    finally:
        await websocket.close()
