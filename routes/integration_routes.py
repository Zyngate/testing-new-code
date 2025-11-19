# stelle_backend/routes/integration_routes.py

import asyncio
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, WebSocket

from services.common_utils import logger
from services.ai_service import get_groq_client
from services.post_generator_service import (
    generate_keywords_post,
    generate_caption_post
)

router = APIRouter(tags=["Integrations"])

# -------------------------------------------------------
# REST Caption Generator (Swagger UI)
# -------------------------------------------------------
@router.post("/generate_caption")
async def generate_caption_endpoint(body: dict):
    try:
        prompt = body.get("prompt", "")
        platforms: List[str] = body.get("platforms", [])

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required.")
        if not platforms:
            raise HTTPException(status_code=400, detail="Please select platforms.")

        client = await get_groq_client()

        # 1. keywords
        keywords = await generate_keywords_post(client, prompt)

        # 2. captions + hashtags
        results = await generate_caption_post(
            query=prompt,
            seed_keywords=keywords,
            platforms=platforms
        )

        return {
            "status": "success",
            "keywords": keywords,
            "captions": results["captions"],
            "hashtags": results["platform_hashtags"]
        }

    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Caption generation failed.")


# -------------------------------------------------------
# WebSocket Caption Generator
# -------------------------------------------------------
@router.websocket("/wss/generate-caption")
async def websocket_generate_caption(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "Ready."})

    try:
        # 1. Receive platforms
        platforms_text = await websocket.receive_text()
        platforms = [p.strip() for p in platforms_text.split(",")]

        # 2. Receive prompt
        prompt = await websocket.receive_text()

        client = await get_groq_client()

        await websocket.send_json({"status": "processing", "message": "Generating keywords..."})
        keywords = await generate_keywords_post(client, prompt)

        await websocket.send_json({"status": "processing", "message": "Generating captions..."})
        results = await generate_caption_post(prompt, keywords, platforms)

        await websocket.send_json({
            "status": "completed",
            "keywords": keywords,
            "captions": results["captions"],
            "hashtags": results["platform_hashtags"]
        })

    except Exception as e:
        logger.error(f"WebSocket Error: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": "Caption generation failed."})

    finally:
        await websocket.close()
