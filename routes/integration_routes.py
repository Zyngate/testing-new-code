# stelle_backend/routes/integration_routes.py
import json
import asyncio
import time
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from models.common_models import ScrapeRequest, PostGenOptions
from services.common_utils import logger
from services.ai_service import get_groq_client
from services.post_generator_service import (
    classify_post_type,
    generate_keywords_post,
    fetch_trending_hashtags_post,
    fetch_seo_keywords_post,
    generate_caption_post,
    Platforms,
)

router = APIRouter(tags=["Integrations"])

# -----------------------------
# A. LINKEDIN SCRAPING
# -----------------------------
def run_linkedin_scrape_sync(query: str, max_results: int) -> Dict[str, Any]:
    logger.info(f"Starting simulated LinkedIn scrape for query: {query}")
    time.sleep(1)
    return {
        "status": "success",
        "query": query,
        "results_count": max_results,
        "data_summary": f"Retrieved summary for {query}'s profile.",
        "note": "Actual scraping logic runs synchronously here."
    }

@router.post("/scrape_linkedin", response_model=Dict[str, Any])
async def scrape_linkedin_endpoint(request: ScrapeRequest):
    try:
        result = await asyncio.to_thread(run_linkedin_scrape_sync, request.query, request.max_results)
        return result
    except Exception as e:
        logger.error(f"Error in /scrape_linkedin endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


# -----------------------------
# B. CAPTION GENERATION (Text-only)
# -----------------------------
@router.websocket("/wss/generate-caption")
async def websocket_generate_caption(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "Connection established for caption generation."})

    try:
        # 1️⃣ Receive platform options
        platform_str = await websocket.receive_text()
        platform_indices = platform_str.split(',')
        platform_options = []

        for x in platform_indices:
            if x.isdigit() and int(x) < len(Platforms.platform_list):
                item = Platforms.platform_list[int(x)]
                if isinstance(item, str):
                    try:
                        item = Platforms(item)
                    except Exception as e:
                        logger.error(f"Invalid platform: {item}, skipping. Error: {e}")
                        continue
                platform_options.append(item)

        if not platform_options:
            await websocket.send_json({"status": "error", "message": "No valid platform options selected"})
            return

        logger.info(f"Platform options: {platform_options}")

        # 2️⃣ Receive prompt
        prompt = await websocket.receive_text()
        logger.info(f"Prompt received: {prompt}")

        # 3️⃣ Classification
        await websocket.send_json({"status": "processing", "message": "Classifying post type..."})
        client_async = await get_groq_client()
        post_type = await classify_post_type(client_async, prompt)

        # 4️⃣ Generate keywords
        await websocket.send_json({"status": "processing", "message": "Generating keywords..."})
        seed_keywords = await generate_keywords_post(client_async, prompt)

        # 5️⃣ Fetch trending hashtags
        await websocket.send_json({"status": "processing", "message": "Fetching trending hashtags..."})
        trending_hashtags = await fetch_trending_hashtags_post(client_async, seed_keywords, platform_options)

        # 6️⃣ Fetch SEO keywords
        await websocket.send_json({"status": "processing", "message": "Fetching SEO keywords..."})
        seo_keywords = await fetch_seo_keywords_post(client_async, seed_keywords)

        # 7️⃣ Generate captions
        await websocket.send_json({"status": "processing", "message": "Generating captions..."})
        captions = await generate_caption_post(client_async, prompt, seed_keywords, trending_hashtags, platform_options)

        # 8️⃣ Final output
        await websocket.send_json({
            "status": "completed",
            "message": "Caption generation finished!",
            "captions": captions,
            "trending_hashtags": trending_hashtags,
            "seo_keywords": seo_keywords,
            "post_type": post_type
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected from /wss/generate-caption")
    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": f"Critical error: {str(e)}"})
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed for /wss/generate-caption")
