# stelle_backend/routes/integration_routes.py

import json
import asyncio
import time
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from models.common_models import ScrapeRequest
from services.common_utils import logger
from services.ai_service import get_groq_client
from services.post_generator_service import (
    classify_post_type,
    generate_keywords_post,
    fetch_trending_hashtags_post,
    fetch_seo_keywords_post,
    generate_caption_post
)

router = APIRouter(tags=["Integrations"])


# ====================================================================
# A. LINKEDIN SCRAPING ENDPOINT
# ====================================================================

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
        scrape_result = await asyncio.to_thread(
            run_linkedin_scrape_sync,
            request.query,
            request.max_results
        )
        return scrape_result

    except Exception as e:
        logger.error(f"Error in /scrape_linkedin endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


# ====================================================================
# B. TEXT-ONLY CAPTION GENERATOR (WEBSOCKET)
# ====================================================================

@router.websocket("/wss/generate-caption")
async def websocket_generate_caption(websocket: WebSocket):
    await websocket.accept()

    await websocket.send_json({
        "status": "connected",
        "message": "Connection established for caption generation."
    })

    try:
        # User sends only the TEXT prompt (no platform selection here)
        prompt = await websocket.receive_text()
        logger.info(f"Prompt received: {prompt}")

        # Async Groq client
        client_async = await get_groq_client()

        # Classification
        await websocket.send_json({"status": "processing", "message": "Classifying post type..."})
        post_type = await classify_post_type(client_async, prompt)

        # Keywords
        await websocket.send_json({"status": "processing", "message": "Generating keywords..."})
        seed_keywords = await generate_keywords_post(client_async, prompt)

        # Trending hashtags
        await websocket.send_json({"status": "processing", "message": "Fetching trending hashtags..."})
        trending_hashtags = await fetch_trending_hashtags_post(client_async, seed_keywords, [])

        # SEO keywords
        await websocket.send_json({"status": "processing", "message": "Fetching SEO keywords..."})
        seo_keywords = await fetch_seo_keywords_post(client_async, seed_keywords)

        # Captions — NEW FORMAT
        await websocket.send_json({"status": "processing", "message": "Generating captions..."})

        caption_result = await generate_caption_post(
            query=prompt,
            seed_keywords=seed_keywords,
            hashtags=trending_hashtags,
            platforms=["instagram"]       # default platform
        )

        # Send Final Response
        await websocket.send_json({
            "status": "completed",
            "message": "Caption generation finished!",
            "post_type": post_type,
            "keywords": seed_keywords,
            "trending_hashtags": trending_hashtags,
            "seo_keywords": seo_keywords,
            "captions": caption_result["captions"]
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected from /wss/generate-caption")

    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": f"Critical error: {str(e)}"})

    finally:
        await websocket.close()
        logger.info("WebSocket connection closed for /wss/generate-caption")
