# stelle_backend/routes/integration_routes.py

import asyncio
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

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
    logger.info(f"Starting simulated LinkedIn scrape for: {query}")
    import time
    time.sleep(1)

    return {
        "status": "success",
        "query": query,
        "results_count": max_results,
        "data_summary": f"Scraped summary for {query} (simulated)."
    }

@router.post("/scrape_linkedin", response_model=Dict[str, Any])
async def scrape_linkedin_endpoint(request: ScrapeRequest):
    try:
        result = await asyncio.to_thread(
            run_linkedin_scrape_sync,
            request.query,
            request.max_results
        )
        return result
    except Exception as e:
        logger.error(f"LinkedIn scrape failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="LinkedIn scrape failed.")


# ====================================================================
# B. TEXT-ONLY CAPTION GENERATOR (REST ENDPOINT)
# ====================================================================

@router.post("/generate_caption")
async def generate_caption_endpoint(body: dict):
    """
    REST-based caption generator (shows in Swagger).
    """
    try:
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required.")

        client_async = await get_groq_client()

        # 1. post classification
        post_type = await classify_post_type(client_async, prompt)

        # 2. keywords
        keywords = await generate_keywords_post(client_async, prompt)

        # 3. trending hashtags
        hashtags = await fetch_trending_hashtags_post(client_async, keywords, [])

        # 4. SEO terms
        seo_keywords = await fetch_seo_keywords_post(client_async, keywords)

        # 5. Captions for ALL platforms
        platforms = ["instagram", "linkedin", "facebook", "twitter", "reddit"]

        captions = await generate_caption_post(
            query=prompt,
            seed_keywords=keywords,
            hashtags=hashtags,
            platforms=platforms
        )

        return {
            "status": "success",
            "post_type": post_type,
            "keywords": keywords,
            "seo_keywords": seo_keywords,
            "hashtags": hashtags,
            "captions": captions["captions"]
        }

    except Exception as e:
        logger.error(f"Caption generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Caption generation failed.")


# ====================================================================
# C. CAPTION GENERATION (WEBSOCKET)
# ====================================================================

@router.websocket("/wss/generate-caption")
async def websocket_generate_caption(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "Caption generation initialized."})

    try:
        # Receive prompt from frontend
        prompt = await websocket.receive_text()

        client_async = await get_groq_client()

        await websocket.send_json({"status": "processing", "message": "Classifying post..."})
        post_type = await classify_post_type(client_async, prompt)

        await websocket.send_json({"status": "processing", "message": "Generating keywords..."})
        keywords = await generate_keywords_post(client_async, prompt)

        await websocket.send_json({"status": "processing", "message": "Trending hashtags..."})
        hashtags = await fetch_trending_hashtags_post(client_async, keywords, [])

        await websocket.send_json({"status": "processing", "message": "Fetching SEO keywords..."})
        seo_keywords = await fetch_seo_keywords_post(client_async, keywords)

        await websocket.send_json({"status": "processing", "message": "Generating captions..."})

        platforms = ["instagram", "linkedin", "facebook", "twitter", "reddit"]

        captions = await generate_caption_post(
            query=prompt,
            seed_keywords=keywords,
            hashtags=hashtags,
            platforms=platforms
        )

        await websocket.send_json({
            "status": "completed",
            "post_type": post_type,
            "keywords": keywords,
            "seo_keywords": seo_keywords,
            "hashtags": hashtags,
            "captions": captions["captions"]
        })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.send_json({"status": "error", "message": "Failed to generate caption."})
    finally:
        await websocket.close()
