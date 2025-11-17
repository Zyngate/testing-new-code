# stelle_backend/routes/integration_routes.py
import json
import asyncio
import time
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
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

# ====================================================================
# A. LINKEDIN SCRAPING ENDPOINTS
# ====================================================================

def run_linkedin_scrape_sync(query: str, max_results: int) -> Dict[str, Any]:
    logger.info(f"Starting simulated LinkedIn scrape for query: {query}")
    time.sleep(1)
    return {
        "status": "success",
        "query": query,
        "results_count": max_results,
        "data_summary": f"Retrieved summary for {query}'s profile.",
        "note": "Actual scraping logic (Selenium) runs synchronously here."
    }

@router.post("/scrape_linkedin", response_model=Dict[str, Any])
async def scrape_linkedin_endpoint(request: Dict[str, Any]):
    try:
        scrape_result = await asyncio.to_thread(
            run_linkedin_scrape_sync, request.get("query", ""), request.get("max_results", 5)
        )
        return scrape_result
    except Exception as e:
        logger.error(f"Error in /scrape_linkedin endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

# ====================================================================
# B. TEXT-ONLY CAPTION GENERATOR ENDPOINT
# ====================================================================

@router.websocket("/wss/generate-post")
async def websocket_generate_post_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "status": "connected",
        "message": "Connection established for caption generation."
    })

    try:
        # Receive prompt directly (no platforms required)
        prompt = await websocket.receive_text()
        logger.info(f"Received prompt for caption generation: '{prompt}'")

        client_async = await get_groq_client()

        # Classify post type
        await websocket.send_json({"status": "processing", "message": "Classifying post type..."})
        post_type = await classify_post_type(client_async, prompt)

        # Generate seed keywords
        await websocket.send_json({"status": "processing", "message": "Generating keywords..."})
        seed_keywords = await generate_keywords_post(client_async, prompt)

        # Fetch trending hashtags (optional: pass empty list if you don't want platform)
        await websocket.send_json({"status": "processing", "message": "Fetching trending hashtags..."})
        trending_hashtags = await fetch_trending_hashtags_post(client_async, seed_keywords, [])

        # Fetch SEO keywords
        await websocket.send_json({"status": "processing", "message": "Fetching SEO keywords..."})
        seo_keywords = await fetch_seo_keywords_post(client_async, seed_keywords)

        # Generate captions
        await websocket.send_json({"status": "processing", "message": "Generating captions..."})
        captions = await generate_caption_post(client_async, prompt, seed_keywords, trending_hashtags, [])

        # Return final output
        await websocket.send_json({
            "status": "completed",
            "message": "Caption generated successfully!",
            "trending_hashtags": trending_hashtags,
            "seo_keywords": seo_keywords,
            "captions": captions,
            "post_type": post_type,
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/generate-post")
    except Exception as e:
        logger.error(f"Caption generation failed with an exception: {e}", exc_info=True)
        await websocket.send_json({
            "status": "error",
            "message": "A critical error occurred while generating the caption."
        })
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed.")
