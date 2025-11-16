# stelle_backend/routes/integration_routes.py
import json
import asyncio
import time  # Needed for time.sleep in run_linkedin_scrape_sync
from typing import List, Dict, Any, Union, Tuple  # Needed for proper typing hints

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Services and Models imported from internal modular structure
from models.common_models import UserInput, PostGenOptions, ScrapeRequest 
from services.common_utils import logger 
from services.ai_service import get_groq_client
from services.post_generator_service import (
    classify_post_type, 
    generate_keywords_post, fetch_trending_hashtags_post, fetch_seo_keywords_post, 
    generate_caption_post, Platforms, generate_html_code_post, parse_media
)

router = APIRouter(tags=["Integrations"])  # Sets the tag in Swagger UI

# ====================================================================
# A. LINKEDIN SCRAPING ENDPOINTS
# ====================================================================

def run_linkedin_scrape_sync(query: str, max_results: int) -> Dict[str, Any]:
    """
    Placeholder for synchronous Selenium scraping logic. 
    This function simulates scraping success and must run in a separate thread.
    """
    logger.info(f"Starting simulated LinkedIn scrape for query: {query}")
    time.sleep(1)  # Simulates the blocking network/browser operation
    return {
        "status": "success",
        "query": query,
        "results_count": max_results,
        "data_summary": f"Retrieved summary for {query}'s profile.",
        "note": "Actual scraping logic (Selenium) runs synchronously here."
    }

@router.post("/scrape_linkedin", response_model=Dict[str, Any])
async def scrape_linkedin_endpoint(request: ScrapeRequest):
    """
    Endpoint to trigger synchronous (blocking) LinkedIn scraping operation,
    running in a background thread to prevent Uvicorn from locking up.
    """
    try:
        scrape_result = await asyncio.to_thread(
            run_linkedin_scrape_sync, request.query, request.max_results
        )
        return scrape_result

    except Exception as e:
        logger.error(f"Error in /scrape_linkedin endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

# ====================================================================
# B. POST GENERATOR ENDPOINTS
# ====================================================================

@router.websocket("/wss/generate-post")
async def websocket_generate_post_endpoint(websocket: WebSocket):
    """Streams the full social media content generation pipeline."""
    await websocket.accept()
    
    await websocket.send_json(
        {"status": "connected", "message": "Connection established for post generation."}
    )

    try:
        post_option_type = websocket.query_params.get("post_option")
        if not post_option_type:
            await websocket.send_json({"status": "error", "message": "Error: Missing post_option parameter in URL."})
            return

        post_option_type = post_option_type.lower()
        client_async = await get_groq_client()

        # 1. Platform Options
        platform_options_str = await websocket.receive_text()
        await websocket.send_json({"status": "processing", "message": "Received platform options..."})
        
        platform_options_indices = platform_options_str.split(',')
        platform_options = [Platforms.platform_list[int(x)] for x in platform_options_indices if x.isdigit() and int(x) < len(Platforms.platform_list)]
        logger.info(f"Selected platform options: {platform_options}")

        # 2. Prompt
        prompt = await websocket.receive_text()
        logger.info(f"Received prompt for post generation: '{prompt}'")

        # 3. Classification
        await websocket.send_json({"status": "processing", "message": "Classifying post type..."})
        post_type = await classify_post_type(client_async, prompt)

        # 4. Keywords
        await websocket.send_json({"status": "processing", "message": f"Post classified as {post_type}. Generating keywords..."})
        seed_keywords = await generate_keywords_post(client_async, prompt)

        # 5. Hashtags
        await websocket.send_json({"status": "processing", "message": "Fetching trending hashtags..."})
        trending_hashtags = await fetch_trending_hashtags_post(client_async, seed_keywords, platform_options)

        # 6. SEO Keywords
        await websocket.send_json({"status": "processing", "message": "Fetching SEO keywords..."})
        seo_keywords = await fetch_seo_keywords_post(client_async, seed_keywords)

        html_code, captions, parsed_media = None, None, None

        if post_option_type == PostGenOptions.Text:
            # Text Post: Generate HTML
            await websocket.send_json({"status": "processing", "message": "Generating text-based post..."})
            html_code = await generate_html_code_post(client_async, prompt, post_type)
            captions = await generate_caption_post(client_async, prompt, seed_keywords, trending_hashtags, platform_options)

        else:
            # Photo/Video Post: Skip Pexels, use empty media
            await websocket.send_json({"status": "processing", "message": "Skipping media fetch (Pexels removed)..."} )
            parsed_media = []

            await websocket.send_json({"status": "processing", "message": "Crafting the perfect caption..."})
            captions = await generate_caption_post(client_async, prompt, seed_keywords, trending_hashtags, platform_options)

        # 7. Final Output
        await websocket.send_json(
            {
                "status": "completed",
                "message": "Post Generated Successfully!",
                "trending_hashtags": trending_hashtags,
                "seo_keywords": seo_keywords,
                "captions": captions,
                "html_code": html_code,
                "media": parsed_media,
                "post_type": post_option_type,
            }
        )

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/generate-post")
    except Exception as e:
        logger.error(f"Post generation failed with an exception: {e}", exc_info=True)
        await websocket.send_json(
            {"status": "error", "message": "A critical error occurred while generating the post."}
        )
    finally:
        await websocket.close()
        logger.info("WebSocket connection for /ws/generate-post has been closed.")
