# stelle_backend/routes/integration_routes.py
import json
import asyncio
import time  # Needed for time.sleep in run_linkedin_scrape_sync
from typing import List, Dict, Any, Union, Tuple

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Services and Models imported from internal modular structure
from models.common_models import UserInput, PostGenOptions, ScrapeRequest 
from services.common_utils import logger 
from services.ai_service import get_groq_client
from services.post_generator_service import (
    classify_post_type, 
    generate_keywords_post, fetch_trending_hashtags_post, fetch_seo_keywords_post, 
    generate_caption_post, Platforms, generate_html_code_post
)

router = APIRouter(tags=["Integrations"])  # Sets the tag in Swagger UI

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
async def scrape_linkedin_endpoint(request: ScrapeRequest):
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
    await websocket.accept()
    
    await websocket.send_json({
        "status": "connected",
        "message": "Connection established for post generation."
    })

    try:
        post_option_type = websocket.query_params.get("post_option")
        if not post_option_type:
            await websocket.send_json({
                "status": "error",
                "message": "Error: Missing post_option parameter in URL."
            })
            return

        post_option_type = post_option_type.lower()
        client_async = await get_groq_client()

        # -------------------
        # 1. Platform Options
        # -------------------
        platform_options_str = await websocket.receive_text()
        platform_options_indices = platform_options_str.split(',')

        platform_options = []
        for x in platform_options_indices:
            if x.isdigit() and int(x) < len(Platforms.platform_list):
                item = Platforms.platform_list[int(x)]
                if isinstance(item, str):
                    try:
                        item = Platforms(item)  # convert string to Enum
                    except Exception as e:
                        logger.error(f"Invalid platform: {item}, skipping. Error: {e}")
                        continue
                platform_options.append(item)

        if not platform_options:
            await websocket.send_json({
                "status": "error",
                "message": "No valid platform options selected"
            })
            return

        logger.info(f"Selected platform options (Enum): {platform_options}")

        # -------------------
        # 2. Prompt
        # -------------------
        prompt = await websocket.receive_text()
        logger.info(f"Received prompt for post generation: '{prompt}'")

        # -------------------
        # 3. Classification
        # -------------------
        await websocket.send_json({"status": "processing", "message": "Classifying post type..."})
        post_type = await classify_post_type(client_async, prompt)

        # -------------------
        # 4. Keywords
        # -------------------
        await websocket.send_json({
            "status": "processing",
            "message": f"Post classified as {post_type}. Generating keywords..."
        })
        seed_keywords = await generate_keywords_post(client_async, prompt)

        # -------------------
        # 5. Hashtags
        # -------------------
        await websocket.send_json({"status": "processing", "message": "Fetching trending hashtags..."})
        trending_hashtags = await fetch_trending_hashtags_post(client_async, seed_keywords, platform_options)

        # -------------------
        # 6. SEO Keywords
        # -------------------
        await websocket.send_json({"status": "processing", "message": "Fetching SEO keywords..."})
        seo_keywords = await fetch_seo_keywords_post(client_async, seed_keywords)

        html_code, captions, parsed_media = None, None, None

        if post_option_type == PostGenOptions.Text:
            await websocket.send_json({"status": "processing", "message": "Generating text-based post..."})
            html_code = await generate_html_code_post(client_async, prompt, post_type)
            captions = await generate_caption_post(client_async, prompt, seed_keywords, trending_hashtags, platform_options)
        else:
            await websocket.send_json({"status": "processing", "message": "Skipping media fetch (Pexels removed)..."})
            parsed_media = []
            await websocket.send_json({"status": "processing", "message": "Crafting the perfect caption..."})
            captions = await generate_caption_post(client_async, prompt, seed_keywords, trending_hashtags, platform_options)

        # -------------------
        # 7. Final Output
        # -------------------
        await websocket.send_json({
            "status": "completed",
            "message": "Post Generated Successfully!",
            "trending_hashtags": trending_hashtags,
            "seo_keywords": seo_keywords,
            "captions": captions,
            "html_code": html_code,
            "media": parsed_media,
            "post_type": post_option_type,
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/generate-post")
    except Exception as e:
        logger.error(f"Post generation failed with an exception: {e}", exc_info=True)
        await websocket.send_json({
            "status": "error",
            "message": "A critical error occurred while generating the post."
        })
    finally:
        await websocket.close()
        logger.info("WebSocket connection for /ws/generate-post has been closed.")
