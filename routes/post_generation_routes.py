import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi import HTTPException
from pydantic import BaseModel

from services.ai_service import get_groq_client
from services.post_generator_service import (
    generate_keywords_post,
    generate_caption_post,
    fetch_platform_hashtags,
)
from config import logger

# This determines the tag shown in Swagger
router = APIRouter(tags=["Caption & Hashtag Generator"])


# ---------------------------------------------------------
# 1) 🔵 REST ENDPOINT (This will appear in Swagger UI)
# ---------------------------------------------------------

class GeneratePostRequest(BaseModel):
    query: str
    platforms: list[str] = ["instagram"]

@router.post("/generate_post")
async def generate_post(body: GeneratePostRequest):
    """
    REST API version of the Caption + Hashtag generator.
    This WILL appear in Swagger Docs.
    """
    try:
        client = await get_groq_client()
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise HTTPException(status_code=500, detail="Server misconfiguration: missing caption API key.")

    # Generate keywords
    try:
        keywords = await generate_keywords_post(client, body.query)
    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        raise HTTPException(status_code=500, detail="Keyword generation failed.")

    # Captions + Hashtags
    try:
        results = await generate_caption_post(body.query, keywords, body.platforms)
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        raise HTTPException(status_code=500, detail="Caption generation failed.")

    return {
        "keywords": keywords,
        "captions": results["captions"],
        "platform_hashtags": results["platform_hashtags"]
    }



# ---------------------------------------------------------
# 2) 🔵 WEBSOCKET ENDPOINT (Stays as you have it)
# ---------------------------------------------------------

@router.websocket("/wss/generate-post")
async def websocket_generate_post(websocket: WebSocket):
    """
    Robust handler that accepts:
      1) JSON: {"query": "...", "platforms": ["instagram","tiktok"]}
      OR
      2) Legacy: "instagram,tiktok" then second message is content string
    
    This WILL NOT appear in Swagger Docs (normal for WebSockets)
    """
    await websocket.accept()

    try:
        first = await websocket.receive_text()

        # Try JSON input
        try:
            payload = json.loads(first)
            if isinstance(payload, dict) and payload.get("query"):
                query = payload.get("query", "").strip()
                platforms = [p.strip().lower() for p in payload.get("platforms", ["instagram"])]
            else:
                raise ValueError("Invalid JSON")
        except (json.JSONDecodeError, ValueError):
            # Legacy CSV mode
            platforms = [p.strip().lower() for p in first.split(",") if p.strip()]
            query = await websocket.receive_text()

        if not query:
            await websocket.send_json({"status": "error", "message": "Missing query/content."})
            await websocket.close()
            return

        if not platforms:
            platforms = ["instagram"]

        await websocket.send_json({"status": "processing", "message": "Initializing AI client..."})

        try:
            client_async = await get_groq_client()
        except Exception as e:
            logger.error(f"Client init failed: {e}")
            await websocket.send_json({"status": "error", "message": "Missing caption API key."})
            await websocket.close()
            return

        # Keywords
        await websocket.send_json({"status": "processing", "message": "Generating keywords..."})
        try:
            keywords = await generate_keywords_post(client_async, query)
        except Exception as e:
            logger.error(f"Keyword generation failed: {e}")
            await websocket.send_json({"status": "error", "message": "Keyword generation failed."})
            await websocket.close()
            return

        # Hashtags
        await websocket.send_json({"status": "processing", "message": "Generating platform hashtags..."})
        platform_hashtags = {}
        for p in platforms:
            try:
                platform_hashtags[p] = await fetch_platform_hashtags(None, keywords, p, query)
            except Exception as inner_e:
                logger.error(f"Hashtag error for {p}: {inner_e}")
                platform_hashtags[p] = []

        # Captions
        await websocket.send_json({"status": "processing", "message": "Generating captions..."})
        try:
            results = await generate_caption_post(query, keywords, platforms)
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            await websocket.send_json({"status": "error", "message": "Caption generation failed."})
            await websocket.close()
            return

        # Final Response
        await websocket.send_json({
            "status": "completed",
            "message": "Post generated successfully",
            "keywords": keywords,
            "captions": results.get("captions", {}),
            "platform_hashtags": results.get("platform_hashtags", platform_hashtags)
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
        try:
            await websocket.send_json({"status": "error", "message": "Server error occurred."})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
