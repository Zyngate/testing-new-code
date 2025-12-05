# stelle_backend/routes/video_routes.py
import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from services.common_utils import logger
from services.video_caption_service import caption_from_video_file, TEMP_DIR
from groq import AsyncGroq
import random
from config import GENERATE_API_KEYS

router = APIRouter(tags=["VideoCaption"])

# POST /video_caption - multipart/form-data
@router.post("/video_caption")
async def video_caption_endpoint(
    file: UploadFile = File(...),
    platforms: str = Form(None),  # comma-separated list of platforms e.g. "instagram,tiktok"
):
    """
    Upload a video file (multipart/form). Optional form field `platforms` (CSV).
    Returns captions + platform hashtags + keywords.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # parse platforms
    if platforms:
        requested_platforms = [p.strip().lower() for p in platforms.split(",") if p.strip()]
    else:
        requested_platforms = ["instagram", "facebook", "linkedin", "tiktok", "youtube"]

    # save uploaded file temporarily
    uid = file.filename or f"video_{os.urandom(4).hex()}"
    out_path = TEMP_DIR / f"{uid}_{os.urandom(4).hex()}"
    try:
        with open(out_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Failed saving uploaded video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # create AsyncGroq client (if needed by services)
    api_key = random.choice(GENERATE_API_KEYS) if GENERATE_API_KEYS else None
    client = AsyncGroq(api_key=api_key) if api_key else None

    try:
        result = await caption_from_video_file(str(out_path), requested_platforms, client=client)
    except Exception as e:
        logger.error(f"Video caption pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass

    return JSONResponse(result)
