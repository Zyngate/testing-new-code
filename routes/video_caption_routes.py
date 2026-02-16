# routes/video_caption_routes.py
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uuid
import os
from pathlib import Path
from config import logger

router = APIRouter(tags=["Video Caption Generator"])

UPLOAD_DIR = Path("uploaded_videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _get_fallback_captions(platforms: list[str], error_msg: str = "") -> dict:
    """Generate fallback response when video processing fails."""
    return {
        "message": "Video processing encountered an issue. Using fallback captions.",
        "error_info": error_msg,
        "detected_person": None,
        "transcript": "",
        "visual_summary": "Could not analyze video",
        "text_summary": "",
        "marketing_prompt": "",
        "keywords": ["video", "content", "social"],
        "captions": {p: f"Check out this amazing content! 🔥" for p in platforms},
        "platform_hashtags": {p: ["#content", "#viral", "#trending"] for p in platforms},
    }

@router.post("/generate_video_caption")
async def generate_video_caption(
    video: UploadFile = File(...),
    platforms: list[str] = Form(...)
):
    """
    Upload a video, process it and return captions + hashtags.
    NEVER crashes - always returns a valid response.
    """
    tmp_path = None
    
    # Normalize platforms
    if not platforms:
        platforms = ["instagram"]
    platforms = [p.lower().strip() for p in platforms if p]

    # -----------------------------
    # SAVE UPLOADED VIDEO FILE
    # -----------------------------
    try:
        ext = video.filename.split(".")[-1] if video.filename else "mp4"
        tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"

        with open(tmp_path, "wb") as f:
            f.write(await video.read())

    except Exception as e:
        logger.error(f"Failed to save video: {e}")
        return JSONResponse(
            status_code=200,
            content=_get_fallback_captions(platforms, f"Upload failed: {str(e)[:100]}")
        )

    # -----------------------------
    # PROCESS VIDEO FOR CAPTIONS
    # -----------------------------
    try:
        # Import here to avoid circular imports and catch import errors
        from services.video_caption_service import caption_from_video_file
        result = await caption_from_video_file(str(tmp_path), platforms)

    except Exception as e:
        logger.error(f"Video captioning failed: {e}", exc_info=True)
        # Return fallback instead of crashing
        return JSONResponse(
            status_code=200,
            content=_get_fallback_captions(platforms, f"Processing failed: {str(e)[:100]}")
        )

    finally:
        # Cleanup temp file
        try:
            if tmp_path and tmp_path.exists():
                os.remove(tmp_path)
        except:
            pass

    # -----------------------------
    # RETURN RESPONSE (with safety checks)
    # -----------------------------
    try:
        return {
            "message": "Video processed successfully",
            "detected_person": result.get("detected_person"),
            "transcript": result.get("transcript", ""),
            "visual_summary": result.get("visual_summary", ""),
            "text_summary": result.get("text_summary", ""),
            "marketing_prompt": result.get("marketing_prompt", ""),
            "keywords": result.get("keywords", []),
            "captions": result.get("captions", {}),
            "platform_hashtags": result.get("platform_hashtags", {}),
            "titles": result.get("titles", {}),
        }
    except Exception as e:
        logger.error(f"Failed to format response: {e}")
        return JSONResponse(
            status_code=200,
            content=_get_fallback_captions(platforms, "Response formatting failed")
        )
