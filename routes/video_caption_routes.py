# routes/video_caption_routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import uuid
import os
from pathlib import Path
from services.video_caption_service import caption_from_video_file

router = APIRouter(tags=["Video Caption Generator"])

UPLOAD_DIR = Path("uploaded_videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/generate_video_caption")
async def generate_video_caption(
    video: UploadFile = File(...),
    platforms: list[str] = Form(...)   # ⭐ User can choose platforms properly
):
    """
    Upload a video, process it and return captions + hashtags.
    """

    # -----------------------------
    # SAVE UPLOADED VIDEO FILE
    # -----------------------------
    try:
        ext = video.filename.split(".")[-1] if video.filename else "mp4"
        tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"

        with open(tmp_path, "wb") as f:
            f.write(await video.read())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded video: {e}")

    # -----------------------------
    # PROCESS VIDEO FOR CAPTIONS
    # -----------------------------
    try:
        result = await caption_from_video_file(str(tmp_path), platforms)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video captioning failed: {e}")

    finally:
        # Cleanup temp file
        try:
            if tmp_path.exists():
                os.remove(tmp_path)
        except:
            pass

    # -----------------------------
    # RETURN EXACT STRUCTURE LIKE BEFORE
    # -----------------------------
    return {
        "message": "Video processed successfully",
        "detected_person": result.get("detected_person"),
        "transcript": result.get("transcript"),
        "visual_summary": result.get("visual_summary"),
        "text_summary": result.get("text_summary"),
        "marketing_prompt": result.get("marketing_prompt"),
        "keywords": result.get("keywords"),
        "captions": result.get("captions"),
        "platform_hashtags": result.get("platform_hashtags"),
    }
