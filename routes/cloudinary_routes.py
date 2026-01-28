# stelle_backend/routes/cloudinary_routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException
import cloudinary.uploader
from config import logger

router = APIRouter(prefix="/media", tags=["Media Upload"])


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video to Cloudinary and return the URL
    """

    try:
        # Basic validation
        if not file.content_type.startswith(("video/", "image/")):
            raise HTTPException(
                status_code=400,
                detail="Only image or video files are allowed"
          )

        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            file.file,
            resource_type="auto",
            folder="scheduler_media"
        )

        return {
            "success": True,
            "public_id": result["public_id"],
            "secure_url": result["secure_url"]
        }

    except Exception as e:
        logger.error("❌ Cloudinary upload failed", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Video upload failed"
        )
