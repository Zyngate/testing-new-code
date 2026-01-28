from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import cloudinary.uploader
from config import logger

router = APIRouter(prefix="/media", tags=["Media Upload"])

MAX_MEDIA_UPLOAD = 10


@router.post("/upload-media")
async def upload_media(files: List[UploadFile] = File(...)):
    """
    Upload up to 10 images/videos to Cloudinary
    Returns Cloudinary URLs
    """

    if len(files) > MAX_MEDIA_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_MEDIA_UPLOAD} media files allowed"
        )

    uploaded_files = []

    for file in files:
        if not file.content_type.startswith(("image/", "video/")):
            raise HTTPException(
                status_code=400,
                detail="Only image or video files are allowed"
            )

        try:
            result = cloudinary.uploader.upload(
                file.file,
                resource_type="auto",
                folder="scheduler_media"
            )

            uploaded_files.append({
                "originalName": file.filename,
                "public_id": result["public_id"],
                "secure_url": result["secure_url"],
                "resource_type": result["resource_type"]
            })

        except Exception:
            logger.error(
                f"❌ Cloudinary upload failed for {file.filename}",
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Media upload failed"
            )

    return {
        "success": True,
        "count": len(uploaded_files),
        "media": uploaded_files
    }
