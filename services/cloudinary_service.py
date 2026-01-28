# services/cloudinary_service.py

import cloudinary.uploader
from fastapi import UploadFile
from config import logger

def upload_video_to_cloudinary(file: UploadFile) -> dict:
    try:
        result = cloudinary.uploader.upload(
            file.file,
            resource_type="auto",
            folder="scheduler_media"
        )
        return {
            "public_id": result["public_id"],
            "secure_url": result["secure_url"]
        }
    except Exception:
        logger.error("Cloudinary upload failed", exc_info=True)
        raise
