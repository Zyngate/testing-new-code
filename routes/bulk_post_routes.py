from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import tempfile
import os

from services.bulk_post_creation_service import process_bulk_videos

router = APIRouter(prefix="/posts", tags=["Bulk Video Scheduling"])


@router.post("/bulk-upload-videos")
async def bulk_upload_videos(
    videos: List[UploadFile] = File(...),
    platforms: List[str] = Form(...),
    userId: str = Form(...)
):
    """
    Upload 10–20 videos and schedule them across multiple platforms
    """

    if len(videos) > 20:
        raise HTTPException(status_code=400, detail="Max 20 videos allowed")

    results = await process_bulk_videos(
        user_id=userId,
        videos=videos,
        platforms=platforms
    )

    return {
        "success": True,
        "totalVideos": len(videos),
        "platforms": platforms,
        "scheduledPosts": results
    }
