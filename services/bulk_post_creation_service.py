import tempfile
import os
import asyncio
import cloudinary.uploader
from fastapi import UploadFile 
from services.post_creation_service import create_post_from_uploaded_video


async def upload_to_cloudinary(file: UploadFile) -> str:
    result = cloudinary.uploader.upload(
        file.file,
        resource_type="video",
        folder="scheduler_videos"
    )
    return result["secure_url"]


async def process_bulk_videos(
    user_id: str,
    videos: list,
    platforms: list[str]
):
    scheduled_posts = []

    for video in videos:
        # 1️⃣ Upload video once
        cloudinary_url = await upload_to_cloudinary(video)

        # 2️⃣ Schedule same video across multiple platforms
        for platform in platforms:
            post = await create_post_from_uploaded_video(
                user_id=user_id,
                cloudinary_url=cloudinary_url,
                platform=platform
            )

            scheduled_posts.append({
                "video": video.filename,
                "platform": platform,
                "scheduledAt": post["scheduledAt"],
                "reason": post["recommendationReason"]
            })

    return scheduled_posts
