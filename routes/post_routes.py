# stelle_backend/routes/post_routes.py

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Dict, Any, List

from database import db
from config import logger
from services.post_creation_service import create_post_from_uploaded_video


router = APIRouter(tags=["Posts"])  # Removed prefix - main.py already adds /posts


# -------------------------------------------------------
# Create / Schedule a Post
# -------------------------------------------------------
@router.post("/schedule", response_model=Dict[str, Any])
async def schedule_post(payload: Dict[str, Any]):
    """
    Schedules a social media post.
    This endpoint ONLY saves the post intent.
    Execution is handled by the background scheduler.
    """

    try:
        # --- Required fields ---
        user_id: str = payload.get("userId")
        media_urls: List[str] = payload.get("mediaUrls", [])
        scheduled_at: str = payload.get("scheduledAt")
        platform: str = payload.get("platform")
        media_type: str = payload.get("mediaType")

        # --- Optional fields ---
        caption: str = payload.get("caption", "")
        title: str = payload.get("title", "")
        board_id: str | None = payload.get("boardId")
        cover_image_url: str = payload.get("coverImageUrl", "")
        thumb_offset: str = payload.get("thumbOffset", "")
        youtube_privacy: str = payload.get("youtubePrivacy", "public")

        # --- Validation ---
        if not all([user_id, media_urls, scheduled_at, platform, media_type]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields"
            )

        # --- Normalize values ---
        platform = platform.lower().strip()
        media_type = media_type.upper().strip()

        # --- Parse datetime ---
        try:
            scheduled_dt = datetime.fromisoformat(scheduled_at)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="scheduledAt must be ISO format"
            )

        if scheduled_dt.tzinfo is None:
            scheduled_dt = scheduled_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        # --- Build DB document ---
        post_doc = {
            "userId": user_id,
            "mediaUrls": media_urls,
            "caption": caption,
            "scheduledAt": scheduled_dt,
            "mediaType": media_type,
            "status": "scheduled",
            "platform": platform,
            "boardId": board_id,
            "title": title,
            "coverImageUrl": cover_image_url,
            "thumbOffset": thumb_offset,
            "youtubePrivacy": youtube_privacy,
            "createdAt": now,
            "updatedAt": now
        }

        result = db["scheduledposts"].insert_one(post_doc)

        logger.info(
            f"📌 Post scheduled | user={user_id} | platform={platform} | time={scheduled_dt}"
        )

        return {
            "success": True,
            "postId": str(result.inserted_id),
            "status": "scheduled",
            "scheduledAt": scheduled_dt
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error("❌ Failed to schedule post", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to schedule post"
        )


# -------------------------------------------------------
# Get Scheduled Posts for a User
# -------------------------------------------------------
@router.get("/", response_model=Dict[str, Any])
async def get_user_posts(userId: str):
    try:
        posts = list(
            db["scheduledposts"].find({"userId": userId}).sort("scheduledAt", -1)
        )

        from bson import ObjectId

        def serialize_post(post: dict) -> dict:
            post["_id"] = str(post["_id"]) if isinstance(post.get("_id"), ObjectId) else post.get("_id")
            for key in ["scheduledAt", "createdAt", "updatedAt"]:
                if key in post and post[key]:
                    post[key] = post[key].isoformat()
            return post

        posts = [serialize_post(post) for post in posts]

        return {
            "success": True,
            "count": len(posts),
            "posts": posts
        }

    except Exception as e:
        logger.error("❌ Failed to fetch posts", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch posts"
        )


@router.post("/upload-video")
async def upload_video_and_schedule(payload: dict):
    """
    Orchestrates:
    Cloudinary URL → AI → Best time → Scheduler
    """

    user_id = payload.get("userId")
    cloudinary_url = payload.get("cloudinaryUrl")
    platform = payload.get("platform")

    if not all([user_id, cloudinary_url, platform]):
        raise HTTPException(status_code=400, detail="Missing fields")

    result = await create_post_from_uploaded_video(
        user_id=user_id,
        cloudinary_url=cloudinary_url,
        platform=platform
    )

    return result



