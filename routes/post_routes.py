# stelle_backend/routes/post_routes.py

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Dict, Any, List

from database import db
from config import logger
from services.post_creation_service import create_post_from_uploaded_media


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
    schedule_mode = payload.get("scheduleMode", "AUTO")
    scheduled_at = payload.get("scheduledAt", {})

    if not all([user_id, cloudinary_url, platform]):
        raise HTTPException(status_code=400, detail="Missing fields")

    result = await create_post_from_uploaded_media(
    user_id=user_id,
    cloudinary_url=cloudinary_url,
    platform=platform,
    schedule_mode=schedule_mode,
    scheduled_at=scheduled_at
)


    return result

@router.post("/upload-bulk")
async def upload_bulk_posts(payload: dict):
    """
    Bulk upload:
    Multiple media URLs → AI → Scheduling (manual/auto)
    """

    user_id = payload.get("userId")
    media_urls = payload.get("mediaUrls", [])
    platform = payload.get("platform")

    schedule_mode = payload.get("scheduleMode", "AUTO")
    scheduled_at = payload.get("scheduledAt", {})

    if not user_id or not media_urls or not platform:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields"
        )

    # 🔒 MAX LIMIT
    MAX_BULK_UPLOAD = 10
    if len(media_urls) > MAX_BULK_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BULK_UPLOAD} media items allowed per bulk upload"
        )

    results = []

    for index, media_url in enumerate(media_urls):
        try:
            result = await create_post_from_uploaded_media(
                user_id=user_id,
                cloudinary_url=media_url,
                platform=platform,
                schedule_mode=schedule_mode,
                scheduled_at=scheduled_at
            )

            results.append({
                "mediaUrl": media_url,
                "status": "success",
                "posts": result.get("results", [])
            })

        except Exception as e:
            logger.error(
                f"❌ Bulk post failed for {media_url}",
                exc_info=True
            )
            results.append({
                "mediaUrl": media_url,
                "status": "failed",
                "error": str(e)
            })

    return {
        "success": True,
        "totalMedia": len(media_urls),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }

@router.post("/upload-bulk-preview")
async def bulk_preview(payload: dict):
    """
    Generate AI captions + recommended times
    WITHOUT saving to DB
    """

    user_id = payload.get("userId")
    media_urls = payload.get("mediaUrls", [])
    platform = payload.get("platform")

    schedule_mode = payload.get("scheduleMode", "AUTO")
    scheduled_at = payload.get("scheduledAt", {})

    if not user_id or not media_urls or not platform:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields"
        )

    previews = []

    for media_url in media_urls:
        result = await create_post_from_uploaded_media(
            user_id=user_id,
            cloudinary_url=media_url,
            platform=platform,
            schedule_mode=schedule_mode,
            scheduled_at=scheduled_at,
            preview_only=True,   # 👈 KEY
        )

        previews.append({
            "mediaUrl": media_url,
            "results": result.get("results", [])
        })

    return {
        "success": True,
        "preview": previews
    }

@router.post("/bulk-confirm")
async def bulk_confirm(payload: dict):
    """
    Final approval step.
    Saves user-edited posts directly to scheduler.
    """

    user_id = payload.get("userId")
    approved_posts = payload.get("approvedPosts", [])

    if not user_id or not approved_posts:
        raise HTTPException(
            status_code=400,
            detail="Missing userId or approvedPosts"
        )

    now = datetime.now(timezone.utc)
    saved_posts = []

    for post in approved_posts:
        try:
            scheduled_at = datetime.fromisoformat(post["scheduledAt"])
            if scheduled_at.tzinfo is None:
                scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)

            post_doc = {
                "userId": user_id,
                "mediaUrls": [post["mediaUrl"]],
                "caption": post.get("caption", ""),
                "platform": post["platform"],
                "mediaType": post["mediaType"],
                "scheduledAt": scheduled_at,
                "status": "pending",
                "createdAt": now,
                "updatedAt": now,
                "timeDataSource": "manual",
                "recommendationReason": "User approved"
            }

            await db["scheduledposts"].insert_one(post_doc)
            saved_posts.append(post_doc)

        except Exception as e:
            logger.error("❌ Failed to save approved post", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to save approved posts"
            )

    return {
        "success": True,
        "scheduledCount": len(saved_posts)
    }

