# stelle_backend/routes/post_routes.py

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Dict, Any, List

from database import db
from config import logger
from services.post_creation_service import create_post_from_uploaded_media
from services.bulk_post_creation_service import (
    process_bulk_media_urls,
    get_bulk_scheduling_preview,
)


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
    Intelligent Bulk Upload with Platform-Optimized Scheduling.
    
    **ORDER MATTERS**: The first URL in mediaUrls gets the earliest time slot,
    second URL gets the next slot, and so on. Reorder your URLs before calling
    to control which video posts first.
    
    Features:
    - Platform-specific posting frequency (e.g., Instagram 2-3/day, YouTube 1/day)
    - Automatic time distribution across days based on platform research
    - Cross-platform conflict avoidance
    - Content-type aware scheduling
    
    Request body:
    {
        "userId": str,
        "mediaUrls": [str],  // ORDER = posting order (first = earliest time)
        "platform": str | list[str],
        "scheduleMode": "AUTO" | "MANUAL",
        "scheduledAt": dict (for MANUAL mode),
        "growthMode": "conservative" | "optimal" | "aggressive",
        "contentType": str (optional, e.g., "VIDEO", "SHORT", "IMAGE")
    }
    
    Growth modes:
    - conservative: Minimum posting frequency (safe)
    - optimal: Recommended frequency for growth (default)
    - aggressive: Maximum frequency for rapid growth
    """

    user_id = payload.get("userId")
    media_urls = payload.get("mediaUrls", [])
    platform = payload.get("platform")

    schedule_mode = payload.get("scheduleMode", "AUTO")
    scheduled_at = payload.get("scheduledAt", {})
    growth_mode = payload.get("growthMode", "optimal")
    content_type = payload.get("contentType")  # Optional: VIDEO, IMAGE, SHORT, etc.

    if not user_id or not media_urls or not platform:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: userId, mediaUrls, platform"
        )

    # Normalize platform to list
    if isinstance(platform, str):
        platforms = [platform]
    else:
        platforms = platform

    # 🔒 MAX LIMIT
    MAX_BULK_UPLOAD = 30  # Increased limit for bulk posting
    if len(media_urls) > MAX_BULK_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BULK_UPLOAD} media items allowed per bulk upload"
        )

    # Validate growth mode
    valid_growth_modes = ["conservative", "optimal", "aggressive"]
    if growth_mode not in valid_growth_modes:
        growth_mode = "optimal"

    try:
        result = await process_bulk_media_urls(
            user_id=user_id,
            media_urls=media_urls,
            platforms=platforms,
            schedule_mode=schedule_mode,
            scheduled_at_manual=scheduled_at,
            growth_mode=growth_mode,
            content_type=content_type,
            preview_only=False,
        )

        return result

    except Exception as e:
        logger.error(f"❌ Bulk upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Bulk upload failed: {str(e)}"
        )


@router.post("/upload-bulk/schedule-preview")
async def bulk_schedule_preview(payload: dict):
    """
    Preview how bulk posts would be scheduled BEFORE uploading.
    
    This shows the time distribution strategy without processing media.
    Useful for users to see the schedule before committing.
    
    Request body:
    {
        "userId": str,
        "numPosts": int,
        "platforms": list[str],
        "contentType": str (optional),
        "growthMode": "conservative" | "optimal" | "aggressive"
    }
    """
    user_id = payload.get("userId")
    num_posts = payload.get("numPosts", 1)
    platforms = payload.get("platforms", [])
    content_type = payload.get("contentType", "VIDEO")
    growth_mode = payload.get("growthMode", "optimal")

    if not user_id or not platforms:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: userId, platforms"
        )

    try:
        preview = await get_bulk_scheduling_preview(
            user_id=user_id,
            num_posts=num_posts,
            platforms=platforms,
            content_type=content_type,
            growth_mode=growth_mode,
        )
        return preview

    except Exception as e:
        logger.error(f"❌ Schedule preview failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Preview generation failed: {str(e)}"
        )

@router.post("/upload-bulk-preview")
async def bulk_preview(payload: dict):
    """
    Bulk Preview - Generate AI captions + recommended times WITHOUT saving to DB.
    
    **ORDER MATTERS**: First URL in mediaUrls gets the earliest time slot.
    Reorder your URLs before calling to control posting sequence.
    
    Request:
    {
        "userId": "...",
        "mediaUrls": ["first_to_post.mp4", "second.mp4", "third.mp4"],
        "platforms": ["instagram", "tiktok"],
        "growthMode": "optimal",  // conservative | optimal | aggressive
        "startDate": "2026-02-15"  // optional
    }
    """

    user_id = payload.get("userId")
    media_urls = payload.get("mediaUrls", [])
    platforms = payload.get("platforms", [])
    
    # Legacy support for single platform (string or list)
    if not platforms and payload.get("platform"):
        platform_val = payload.get("platform")
        if isinstance(platform_val, list):
            platforms = platform_val  # Already a list, don't wrap again
        else:
            platforms = [platform_val]  # Single string, wrap in list

    growth_mode = payload.get("growthMode", "optimal")
    content_type = payload.get("contentType")
    start_date_str = payload.get("startDate")  # e.g. "2026-02-15" or "2026-02-15T10:00:00"

    if not user_id or not media_urls or not platforms:
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: userId, mediaUrls, and platforms"
        )

    # Parse startDate if provided
    start_date = None
    if start_date_str:
        try:
            start_date = datetime.fromisoformat(start_date_str)
            # If no timezone info, treat as UTC
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid startDate format: '{start_date_str}'. Use ISO format like '2026-02-15' or '2026-02-15T10:00:00'"
            )

    # Use bulk scheduling service with preview mode
    # This ensures coordinated time allocation with all our fixes:
    # - Platform-specific frequency limits (Instagram 2-3/day, YouTube 3-4/week, etc.)
    # - Unique time slots on each API call (randomization with seed + jitter)
    # - Chronological post indices
    # - Daily and weekly limit enforcement
    # - Optional startDate to begin scheduling from a future date
    result = await process_bulk_media_urls(
        user_id=user_id,
        media_urls=media_urls,
        platforms=platforms,
        schedule_mode="AUTO",
        growth_mode=growth_mode,
        content_type=content_type,
        preview_only=True,  # Don't save to database
        start_date=start_date,
    )

    return result

@router.post("/bulk-confirm")
async def bulk_confirm(payload: dict):
    """
    Bulk Confirm - Save user-approved posts to scheduler.
    
    After preview, user can edit captions/times, then call this to save.
    Order in approvedPosts = order saved.
    
    Request:
    {
        "userId": "...",
        "approvedPosts": [
            {
                "mediaUrl": "...",
                "caption": "edited caption",
                "platform": "instagram",
                "mediaType": "VIDEO",
                "scheduledAt": "2026-02-15T14:00:00Z"
            }
        ]
    }
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
            # scheduledAt may be local time (e.g., 2026-02-16T20:14:00-07:00)
            # or scheduledAtUTC if frontend sends it. Convert to UTC for DB storage.
            scheduled_at_raw = post.get("scheduledAtUTC") or post["scheduledAt"]
            scheduled_at = datetime.fromisoformat(scheduled_at_raw)
            if scheduled_at.tzinfo is None:
                scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)
            # Always store in UTC in the database
            scheduled_at_utc = scheduled_at.astimezone(timezone.utc)

            post_doc = {
                "userId": user_id,
                "mediaUrls": [post["mediaUrl"]],
                "caption": post.get("caption", ""),
                "platform": post["platform"],
                "mediaType": post["mediaType"],
                "scheduledAt": scheduled_at_utc,
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
