# stelle_backend/routes/engagement_routes.py

"""
Autonomous Engagement API Endpoints
────────────────────────────────────
Handles tone calibration, engagement settings, reply queue management,
statistics, manual engagement triggers, and comment management
(fetch, reply, post, delete — mirroring CommentsModal.tsx from analytics).
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from config import logger
from services.tone_calibration_service import (
    calibrate_tone_from_form,
    get_tone_status,
    reset_tone,
    get_engagement_settings,
    update_engagement_settings,
)
from services.engagement_service import (
    approve_reply,
    reject_reply,
    get_reply_queue,
    get_reply_stats,
)
from services.comment_poller import trigger_engagement_for_user
from services.social_engagement_api import (
    get_user_auth,
    fetch_comments,
    fetch_comments_with_replies_instagram,
    fetch_more_comments_instagram,
    post_comment_instagram,
    post_reply_instagram,
    delete_comment,
    post_reply,
)


router = APIRouter(tags=["Autonomous Engagement"])

SUPPORTED_COMMENT_CRUD_PLATFORMS = {"instagram", "threads"}


def _ensure_supported_comment_platform(platform: str) -> str:
    normalized = (platform or "").strip().lower()
    if normalized not in SUPPORTED_COMMENT_CRUD_PLATFORMS:
        raise HTTPException(
            status_code=501,
            detail=(
                f"{normalized or 'This'} platform is coming soon. "
                "Autonomous Engagement comment operations currently support only Instagram and Threads."
            ),
        )
    return normalized


def _ensure_threads_token(token: str):
    if not (token or "").strip().startswith("THAA"):
        raise HTTPException(
            status_code=401,
            detail="Invalid Threads token. Threads operations require a token starting with THAA.",
        )


# ───────────────────────────────────
#  TONE CALIBRATION
# ───────────────────────────────────

@router.post("/calibrate-tone")
async def calibrate_tone(payload: Dict[str, Any]):
    """
    Submit the tone-calibration form (Method A).
    Extracts a Tone DNA profile via LLM and stores it.

    Required: userId, formData (style, emoji_usage, reply_length, language, etc.)
    """
    try:
        user_id = payload.get("userId")
        form_data = payload.get("formData")

        if not user_id or not form_data:
            raise HTTPException(status_code=400, detail="userId and formData are required")

        result = await calibrate_tone_from_form(user_id, form_data)
        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tone calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Tone calibration failed")


@router.get("/tone-status/{user_id}")
async def tone_status(user_id: str):
    """
    Check whether a user has completed tone calibration.
    Returns calibration status and profile summary.
    """
    try:
        status = await get_tone_status(user_id)
        return {"success": True, "data": status}
    except Exception as e:
        logger.error(f"Failed to get tone status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get tone status")


@router.post("/reset-tone")
async def reset_user_tone(payload: Dict[str, Any]):
    """
    Archive the current tone profile and allow re-calibration.
    Required: userId
    """
    try:
        user_id = payload.get("userId")
        if not user_id:
            raise HTTPException(status_code=400, detail="userId is required")

        result = await reset_tone(user_id)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset tone: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset tone")


# ───────────────────────────────────
#  ENGAGEMENT SETTINGS
# ───────────────────────────────────

@router.get("/settings/{user_id}")
async def get_settings(user_id: str):
    """
    Retrieve the user's engagement settings (reply mode, daily limits, platforms, etc.)
    """
    try:
        settings = await get_engagement_settings(user_id)
        return {"success": True, "data": settings}
    except Exception as e:
        logger.error(f"Failed to get engagement settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get engagement settings")


@router.put("/settings")
async def update_settings(payload: Dict[str, Any]):
    """
    Update engagement settings for a user.
    Required: userId
    Optional fields: reply_mode, daily_reply_limit, max_replies_per_post,
                     reply_window_hours, active_platforms, enabled
    """
    try:
        user_id = payload.get("userId")
        if not user_id:
            raise HTTPException(status_code=400, detail="userId is required")

        # Extract only the settings fields (exclude userId)
        updates = {k: v for k, v in payload.items() if k != "userId"}

        if not updates:
            raise HTTPException(status_code=400, detail="No settings to update")

        result = await update_engagement_settings(user_id, updates)
        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update engagement settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update engagement settings")


# ───────────────────────────────────
#  REPLY QUEUE (Review Mode)
# ───────────────────────────────────

@router.get("/reply-queue/{user_id}")
async def reply_queue(user_id: str, status: str = "pending_review", limit: int = 50):
    """
    Get queued replies for review.
    Filters by status: pending_review | approved | rejected | draft
    """
    try:
        queue = await get_reply_queue(user_id, status=status, limit=limit)
        return {"success": True, "data": queue, "count": len(queue)}
    except Exception as e:
        logger.error(f"Failed to get reply queue: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get reply queue")


@router.post("/approve-reply")
async def approve_queued_reply(payload: Dict[str, Any]):
    """
    Approve a pending reply and post it to the platform.
    Required: logId, userId
    Optional: editedText (to modify the reply before posting)
    """
    try:
        log_id = payload.get("logId")
        user_id = payload.get("userId")
        edited_text = payload.get("editedText")

        if not log_id or not user_id:
            raise HTTPException(status_code=400, detail="logId and userId are required")

        result = await approve_reply(log_id, user_id, edited_text=edited_text)
        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve reply: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to approve reply")


@router.post("/reject-reply")
async def reject_queued_reply(payload: Dict[str, Any]):
    """
    Reject a pending reply (will not be posted).
    Required: logId, userId
    """
    try:
        log_id = payload.get("logId")
        user_id = payload.get("userId")

        if not log_id or not user_id:
            raise HTTPException(status_code=400, detail="logId and userId are required")

        result = await reject_reply(log_id, user_id)
        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject reply: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reject reply")


# ───────────────────────────────────
#  STATISTICS & DASHBOARD
# ───────────────────────────────────

@router.get("/stats/{user_id}")
async def engagement_stats(user_id: str):
    """
    Get engagement reply statistics for a user.
    Returns counts by status, daily totals, and platform breakdown.
    """
    try:
        stats = await get_reply_stats(user_id)
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error(f"Failed to get engagement stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get engagement stats")


# ───────────────────────────────────
#  MANUAL TRIGGER (Testing / On-Demand)
# ───────────────────────────────────

@router.post("/trigger")
async def trigger_engagement(payload: Dict[str, Any]):
    """
    Manually trigger an engagement cycle for a specific user.
    Useful for testing or on-demand processing outside the polling interval.
    Required: userId
    """
    try:
        user_id = payload.get("userId")
        if not user_id:
            raise HTTPException(status_code=400, detail="userId is required")

        result = await trigger_engagement_for_user(user_id)
        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger engagement: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to trigger engagement")


# ───────────────────────────────────
#  POSTED POSTS (for comment browsing)
# ───────────────────────────────────

@router.get("/posted-posts/{user_id}")
async def get_posted_posts(user_id: str, limit: int = 50):
    """
    Get the user's posts that have been published (status=posted)
    and have a valid platformPostId — these are the posts we can
    fetch comments for.

    Returns a lightweight list for the Comments panel.
    """
    from database import db
    from bson import ObjectId

    try:
        cursor = db["scheduledposts"].find(
            {
                "userId": user_id,
                "status": "posted",
                "platformPostId": {"$exists": True, "$ne": ""},
            },
            {
                "_id": 1,
                "caption": 1,
                "platform": 1,
                "platformPostId": 1,
                "mediaUrls": 1,
                "mediaType": 1,
                "scheduledAt": 1,
                "updatedAt": 1,
            },
        ).sort("updatedAt", -1).limit(limit)

        posts = []
        async for post in cursor:
            post["_id"] = str(post["_id"]) if isinstance(post.get("_id"), ObjectId) else post.get("_id")
            for dt_key in ("scheduledAt", "updatedAt"):
                val = post.get(dt_key)
                if val and hasattr(val, "isoformat"):
                    post[dt_key] = val.isoformat()
            posts.append(post)

        return {"success": True, "posts": posts, "count": len(posts)}

    except Exception as e:
        logger.error(f"Failed to get posted posts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get posted posts")


# ───────────────────────────────────────────────────────────
#  COMMENT MANAGEMENT (mirrors CommentsModal.tsx from analytics)
#  Fetch comments with replies, post comments, reply, delete.
#
#  The frontend (CommentsModal) gets the access token from
#  useSchedulerOauth() context and calls the Graph API directly.
#  These endpoints follow the same pattern: the frontend passes
#  the accessToken in the request body. If not provided, the
#  backend falls back to looking up the token from the DB.
# ───────────────────────────────────────────────────────────


async def _resolve_access_token(
    payload: Dict[str, Any],
    platform: str,
) -> str:
    """
    Get access token from the request payload, or fall back to DB lookup.

    The CommentsModal.tsx in analytics gets the token from useSchedulerOauth()
    and calls the Graph API directly. Our endpoints support both:
      1. Frontend passes accessToken directly (preferred — matches analytics flow)
      2. Falls back to oauthcredentials DB lookup using userId + platform
    """
    # 1. Token passed directly by frontend (like CommentsModal.tsx does)
    token = (payload.get("accessToken") or "").strip()
    if token:
        if platform == "threads":
            _ensure_threads_token(token)
        return token

    # 2. Fallback: look up from DB
    user_id = payload.get("userId", "")
    if not user_id:
        return ""

    # For Instagram/Facebook, prefer instafb (Business Graph token needed for comments)
    if platform in ("instagram", "facebook"):
        auth = await get_user_auth(user_id, "instafb")
        if auth:
            return auth.get("accessToken", "")

    auth = await get_user_auth(user_id, platform)
    if auth:
        token = auth.get("accessToken", "")
        if platform == "threads":
            _ensure_threads_token(token)
        return token

    return ""


@router.post("/comments/fetch")
async def fetch_comments_with_replies(payload: Dict[str, Any]):
    """
    Fetch comments with nested replies for a post.
    Works exactly like CommentsModal.tsx in publisher analytics:
      - Fetches top-level comments with like_count
      - Fetches replies for each comment
      - Supports pagination via paging.next

    Required: mediaId, platform
    Auth (one of): accessToken (preferred) OR userId (for DB lookup)
    Optional: limit (default 25)

    Response:
    {
        "success": true,
        "data": {
            "comments": [
                {
                    "comment_id": "...",
                    "author": "username",
                    "text": "...",
                    "timestamp": "...",
                    "like_count": 5,
                    "replies": [
                        {"id": "...", "username": "...", "text": "...", "timestamp": "...", "like_count": 0}
                    ]
                }
            ],
            "paging": {"next": "..." | null}
        }
    }
    """
    try:
        media_id = payload.get("mediaId")
        platform = _ensure_supported_comment_platform(payload.get("platform", "instagram"))
        limit = payload.get("limit", 25)

        if not media_id:
            raise HTTPException(status_code=400, detail="mediaId is required")

        access_token = await _resolve_access_token(payload, platform)
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail=(
                    f"No access token available for platform '{platform}'. "
                    "Pass accessToken in the request body, or ensure OAuth credentials are saved for this user."
                )
            )

        if platform == "instagram":
            result = await fetch_comments_with_replies_instagram(
                media_id=media_id,
                access_token=access_token,
                limit=limit,
            )
        elif platform == "threads":
            # Threads uses graph.threads.net and returns replies for the post.
            thread_comments = await fetch_comments(
                platform="threads",
                post_id=media_id,
                access_token=access_token,
                limit=limit,
            )
            result = {
                "comments": thread_comments,
                "paging": {"next": None},
            }

        return {"success": True, "data": result}

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to fetch comments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch comments")


@router.post("/comments/fetch-more")
async def fetch_more_comments(payload: Dict[str, Any]):
    """
    Load more comments using the pagination URL from a previous fetch.
    Required: nextUrl
    Auth (one of): accessToken (preferred) OR userId (for DB lookup)
    Optional: platform (default instagram)
    """
    try:
        next_url = payload.get("nextUrl")
        platform = _ensure_supported_comment_platform(payload.get("platform", "instagram"))

        if not next_url:
            raise HTTPException(status_code=400, detail="nextUrl is required")

        access_token = await _resolve_access_token(payload, platform)
        if not access_token:
            raise HTTPException(status_code=401, detail="No access token available.")

        result = await fetch_more_comments_instagram(
            next_url=next_url,
            access_token=access_token,
        )

        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load more comments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load more comments")


@router.post("/comments/post")
async def post_new_comment(payload: Dict[str, Any]):
    """
    Post a new top-level comment on a media/post.
    Required: mediaId, message, platform
    Auth (one of): accessToken (preferred) OR userId (for DB lookup)
    """
    try:
        media_id = payload.get("mediaId")
        message = payload.get("message", "").strip()
        platform = _ensure_supported_comment_platform(payload.get("platform", "instagram"))

        if not media_id or not message:
            raise HTTPException(status_code=400, detail="mediaId and message are required")

        access_token = await _resolve_access_token(payload, platform)
        if not access_token:
            raise HTTPException(status_code=401, detail="No access token available.")

        if platform == "instagram":
            result = await post_comment_instagram(
                media_id=media_id,
                message=message,
                access_token=access_token,
            )
        else:
            user_id = payload.get("userId", "")
            auth = await get_user_auth(user_id, "threads") if user_id else None
            user_id_threads = auth.get("accountId", "") if auth else ""

            success = await post_reply(
                platform="threads",
                comment_id=media_id,
                reply_text=message,
                access_token=access_token,
                user_id_threads=user_id_threads,
            )
            result = {"success": success}

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to post comment"))

        return {"success": True, "data": result}

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to post comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to post comment")


@router.post("/comments/reply")
async def reply_to_comment(payload: Dict[str, Any]):
    """
    Reply to a specific comment.
    Required: commentId, message, platform
    Auth (one of): accessToken (preferred) OR userId (for DB lookup)
    Optional: mediaId (needed for TikTok, LinkedIn)
    """
    try:
        comment_id = payload.get("commentId")
        message = payload.get("message", "").strip()
        platform = _ensure_supported_comment_platform(payload.get("platform", "instagram"))
        if not comment_id or not message:
            raise HTTPException(status_code=400, detail="commentId and message are required")

        access_token = await _resolve_access_token(payload, platform)
        if not access_token:
            raise HTTPException(status_code=401, detail="No access token available.")

        if platform == "instagram":
            success = await post_reply_instagram(
                comment_id=comment_id,
                reply_text=message,
                access_token=access_token,
            )
        else:
            user_id = payload.get("userId", "")
            auth = await get_user_auth(user_id, "threads") if user_id else None
            user_id_threads = auth.get("accountId", "") if auth else ""
            success = await post_reply(
                platform="threads",
                comment_id=comment_id,
                reply_text=message,
                access_token=access_token,
                user_id_threads=user_id_threads,
            )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to post reply. Check permissions and token.")

        return {"success": True, "message": "Reply posted successfully"}

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to reply to comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reply to comment")


@router.post("/comments/delete")
async def delete_comment_endpoint(payload: Dict[str, Any]):
    """
    Delete a comment.
    Required: commentId, platform
    Auth (one of): accessToken (preferred) OR userId (for DB lookup)
    Optional: videoId (needed for TikTok)
    """
    try:
        comment_id = payload.get("commentId")
        platform = _ensure_supported_comment_platform(payload.get("platform", "instagram"))
        if not comment_id:
            raise HTTPException(status_code=400, detail="commentId is required")

        access_token = await _resolve_access_token(payload, platform)
        if not access_token:
            raise HTTPException(status_code=401, detail="No access token available.")

        success = await delete_comment(
            platform=platform,
            comment_id=comment_id,
            access_token=access_token,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete comment. Check permissions.")

        return {"success": True, "message": "Comment deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete comment")
