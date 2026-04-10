# routes/recommendation_routes.py
import traceback
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from services.recommendation_service import (
    RecommendationService,
    PostData,
    InstagramPost,
    YouTubePost,
    LinkedInPost,
    ThreadsPost,
    TikTokPost,
    FacebookPost,
    RecommendationResponse
)
from services.time_slot_service import (
    save_user_post_analytics,
    get_optimal_times_for_platforms,
    get_single_optimal_time,
    TimeSlotService,
    PLATFORM_PEAK_HOURS,
)

router = APIRouter()
logger = logging.getLogger(__name__)
service = RecommendationService()  # Uses .env key automatically

class RecommendationRequest(BaseModel):
    posts: List[Dict[str, Any]]
    analytics: Optional[Dict[str, Dict[str, Any]]] = None
    timezone: Optional[str] = "UTC"

class SaveAnalyticsRequest(BaseModel):
    userId: str
    platform: str
    posts: List[Dict[str, Any]]  # Raw posts from Meta API

class GetOptimalTimesRequest(BaseModel):
    userId: str
    platforms: List[str]
    timezone: Optional[str] = None  # User's timezone (e.g., 'Asia/Kolkata'); auto-fetched from DB if not provided


def _detect_platform_from_link(link: str) -> str:
    url = (link or "").lower()
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    if "threads.net" in url or "threads.com" in url:
        return "threads"
    if "instagram.com" in url:
        return "instagram"
    if "linkedin.com" in url:
        return "linkedin"
    if "tiktok.com" in url:
        return "tiktok"
    if "facebook.com" in url or "fb.com" in url:
        return "facebook"
    return "instagram"


def _normalize_single_post(post: Dict[str, Any], default_tz: str = "UTC") -> PostData:
    platform = _detect_platform_from_link(str(post.get("link", "")))
    post_type = str(post.get("type", "VIDEO")).upper()
    posting_time = str(post.get("posting_time") or post.get("timestamp") or "")
    if not posting_time:
        from datetime import datetime
        posting_time = datetime.utcnow().isoformat() + "Z"

    common = {
        "link": str(post.get("link", "")),
        "type": post_type,
        "likes": int(post.get("likes", 0) or 0),
        "views": int(post.get("views", 0) or 0),
        "posting_time": posting_time,
        "caption": str(post.get("caption", "") or ""),
        "time_zone": str(post.get("time_zone", default_tz) or default_tz),
    }

    if platform == "youtube":
        return YouTubePost(
            **common,
            comments=int(post.get("comments", 0) or 0),
            shares=int(post.get("shares", 0) or 0),
            favourites=int(post.get("favourites", 0) or 0),
        )

    if platform == "threads":
        return ThreadsPost(
            **common,
            replies=int(post.get("replies", 0) or 0),
            reposts=int(post.get("reposts", 0) or 0),
        )

    if platform == "linkedin":
        return LinkedInPost(
            **common,
            comments=int(post.get("comments", 0) or 0),
            shares=int(post.get("shares", 0) or 0),
        )

    if platform == "tiktok":
        return TikTokPost(
            **common,
            comments=int(post.get("comments", 0) or 0),
            shares=int(post.get("shares", 0) or 0),
        )

    if platform == "facebook":
        return FacebookPost(
            **common,
            comments=int(post.get("comments", 0) or 0),
            shares=int(post.get("shares", 0) or 0),
            saved=int(post.get("saved", 0) or 0),
            interactions=int(post.get("interactions", 0) or 0),
            reach=int(post.get("reach", 0) or 0),
        )

    return InstagramPost(
        **common,
        comments=int(post.get("comments", 0) or 0),
        shares=int(post.get("shares", 0) or 0),
        saved=int(post.get("saved", 0) or 0),
        interactions=int(post.get("interactions", 0) or 0),
        reach=int(post.get("reach", 0) or 0),
    )

@router.post("/analyze", response_model=dict)
async def analyze_recommendation(request: RecommendationRequest):
    """Full recommendation analysis endpoint"""
    try:
        # Supports legacy normalized payloads and raw YouTube payloads with separate analytics.
        normalized_posts = []

        if request.analytics is not None:
            normalized_posts = service.normalize_youtube_payload(
                posts=request.posts,
                analytics=request.analytics,
                time_zone=request.timezone or "UTC"
            )
        else:
            for post in request.posts:
                normalized_posts.append(_normalize_single_post(post, request.timezone or "UTC"))

        result = service.generate_recommendations(normalized_posts)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[/analyze] Error: {e}\n{tb}")
        print(f"[/analyze] TRACEBACK:\n{tb}")  # Always visible in Render logs
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.post("/save-analytics")
async def save_analytics_endpoint(request: SaveAnalyticsRequest):
    """
    Save user's post analytics from Meta API for time slot optimization.
    Call this when fetching user's posts from Meta API.
    """
    try:
        result = await save_user_post_analytics(
            user_id=request.userId,
            platform=request.platform,
            posts_data=request.posts
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimal-times")
async def get_optimal_times_endpoint(request: GetOptimalTimesRequest):
    """
    Get optimal posting times for specified platforms.
    Uses user data if ≥15 posts available, otherwise uses AI research data.
    
    Returns timezone-aware times with minute-level precision for maximum engagement.
    """
    try:
        # Get user's timezone (from request or auto-fetched from DB)
        user_timezone = request.timezone
        
        result = await get_optimal_times_for_platforms(
            user_id=request.userId,
            platforms=request.platforms,
            user_timezone=user_timezone
        )
        
        # Format result for JSON response (already includes detailed metadata)
        formatted_result = {}
        for platform, info in result.items():
            formatted_result[platform] = {
                "scheduledAt": info.get("scheduledAtISO") or info["scheduledAt"].isoformat(),
                "localTime": info.get("localTime"),
                "timezone": info.get("timezone"),
                "reason": info["reason"],
                "dataSource": info["dataSource"],
                "dayOfWeek": info.get("dayOfWeek"),
                "hour": info.get("hour"),
                "minute": info.get("minute"),
                "engagementScore": info.get("engagementScore"),
                "isPeakHour": info.get("isPeakHour", False),
                "isBestDay": info.get("isBestDay", False)
            }
        
        return {
            "success": True,
            "userId": request.userId,
            "platforms": formatted_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{user_id}/{platform}")
async def get_user_analytics(user_id: str, platform: str, timezone: str = None):
    """
    Get cached analytics for a user on a specific platform.
    Includes staleness info so frontend knows when to refresh.
    """
    try:
        time_service = TimeSlotService(user_id, timezone)
        analytics, is_stale = await time_service._get_user_analytics(platform.lower())
        
        if not analytics:
            return {
                "success": True,
                "hasData": False,
                "needsRefresh": True,
                "message": f"No analytics data found for {platform}. Post more to get personalized recommendations!"
            }
        
        return {
            "success": True,
            "hasData": True,
            "isStale": is_stale,
            "needsRefresh": is_stale,
            "postCount": analytics.get("post_count", 0),
            "bestTimeSlots": analytics.get("best_time_slots", []),
            "dataSource": analytics.get("data_source", "unknown"),
            "updatedAt": analytics.get("updatedAt").isoformat() if analytics.get("updatedAt") else None,
            "message": "Data is stale (>24 hours). Please refresh by fetching latest posts." if is_stale else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-freshness/{user_id}")
async def check_analytics_freshness(user_id: str, timezone: str = None):
    """
    Check which platforms need analytics refresh.
    Frontend should call this to know when to fetch fresh data from Meta API.
    """
    try:
        time_service = TimeSlotService(user_id, timezone)
        platforms_status = {}
        
        for platform in ["instagram", "youtube", "tiktok", "facebook", "linkedin", "threads", "pinterest"]:
            analytics, is_stale = await time_service._get_user_analytics(platform)
            
            if not analytics:
                platforms_status[platform] = {
                    "hasData": False,
                    "needsRefresh": True,
                    "reason": "No data"
                }
            else:
                platforms_status[platform] = {
                    "hasData": True,
                    "needsRefresh": is_stale,
                    "postCount": analytics.get("post_count", 0),
                    "updatedAt": analytics.get("updatedAt").isoformat() if analytics.get("updatedAt") else None,
                    "reason": "Data older than 24 hours" if is_stale else "Data is fresh"
                }
        
        return {
            "success": True,
            "userId": user_id,
            "platforms": platforms_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform-peak-hours")
async def get_platform_peak_hours():
    """
    Get peak hours and engagement data for all supported platforms.
    
    This is useful for the frontend to display optimal posting times
    even before the user has analytics data.
    """
    try:
        result = {}
        for platform, config in PLATFORM_PEAK_HOURS.items():
            result[platform] = {
                "peakHours": config.get("peak_hours", []),
                "bestHours": config.get("best_hours", []),
                "weekendHours": config.get("weekend_hours", []),
                "weekdayHours": config.get("weekday_hours", []),
                "bestDays": config.get("best_days", []),
                "worstDays": config.get("worst_days", []),
                "engagementMultiplier": config.get("engagement_multiplier", {}),
                "description": config.get("description", "")
            }
        
        return {
            "success": True,
            "platforms": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/single-optimal-time")
async def get_single_optimal_time_endpoint(
    userId: str,
    platform: str,
    timezone: str = None,
    lastPostTime: str = None
):
    """
    Get the optimal posting time for a single platform.
    
    This is useful when scheduling one post at a time.
    
    Args:
        userId: The user's ID
        platform: Platform name (e.g., 'instagram')
        timezone: User's timezone (e.g., 'America/New_York')
        lastPostTime: ISO format string of the last post time to avoid
    """
    try:
        last_post = None
        if lastPostTime:
            from datetime import datetime
            last_post = datetime.fromisoformat(lastPostTime.replace("Z", "+00:00"))
        
        result = await get_single_optimal_time(
            user_id=userId,
            platform=platform,
            user_timezone=timezone,
            last_post_time=last_post
        )
        
        return {
            "success": True,
            "platform": platform,
            "scheduledAt": result["scheduledAtISO"],
            "localTime": result.get("localTime"),
            "timezone": result.get("timezone"),
            "reason": result["reason"],
            "dataSource": result["dataSource"],
            "dayOfWeek": result.get("dayOfWeek"),
            "hour": result.get("hour"),
            "minute": result.get("minute"),
            "engagementScore": result.get("engagementScore"),
            "isPeakHour": result.get("isPeakHour", False),
            "isBestDay": result.get("isBestDay", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
