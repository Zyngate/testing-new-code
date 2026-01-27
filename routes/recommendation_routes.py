# routes/recommendation_routes.py
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
    TimeSlotService,
)

router = APIRouter()
service = RecommendationService()  # Uses .env key automatically

class RecommendationRequest(BaseModel):
    posts: List[PostData]  # ✅ Keep Pydantic Union

class SaveAnalyticsRequest(BaseModel):
    userId: str
    platform: str
    posts: List[Dict[str, Any]]  # Raw posts from Meta API

class GetOptimalTimesRequest(BaseModel):
    userId: str
    platforms: List[str]

@router.post("/analyze", response_model=dict)
async def analyze_recommendation(request: RecommendationRequest):
    """Full recommendation analysis endpoint"""
    try:
        # ✅ Pass Pydantic models DIRECTLY - NO .dict() conversion!
        result = service.generate_recommendations(request.posts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    try:
        result = await get_optimal_times_for_platforms(
            user_id=request.userId,
            platforms=request.platforms
        )
        
        # Convert datetime to ISO format for JSON response
        formatted_result = {}
        for platform, info in result.items():
            formatted_result[platform] = {
                "scheduledAt": info["scheduledAt"].isoformat(),
                "reason": info["reason"],
                "dataSource": info["dataSource"]
            }
        
        return {
            "success": True,
            "userId": request.userId,
            "platforms": formatted_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{user_id}/{platform}")
async def get_user_analytics(user_id: str, platform: str):
    """
    Get cached analytics for a user on a specific platform.
    Includes staleness info so frontend knows when to refresh.
    """
    try:
        time_service = TimeSlotService(user_id)
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
async def check_analytics_freshness(user_id: str):
    """
    Check which platforms need analytics refresh.
    Frontend should call this to know when to fetch fresh data from Meta API.
    """
    try:
        time_service = TimeSlotService(user_id)
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