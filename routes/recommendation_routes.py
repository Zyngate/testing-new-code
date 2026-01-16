# routes/recommendation_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union
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

router = APIRouter()
service = RecommendationService()  # Uses .env key automatically

class RecommendationRequest(BaseModel):
    posts: List[PostData]  # ✅ Keep Pydantic Union

@router.post("/analyze", response_model=dict)
async def analyze_recommendation(request: RecommendationRequest):
    """Full recommendation analysis endpoint"""
    try:
        # ✅ Pass Pydantic models DIRECTLY - NO .dict() conversion!
        result = service.generate_recommendations(request.posts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))