# stelle_backend/routes/recommendation_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from config import settings
from services.recommendation_service import RecommendationService

router = APIRouter()

service = RecommendationService(settings.GROQ_API_KEY_STELLE_MODEL)

class PostItem(BaseModel):
    link: str
    type: str
    likes: int = 0
    comments: int = 0
    shares: int = 0
    views: int = 0
    reach: int = 0
    posting_time: str
    caption: str
    time_zone: str = "UTC"

class RecommendationRequest(BaseModel):
    posts: List[PostItem]

@router.post("/analyze")
async def analyze_recommendation(request: RecommendationRequest):
    try:
        result = service.generate_recommendations([p.dict() for p in request.posts])
        return {"status": "success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
