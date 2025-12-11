# stelle_backend/routes/recommendation_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from config import settings
from services.combined_engine import RecommendationEngine


# -------------------------
#  Request Model
# -------------------------
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


# -------------------------
#  Router Init
# -------------------------
router = APIRouter()

# Initialize engine once
engine = RecommendationEngine(settings.GROQ_API_KEY)


# -------------------------
#  ENDPOINT: POST /recommend/analyze
# -------------------------
@router.post("/analyze")
async def analyze_recommendations(request: RecommendationRequest):
    try:
        if not request.posts:
            raise HTTPException(status_code=400, detail="No posts provided")

        result = engine.generate_recommendations(request.posts)
        return {"status": "success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
