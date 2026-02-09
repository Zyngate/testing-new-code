# stelle_backend/services/video_cache_repo.py
"""
Cache repository for video analysis results.
Stores expensive AI outputs (vision, STT) so regenerate is instant.
Does NOT cache: captions, hashtags, titles (cheap to regenerate, platform-specific)
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from database import db
from config import logger


async def get_cached_video_analysis(video_hash: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached video analysis by video hash.
    Returns None if not found.
    """
    try:
        cache_collection = db["video_analysis_cache"]
        doc = await cache_collection.find_one({"video_hash": video_hash})
        
        if not doc:
            return None
        
        logger.info(f"Cache HIT for video hash: {video_hash[:16]}...")
        
        return {
            "transcript": doc.get("transcript", ""),
            "visual_summary": doc.get("visual_summary", ""),
            "visual_captions": doc.get("visual_captions", []),
            "detected_texts": doc.get("detected_texts", []),
            "ocr_text_combined": doc.get("ocr_text_combined", ""),
            "detected_person": doc.get("detected_person"),
            "marketing_prompt": doc.get("marketing_prompt", ""),
            "objects": doc.get("objects", []),
            "actions": doc.get("actions", []),
        }
    except Exception as e:
        logger.error(f"Error retrieving video cache: {e}")
        return None


async def save_video_analysis(video_hash: str, data: Dict[str, Any]) -> bool:
    """
    Save video analysis to cache.
    Uses upsert to handle both insert and update.
    """
    try:
        cache_collection = db["video_analysis_cache"]
        
        doc = {
            "video_hash": video_hash,
            "transcript": data.get("transcript", ""),
            "visual_summary": data.get("visual_summary", ""),
            "visual_captions": data.get("visual_captions", []),
            "detected_texts": data.get("detected_texts", []),
            "ocr_text_combined": data.get("ocr_text_combined", ""),
            "detected_person": data.get("detected_person"),
            "marketing_prompt": data.get("marketing_prompt", ""),
            "objects": data.get("objects", []),
            "actions": data.get("actions", []),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        
        await cache_collection.update_one(
            {"video_hash": video_hash},
            {"$set": doc},
            upsert=True
        )
        
        logger.info(f"Cached video analysis for hash: {video_hash[:16]}...")
        return True
        
    except Exception as e:
        logger.error(f"Error saving video cache: {e}")
        return False


async def delete_video_cache(video_hash: str) -> bool:
    """Delete cached analysis for a specific video."""
    try:
        cache_collection = db["video_analysis_cache"]
        result = await cache_collection.delete_one({"video_hash": video_hash})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Error deleting video cache: {e}")
        return False
