import tempfile
import os
import requests
from datetime import datetime, timezone

from database import db
from config import logger
from services.video_caption_service import caption_from_video_file
from services.ai_timing_reasoner import (
    ai_recommend_time,
    compute_next_datetime
)



async def create_post_from_uploaded_video(
    user_id: str,
    cloudinary_url: str,
    platform: str
):
    """
    SYSTEM-LEVEL AI PIPELINE:
    1. Video understanding
    2. Caption + hashtag generation
    3. Summary-based best-time recommendation
    4. Save for scheduler
    """

    logger.info("🧠 Starting AI post creation pipeline")

    # -----------------------------
    # 1️⃣ Download video temporarily
    # -----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        response = requests.get(cloudinary_url, stream=True)
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        local_video_path = tmp.name

    try:
        # ---------------------------------
        # 2️⃣ FULL VIDEO ANALYSIS
        # ---------------------------------
        video_ai_result = await caption_from_video_file(
            video_filepath=local_video_path,
            platforms=[platform]
        )

        caption = video_ai_result["captions"][platform]
        hashtags = video_ai_result["platform_hashtags"][platform]

        summary = (
            video_ai_result.get("text_summary")
            or video_ai_result.get("visual_summary")
            or ""
        )

        # ---------------------------------
        # 3️⃣ CONTENT-AWARE TIME RECOMMENDER ✅
        # ---------------------------------
        best_hour, reason = ai_recommend_time(
            summary=summary,
            platform=platform
        )
        scheduled_at = compute_next_datetime(best_hour)


        final_caption = f"{caption}\n\n{' '.join(hashtags)}"

        now = datetime.now(timezone.utc)

        # ---------------------------------
        # 4️⃣ SAVE FOR SCHEDULER
        # ---------------------------------
        post_doc = {
            "userId": user_id,
            "mediaUrls": [cloudinary_url],
            "caption": final_caption,
            "platform": platform,
            "mediaType": "VIDEO",
            "scheduledAt": scheduled_at,
            "recommendationReason": reason,
            "status": "scheduled",
            "createdAt": now,
            "updatedAt": now
        }

        db["scheduledposts"].insert_one(post_doc)

        logger.info(
            f"✅ Post scheduled | {platform} | {scheduled_at} | reason={reason}"
        )

        return post_doc

    finally:
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
