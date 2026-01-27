import tempfile
import os
import requests
from datetime import datetime, timedelta, timezone
import asyncio

from database import db
from config import logger
from services.video_caption_service import caption_from_video_file
from services.time_slot_service import (
    TimeSlotService,
    get_optimal_times_for_platforms,
    auto_refresh_analytics_for_user,
    PLATFORM_NAME_MAP,
    PLATFORM_MINUTE_OFFSET,
)
from services.recommendation_service import (
    RecommendationService,
    InstagramPost,
    YouTubePost,
    TikTokPost,
    FacebookPost,
    LinkedInPost,
    ThreadsPost,
)

# Limits how many videos are processed at once
VIDEO_PROCESS_SEMAPHORE = asyncio.Semaphore(3)

PLATFORM_MODEL_MAP = {
    "instagram": InstagramPost,
    "youtube": YouTubePost,
    "tiktok": TikTokPost,
    "facebook": FacebookPost,
    "linkedin": LinkedInPost,
    "threads": ThreadsPost,
}

recommendation_service = RecommendationService()


async def create_post_from_uploaded_video(
    user_id: str,
    cloudinary_url: str,
    platform
):
    """
    SYSTEM-LEVEL AI PIPELINE:
    1. Video understanding
    2. Caption + hashtag generation
    3. Content-aware best-time recommendation (uses user data if ≥15 posts, else AI research)
    4. Save scheduled posts (per platform)
    """

    async with VIDEO_PROCESS_SEMAPHORE:

        # -----------------------------
        # Normalize platform input
        # -----------------------------
        if isinstance(platform, list):
            platforms = platform
        else:
            platforms = [platform]

        flat_platforms = []
        for p in platforms:
            if isinstance(p, list):
                flat_platforms.extend(p)
            else:
                flat_platforms.append(p)

        platforms = [p.lower().strip() for p in flat_platforms]

        logger.info("🧠 Starting AI post creation pipeline")

        # -----------------------------
        # 0️⃣ AUTO-REFRESH ANALYTICS (before caption generation)
        # -----------------------------
        # Automatically fetch latest posts from Meta API for Meta platforms
        # Uses OAuth tokens stored in oauthcredentials collection
        meta_platforms = [p for p in platforms if p in ["instagram", "facebook"]]
        if meta_platforms:
            logger.info(f"🔄 Auto-refreshing analytics for platforms: {meta_platforms}")
            try:
                refresh_results = await auto_refresh_analytics_for_user(user_id, meta_platforms)
                for platform_name, result in refresh_results.items():
                    if result.get("refreshed"):
                        logger.info(f"✅ Refreshed {result.get('posts_count', 0)} posts for {platform_name}")
                    elif not result.get("success"):
                        logger.warning(f"⚠️ Could not refresh {platform_name}: {result.get('message')}")
            except Exception as e:
                logger.warning(f"⚠️ Auto-refresh failed, using cached data: {e}")

        # -----------------------------
        # 1️⃣ Download video temporarily
        # -----------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            response = requests.get(
                cloudinary_url,
                stream=True,
                timeout=(10, 120)
            )
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

            local_video_path = tmp.name

        posts = []

        try:
            # ---------------------------------
            # 2️⃣ FULL VIDEO ANALYSIS
            # ---------------------------------
            video_ai_result = await caption_from_video_file(
                video_filepath=local_video_path,
                platforms=platforms
            )

            # ---------------------------------
            # 3️⃣ GET OPTIMAL TIME SLOTS FOR ALL PLATFORMS
            # ---------------------------------
            optimal_times = await get_optimal_times_for_platforms(user_id, platforms)

            # ---------------------------------
            # 4️⃣ PER-PLATFORM POST CREATION
            # ---------------------------------
            for p in platforms:

                caption = video_ai_result.get("captions", {}).get(p, "")
                hashtags = video_ai_result.get("platform_hashtags", {}).get(p, [])

                final_caption = caption or " "
                if hashtags:
                    final_caption += "\n\n" + " ".join(hashtags)

                now = datetime.now(timezone.utc)
                
                # Get optimal time from TimeSlotService
                time_info = optimal_times.get(p, {})
                scheduled_at = time_info.get("scheduledAt", now + timedelta(hours=1))
                reason = time_info.get("reason", "Safe fallback scheduling time.")
                data_source = time_info.get("dataSource", "fallback")

                post_doc = {
                    "userId": user_id,
                    "mediaUrls": [cloudinary_url],
                    "caption": final_caption,
                    "platform": PLATFORM_NAME_MAP.get(p, p.capitalize()),
                    "mediaType": "VIDEO",
                    "scheduledAt": scheduled_at,
                    "recommendationReason": reason,
                    "timeDataSource": data_source,
                    "status": "pending",
                    "createdAt": now,
                    "updatedAt": now,
                }

                await db["scheduledposts"].insert_one(post_doc)
                
                # Convert datetime to JSON-serializable formats for response
                post_response = {
                    "userId": user_id,
                    "mediaUrls": [cloudinary_url],
                    "caption": final_caption,
                    "platform": PLATFORM_NAME_MAP.get(p, p.capitalize()),
                    "mediaType": "VIDEO",
                    "scheduledAt": scheduled_at.isoformat(),
                    "recommendationReason": reason,
                    "timeDataSource": data_source,
                    "status": "pending",
                    "createdAt": now.isoformat(),
                    "updatedAt": now.isoformat(),
                }
                
                posts.append(post_response)

                logger.info(f"✅ Post scheduled | {p} | {scheduled_at} | source: {data_source}")

            return {
                "success": True,
                "totalPlatforms": len(posts),
                "results": posts,
            }

        finally:
            # ---------------------------------
            # 4️⃣ Cleanup temp video
            # ---------------------------------
            if os.path.exists(local_video_path):
                os.remove(local_video_path)
