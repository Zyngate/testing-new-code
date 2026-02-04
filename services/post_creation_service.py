import tempfile
import os
import requests
import asyncio
from datetime import datetime, timedelta, timezone

from groq import Groq

from database import db
from config import logger, GROQ_API_KEY_CAPTION
from services.image_caption_service import caption_from_image_file
from services.video_caption_service import caption_from_video_file
from services.time_slot_service import (
    get_optimal_times_for_platforms,
    auto_refresh_analytics_for_user,
    PLATFORM_NAME_MAP,
)

# -----------------------------------
# Concurrency control (optimized for Render Standard with 10 API keys)
# -----------------------------------
VIDEO_PROCESS_SEMAPHORE = asyncio.Semaphore(10)  # Max 10 concurrent video processes
API_RATE_LIMIT_SEMAPHORE = asyncio.Semaphore(15)  # Max 15 concurrent API calls


# -----------------------------------
# Media type detection
# -----------------------------------
def detect_media_type(url: str) -> str:
    url = url.lower()
    if url.endswith((".mp4", ".mov", ".webm")):
        return "VIDEO"
    return "IMAGE"


# -----------------------------------
# MAIN PIPELINE
# -----------------------------------
async def create_post_from_uploaded_media(
    user_id: str,
    cloudinary_url: str,
    platform,
    schedule_mode: str = "AUTO",
    scheduled_at: dict | None = None,
    preview_only: bool = False,   # 👈 ADD THIS
):

    """
    SYSTEM-LEVEL AI PIPELINE:
    1. Media understanding (video / image)
    2. Caption + hashtag generation
    3. Best-time recommendation OR manual time
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
        # Auto-refresh analytics
        # -----------------------------
        meta_platforms = [p for p in platforms if p in ["instagram", "facebook"]]
        if meta_platforms:
            try:
                logger.info(f"🔄 Auto-refreshing analytics for {meta_platforms}")
                await auto_refresh_analytics_for_user(user_id, meta_platforms)
            except Exception as e:
                logger.warning(f"⚠️ Auto-refresh failed: {e}")

        # -----------------------------
        # Download media temporarily
        # -----------------------------
        media_type = detect_media_type(cloudinary_url)
        logger.info(f"🧪 Media type detected: {media_type}")

        suffix = ".mp4" if media_type == "VIDEO" else ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            response = requests.get(
                cloudinary_url,
                stream=True,
                timeout=(10, 120),
            )
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)

            local_media_path = tmp.name

        posts = []

        try:
            # -----------------------------
            # Run AI analysis AND time calculation IN PARALLEL
            # (with API rate limiting)
            # -----------------------------
            async def get_ai_captions():
                async with API_RATE_LIMIT_SEMAPHORE:
                    if media_type == "VIDEO":
                        return await caption_from_video_file(
                            video_filepath=local_media_path,
                            platforms=platforms,
                        )
                    else:
                        client = Groq(api_key=GROQ_API_KEY_CAPTION)
                        return await caption_from_image_file(
                            image_filepath=local_media_path,
                            platforms=platforms,
                            client=client,
                        )

            async def get_ai_times():
                return await get_optimal_times_for_platforms(
                    user_id,
                    platforms,
                )

            # Execute both simultaneously
            ai_result, optimal_times = await asyncio.gather(
                get_ai_captions(),
                get_ai_times()
            )

            # -----------------------------
            # Create posts per platform
            # -----------------------------
            for p in platforms:
                caption = ai_result.get("captions", {}).get(p, "")
                hashtags = ai_result.get("platform_hashtags", {}).get(p, [])

                # Handle rate limit failures gracefully
                if not caption:
                    logger.warning(f"⚠️ No caption generated for {p}, using fallback")
                    caption = "Check out this content!"
                
                final_caption = caption
                if hashtags:
                    final_caption += "\n\n" + " ".join(hashtags)

                now = datetime.now(timezone.utc)

                # ---------------------------------
                # MANUAL per-platform scheduling
                # ---------------------------------
                if (
                    schedule_mode == "MANUAL"
                    and scheduled_at
                    and p in scheduled_at
                ):
                    scheduled_time = datetime.fromisoformat(scheduled_at[p])
                    if scheduled_time.tzinfo is None:
                        scheduled_time = scheduled_time.replace(
                            tzinfo=timezone.utc
                        )

                    scheduled_at_final = scheduled_time
                    reason = "User selected time"
                    data_source = "manual"

                # ---------------------------------
                # AUTO scheduling (AI)
                # ---------------------------------
                else:
                    time_info = optimal_times.get(p, {})
                    scheduled_at_final = time_info.get(
                        "scheduledAt",
                        now + timedelta(hours=1),
                    )
                    reason = time_info.get("reason", "AI fallback")
                    data_source = time_info.get("dataSource", "fallback")

                # ---------------------------------
                # Save to DB
                # ---------------------------------
                post_doc = {
                    "userId": user_id,
                    "mediaUrls": [cloudinary_url],
                    "caption": final_caption,
                    "platform": PLATFORM_NAME_MAP.get(p, p.capitalize()),
                    "mediaType": media_type,
                    "scheduledAt": scheduled_at_final,
                    "recommendationReason": reason,
                    "timeDataSource": data_source,
                    "status": "pending",
                    "createdAt": now,
                    "updatedAt": now,
                }

                if not preview_only:
                    await db["scheduledposts"].insert_one(post_doc)


                # ---------------------------------
                # Response-safe object
                # ---------------------------------
                posts.append({
                    "userId": user_id,
                    "mediaUrls": [cloudinary_url],
                    "caption": final_caption,
                    "platform": PLATFORM_NAME_MAP.get(p, p.capitalize()),
                    "mediaType": media_type,
                    "scheduledAt": scheduled_at_final.isoformat(),
                    "recommendationReason": reason,
                    "timeDataSource": data_source,
                    "status": "pending",
                    "createdAt": now.isoformat(),
                    "updatedAt": now.isoformat(),
                })

                logger.info(
                    f"{'📝 Preview generated' if preview_only else '✅ Post scheduled'} | "
                    f"{p} | {scheduled_at_final} | {media_type}"
)


            return {
                "success": True,
                "totalPlatforms": len(posts),
                "results": posts,
            }

        finally:
            # -----------------------------
            # Cleanup
            # -----------------------------
            if os.path.exists(local_media_path):
                os.remove(local_media_path)
