"""
Bulk Post Creation Service

Intelligent bulk posting with:
- Platform-specific posting frequency optimization
- Content-type aware scheduling (Shorts vs Long-form)
- Cross-platform conflict avoidance
- Distributed time allocation across days
"""

import tempfile
import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import cloudinary.uploader
from fastapi import UploadFile
from groq import Groq

from database import db
from config import logger, GROQ_API_KEY_CAPTION
from services.post_creation_service import detect_media_type
from services.video_caption_service import caption_from_video_file
from services.image_caption_service import caption_from_image_file
from services.common_utils import get_user_timezone_from_db
from services.time_slot_service import (
    get_bulk_optimal_times_for_platform,
    get_bulk_optimal_times_multi_platform,
    auto_refresh_analytics_for_user,
    PLATFORM_NAME_MAP,
    BULK_POSTING_CONFIG,
)


# Concurrency control
UPLOAD_SEMAPHORE = asyncio.Semaphore(4)  # Max 4 concurrent uploads
CAPTION_SEMAPHORE = asyncio.Semaphore(6)  # Max 6 concurrent caption generations


async def upload_to_cloudinary(file: UploadFile) -> str:
    """Upload a file to Cloudinary with rate limiting."""
    async with UPLOAD_SEMAPHORE:
        result = cloudinary.uploader.upload(
            file.file,
            resource_type="video",
            folder="scheduler_videos"
        )
        return result["secure_url"]


async def upload_url_to_cloudinary(url: str, resource_type: str = "auto") -> str:
    """Upload a URL to Cloudinary (for pre-existing URLs)."""
    async with UPLOAD_SEMAPHORE:
        result = cloudinary.uploader.upload(
            url,
            resource_type=resource_type,
            folder="scheduler_videos"
        )
        return result["secure_url"]


def detect_content_type(url: str, media_type: str = None) -> str:
    """
    Detect content type for scheduling purposes.
    
    For YouTube:
    - If video duration < 60 seconds: SHORT
    - Otherwise: VIDEO (long-form)
    
    For other platforms:
    - VIDEO, IMAGE, etc.
    """
    url_lower = url.lower()
    
    if media_type:
        return media_type.upper()
    
    if url_lower.endswith((".mp4", ".mov", ".webm", ".avi")):
        return "VIDEO"
    elif url_lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return "IMAGE"
    
    return "VIDEO"  # Default to video


async def process_bulk_videos(
    user_id: str,
    videos: list,
    platforms: list[str],
    growth_mode: str = "optimal"
) -> List[Dict[str, Any]]:
    """
    Process multiple video files for bulk posting.
    
    FLOW:
    1. Upload all videos to Cloudinary in parallel
    2. Pre-calculate optimal time slots for ALL posts across ALL platforms
    3. Generate captions in parallel
    4. Create posts with intelligent time distribution
    
    Args:
        user_id: User ID
        videos: List of UploadFile objects
        platforms: List of platform names
        growth_mode: 'conservative', 'optimal', or 'aggressive'
    
    Returns:
        List of scheduled post info
    """
    num_videos = len(videos)
    num_platforms = len(platforms)
    total_posts = num_videos * num_platforms
    
    logger.info(f"📦 Bulk processing: {num_videos} videos × {num_platforms} platforms = {total_posts} posts")
    
    # Fetch user timezone for proper scheduling
    user_tz = await get_user_timezone_from_db(user_id)
    
    # Step 1: Upload all videos in parallel
    upload_tasks = [upload_to_cloudinary(video) for video in videos]
    cloudinary_urls = await asyncio.gather(*upload_tasks)
    
    logger.info(f"✅ Uploaded {len(cloudinary_urls)} videos to Cloudinary")
    
    # Step 2: Prepare posts_per_platform for time allocation
    posts_per_platform = {p.lower(): num_videos for p in platforms}
    content_types = {p.lower(): "VIDEO" for p in platforms}  # All videos for now
    
    # Step 3: Get intelligent time allocation for ALL posts
    time_slots = await get_bulk_optimal_times_multi_platform(
        user_id=user_id,
        platforms=platforms,
        posts_per_platform=posts_per_platform,
        content_types=content_types,
        growth_mode=growth_mode
    )
    
    logger.info(f"📅 Time slots allocated for {len(time_slots)} platforms")
    
    # Step 4: Process each video with its assigned time slots
    scheduled_posts = []
    
    for video_idx, (video, cloudinary_url) in enumerate(zip(videos, cloudinary_urls)):
        for platform in platforms:
            platform_lower = platform.lower()
            
            # Get the time slot for this video on this platform
            platform_slots = time_slots.get(platform, time_slots.get(platform_lower, []))
            
            if video_idx < len(platform_slots):
                slot = platform_slots[video_idx]
                scheduled_at = slot['scheduledAt']
                reason = slot['reason']
            else:
                # Fallback if not enough slots (shouldn't happen)
                scheduled_at = datetime.now(timezone.utc) + timedelta(hours=video_idx + 1)
                reason = "Fallback scheduling"
            
            scheduled_posts.append({
                "video": video.filename,
                "videoIndex": video_idx + 1,
                "cloudinaryUrl": cloudinary_url,
                "platform": PLATFORM_NAME_MAP.get(platform_lower, platform.capitalize()),
                "scheduledAt": scheduled_at.isoformat() if isinstance(scheduled_at, datetime) else scheduled_at,
                "reason": reason,
                "contentType": "VIDEO",
            })
    
    return scheduled_posts


async def process_bulk_media_urls(
    user_id: str,
    media_urls: List[str],
    platforms: List[str],
    schedule_mode: str = "AUTO",
    scheduled_at_manual: Dict[str, str] = None,
    growth_mode: str = "optimal",
    content_type: str = None,
    preview_only: bool = False,
    start_date: datetime = None,
) -> Dict[str, Any]:
    """
    Process multiple media URLs for bulk posting with intelligent time allocation.
    
    This is the main entry point for bulk posting with:
    - Platform-specific posting frequency limits
    - Intelligent time distribution across days
    - Cross-platform conflict avoidance
    - Content-type aware scheduling
    
    Args:
        user_id: User ID
        media_urls: List of Cloudinary URLs or media URLs
        platforms: List of platform names to post to
        schedule_mode: "AUTO" or "MANUAL"
        scheduled_at_manual: Manual scheduling times (if schedule_mode is MANUAL)
        growth_mode: 'conservative', 'optimal', or 'aggressive'
        content_type: Override content type detection
        preview_only: If True, don't save to database (preview mode)
    
    Returns:
        Dict with success status, results, and scheduling summary
    """
    num_posts = len(media_urls)
    
    # Normalize and flatten platforms
    flat_platforms = []
    for p in platforms:
        if isinstance(p, list):
            flat_platforms.extend(p)
        else:
            flat_platforms.append(p)
    
    platforms_lower = [p.lower().strip() for p in flat_platforms if p]
    
    logger.info(
        f"📦 Bulk processing: {num_posts} media × {len(platforms_lower)} platforms | "
        f"Mode: {schedule_mode} | Growth: {growth_mode}"
    )
    
    # Auto-refresh analytics for Meta platforms
    meta_platforms = [p for p in platforms_lower if p in ["instagram", "facebook"]]
    if meta_platforms:
        try:
            logger.info(f"🔄 Auto-refreshing analytics for {meta_platforms}")
            await auto_refresh_analytics_for_user(user_id, meta_platforms)
        except Exception as e:
            logger.warning(f"⚠️ Auto-refresh failed: {e}")
    
    # Detect content types for each media
    content_types_list = []
    media_types_list = []
    for url in media_urls:
        media_type = detect_media_type(url)
        detected_content = content_type or detect_content_type(url, media_type)
        content_types_list.append(detected_content)
        media_types_list.append(media_type)
    
    # For time allocation, use the most common content type
    from collections import Counter
    content_counter = Counter(content_types_list)
    dominant_content_type = content_counter.most_common(1)[0][0]
    
    # Step 1: Get intelligent time allocation for ALL posts
    if schedule_mode == "AUTO":
        # Calculate posts per platform
        posts_per_platform = {p: num_posts for p in platforms_lower}
        platform_content_types = {}
        for p in platforms_lower:
            # Use SHORT content type for YouTube (optimized for Shorts)
            if p.lower() == "youtube":
                platform_content_types[p] = "SHORT"
            else:
                platform_content_types[p] = dominant_content_type
        
        time_slots = await get_bulk_optimal_times_multi_platform(
            user_id=user_id,
            platforms=platforms_lower,
            posts_per_platform=posts_per_platform,
            content_types=platform_content_types,
            growth_mode=growth_mode,
            start_date=start_date
        )
        
        logger.info(f"📅 Allocated time slots for {len(time_slots)} platforms")
    else:
        time_slots = {}
    
    # Step 2: Generate captions in parallel (if needed)
    async def generate_caption_for_media(url: str, idx: int):
        """Generate caption for a single media item."""
        async with CAPTION_SEMAPHORE:
            try:
                import tempfile
                import requests
                
                media_type = media_types_list[idx]
                suffix = ".mp4" if media_type == "VIDEO" else ".png"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    response = requests.get(url, stream=True, timeout=(10, 60))
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    local_path = tmp.name
                
                try:
                    if media_type == "VIDEO":
                        result = await caption_from_video_file(
                            video_filepath=local_path,
                            platforms=platforms_lower,
                        )
                    else:
                        client = Groq(api_key=GROQ_API_KEY_CAPTION)
                        result = await caption_from_image_file(
                            image_filepath=local_path,
                            platforms=platforms_lower,
                            client=client,
                        )
                    return result
                finally:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        
            except Exception as e:
                logger.error(f"Caption generation failed for {url}: {e}")
                return {"captions": {}, "platform_hashtags": {}}
    
    # Generate all captions in parallel
    caption_tasks = [generate_caption_for_media(url, idx) for idx, url in enumerate(media_urls)]
    caption_results = await asyncio.gather(*caption_tasks)
    
    logger.info(f"✅ Generated captions for {len(caption_results)} media items")
    
    # Step 3: Create posts with allocated time slots
    all_posts = []
    posts_to_insert = []  # Deferred DB inserts
    now = datetime.now(timezone.utc)
    
    for media_idx, (media_url, caption_result) in enumerate(zip(media_urls, caption_results)):
        for platform_idx, platform in enumerate(platforms_lower):
            captions_dict = caption_result.get("captions", {})
            hashtags_dict = caption_result.get("platform_hashtags", {})
            titles_dict = caption_result.get("titles", {})
            # Always get a caption for every platform
            caption = captions_dict.get(platform)
            if not caption:
                # Fallback: use Pinterest or YouTube caption if available, else generic
                fallback_caption = captions_dict.get("pinterest") or captions_dict.get("youtube") or "Check out this content!"
                caption = fallback_caption
            hashtags = hashtags_dict.get(platform, [])
            
            final_caption = caption
            if hashtags:
                final_caption += "\n\n" + " ".join(hashtags)

            # Scheduling logic unchanged
            if schedule_mode == "MANUAL" and scheduled_at_manual:
                manual_time = scheduled_at_manual.get(platform)
                if manual_time:
                    scheduled_time = datetime.fromisoformat(manual_time)
                    if scheduled_time.tzinfo is None:
                        scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
                else:
                    scheduled_time = now + timedelta(hours=media_idx + 1)
                reason = "User selected time"
                data_source = "manual"
            else:
                platform_slots = time_slots.get(platform, [])
                if media_idx < len(platform_slots):
                    slot = platform_slots[media_idx]
                    scheduled_time = slot['scheduledAt']
                    reason = slot['reason']
                    data_source = slot.get('dataSource', 'research_data')
                    local_time = slot.get('localTime')
                    user_timezone = slot.get('timezone', 'UTC')
                else:
                    scheduled_time = now + timedelta(hours=media_idx + 1)
                    reason = "Fallback scheduling - all optimal slots used"
                    data_source = "fallback"
                    local_time = scheduled_time.isoformat()
                    user_timezone = 'UTC'

            post_obj = {
                "userId": user_id,
                "mediaUrls": [media_url],
                "caption": final_caption,
                "platform": PLATFORM_NAME_MAP.get(platform, platform.capitalize()),
                "mediaType": media_types_list[media_idx],
                "scheduledAt": local_time if 'local_time' in locals() else (scheduled_time.isoformat() if isinstance(scheduled_time, datetime) else scheduled_time),
                "scheduledAtUTC": scheduled_time.isoformat() if isinstance(scheduled_time, datetime) else scheduled_time,
                "localTime": local_time if 'local_time' in locals() else (scheduled_time.isoformat() if isinstance(scheduled_time, datetime) else scheduled_time),
                "timezone": user_timezone if 'user_timezone' in locals() else 'UTC',
                "scheduledAtObj": scheduled_time,
                "recommendationReason": reason,
                "timeDataSource": data_source,
                "status": "pending",
                "createdAt": now.isoformat(),
                "updatedAt": now.isoformat(),
                "postIndex": 0,
                "totalPosts": num_posts,
            }
            # Only add title for Pinterest and YouTube
            if platform in ("youtube", "pinterest"):
                post_obj["title"] = titles_dict.get(platform, "")
            all_posts.append(post_obj)
    # Sort all posts by scheduled time
    all_posts.sort(key=lambda p: p['scheduledAtObj'])
    
    # Reassign postIndex based on chronological order per platform
    platform_counters = {p: 0 for p in platforms_lower}
    for post in all_posts:
        platform_key = post['platform'].lower()
        platform_counters[platform_key] += 1
        post['postIndex'] = platform_counters[platform_key]
        
        # Remove the datetime object from response (no longer needed)
        scheduled_time_obj = post.pop('scheduledAtObj')
        
        # NOW build and insert database document with correct index
        if not preview_only:
            post_doc = {
                "userId": post['userId'],
                "mediaUrls": post['mediaUrls'],
                "caption": post['caption'],
                "platform": post['platform'],
                "mediaType": post['mediaType'],
                "scheduledAt": scheduled_time_obj,
                "recommendationReason": post['recommendationReason'],
                "timeDataSource": post['timeDataSource'],
                "status": "pending",
                "createdAt": now,
                "updatedAt": now,
                "postIndex": post['postIndex'],  # Correct chronological index
                "totalPosts": num_posts,
            }
            posts_to_insert.append(post_doc)
    
    # Bulk insert to database (more efficient than individual inserts)
    if not preview_only and posts_to_insert:
        await db["scheduledposts"].insert_many(posts_to_insert)
    
    # Build scheduling summary
    summary = _build_scheduling_summary(all_posts, platforms_lower, num_posts)
    
    action_text = "Preview generated" if preview_only else "Posts scheduled"
    logger.info(f"✅ {action_text}: {len(all_posts)} posts across {len(platforms)} platforms")
    
    return {
        "success": True,
        "totalPosts": len(all_posts),
        "totalPlatforms": len(platforms),
        "postsPerPlatform": num_posts,
        "growthMode": growth_mode,
        "results": all_posts,
        "summary": summary,
        "previewOnly": preview_only,
        "startDate": start_date.isoformat() if start_date else None,
    }


def _build_scheduling_summary(
    posts: List[Dict],
    platforms: List[str],
    num_posts: int
) -> Dict[str, Any]:
    """Build a summary of the scheduling distribution."""
    summary = {
        "byPlatform": {},
        "byDay": {},
        "dateRange": {},
    }
    
    all_times = []
    
    for platform in platforms:
        platform_posts = [p for p in posts if p['platform'].lower() == platform.lower() 
                         or PLATFORM_NAME_MAP.get(platform, '').lower() == p['platform'].lower()]
        
        if platform_posts:
            times = []
            for p in platform_posts:
                # Use localTime for proper day grouping in user's timezone
                t = p.get('localTime', p['scheduledAt'])
                if isinstance(t, str):
                    t = datetime.fromisoformat(t.replace('Z', '+00:00'))
                times.append(t)
                all_times.append(t)
            
            times.sort()
            
            # Group by day
            days_used = set()
            for t in times:
                days_used.add(t.strftime("%A"))
            
            platform_config = BULK_POSTING_CONFIG.get(platform.lower(), {})
            
            summary["byPlatform"][platform] = {
                "postCount": len(platform_posts),
                "daysUsed": list(days_used),
                "firstPost": times[0].isoformat() if times else None,
                "lastPost": times[-1].isoformat() if times else None,
                "strategy": platform_config.get('algorithm_notes', 'Standard scheduling'),
            }
    
    # Overall date range
    if all_times:
        all_times.sort()
        summary["dateRange"] = {
            "start": all_times[0].isoformat(),
            "end": all_times[-1].isoformat(),
            "totalDays": (all_times[-1].date() - all_times[0].date()).days + 1,
        }
        
        # Count by day
        for t in all_times:
            day = t.strftime("%A, %b %d")
            summary["byDay"][day] = summary["byDay"].get(day, 0) + 1
    
    return summary


async def get_bulk_scheduling_preview(
    user_id: str,
    num_posts: int,
    platforms: List[str],
    content_type: str = "VIDEO",
    growth_mode: str = "optimal",
    start_date: datetime = None,
) -> Dict[str, Any]:
    """
    Get a preview of how bulk posts would be scheduled.
    
    This is useful for showing users the distribution before they upload.
    
    Args:
        user_id: User ID
        num_posts: Number of posts planned
        platforms: List of platforms
        content_type: Type of content
        growth_mode: Scheduling aggressiveness
    
    Returns:
        Preview of scheduling distribution
    """
    platforms_lower = [p.lower() for p in platforms]
    
    # Fetch user timezone for proper scheduling
    user_timezone = await get_user_timezone_from_db(user_id)
    
    posts_per_platform = {p: num_posts for p in platforms_lower}
    content_types = {p: content_type for p in platforms_lower}
    
    time_slots = await get_bulk_optimal_times_multi_platform(
        user_id=user_id,
        platforms=platforms_lower,
        posts_per_platform=posts_per_platform,
        content_types=content_types,
        growth_mode=growth_mode,
        start_date=start_date
    )
    
    preview = {
        "totalPosts": num_posts * len(platforms),
        "postsPerPlatform": num_posts,
        "platforms": [],
        "growthMode": growth_mode,
        "user_timezone": user_timezone,
    }
    
    for platform in platforms_lower:
        slots = time_slots.get(platform, [])
        
        platform_config = BULK_POSTING_CONFIG.get(platform, {})
        posts_per_day = platform_config.get('posts_per_day', {}).get(growth_mode, 2)
        
        days_used = set()
        for slot in slots:
            days_used.add(slot['dayOfWeek'])
        
        preview["platforms"].append({
            "name": PLATFORM_NAME_MAP.get(platform, platform.capitalize()),
            "postCount": len(slots),
            "postsPerDay": posts_per_day,
            "daysNeeded": len(days_used),
            "daysUsed": list(days_used),
            "slots": [
                {
                    "postIndex": s['postIndex'],
                    "day": s['dayOfWeek'],
                    "time": s['localTime'],
                    "reason": s['reason'][:100] + "..." if len(s.get('reason', '')) > 100 else s.get('reason', ''),
                }
                for s in slots
            ],
            "strategy": platform_config.get('algorithm_notes', ''),
        })
    
    return preview
