# services/time_slot_service.py
"""
Intelligent Time Slot Service for Auto-Posting
Analyzes user's historical posting data to determine optimal posting times.
Falls back to AI research-based defaults when insufficient data is available.

DATA FRESHNESS POLICY:
- Only analyzes the LAST 25 posts (most recent)
- Considers data stale after 24 hours
- Requires minimum 15 posts to use user data
- Old data is DELETED before storing new data (complete replacement)
"""

import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
import pytz
import httpx
from config import logger
from database import db

# Minimum posts required per platform to use user data
MIN_POSTS_FOR_USER_DATA = 15

# Maximum posts to store and analyze (only recent posts matter)
MAX_POSTS_TO_STORE = 25

# Data is considered stale after this many hours
DATA_STALE_AFTER_HOURS = 24

# Platform name mapping (lowercase -> proper case)
PLATFORM_NAME_MAP = {
    "instagram": "Instagram",
    "youtube": "Youtube",
    "tiktok": "TikTok",
    "facebook": "Facebook",
    "linkedin": "LinkedIn",
    "threads": "Threads",
    "pinterest": "Pinterest",
}

# Platform-specific research data for best posting times
PLATFORM_PEAK_HOURS = {
    'instagram': {
        'peak_hours': [7, 8, 9, 11, 12, 17, 18, 19, 21, 22],
        'best_hours': [8, 12, 18, 21],
        'weekend_hours': [9, 11, 18, 20, 21],
        'weekday_hours': [7, 8, 12, 18, 19, 20, 21],
        'description': 'mornings (7-9 AM), lunch (11 AM-12 PM), evenings (5-7 PM), and late night (9-10 PM)'
    },
    'youtube': {
        'peak_hours': [12, 13, 14, 15, 16, 17, 18, 19, 20],
        'best_hours': [14, 15, 17],
        'weekend_hours': [11, 12, 13, 17, 18],
        'weekday_hours': [14, 15, 16, 17, 18],
        'description': 'early afternoon (2-3 PM) and evening (5-8 PM)'
    },
    'tiktok': {
        'peak_hours': [7, 8, 9, 12, 15, 19, 20, 21, 22],
        'best_hours': [12, 19, 21],
        'weekend_hours': [10, 14, 19, 20, 21, 22],
        'weekday_hours': [12, 18, 19, 20, 21, 22],
        'description': 'lunch (12 PM), evening (7 PM), and late night (9-10 PM)'
    },
    'facebook': {
        'peak_hours': [9, 10, 11, 13, 14, 18, 19],
        'best_hours': [10, 13, 19],
        'weekend_hours': [10, 11, 14, 18, 19],
        'weekday_hours': [9, 10, 11, 13, 14, 18, 19],
        'description': 'mid-morning (9-11 AM), early afternoon (1-2 PM), and evening (6-7 PM)'
    },
    'linkedin': {
        'peak_hours': [7, 8, 9, 10, 12, 17, 18],
        'best_hours': [8, 10, 12],
        'weekend_hours': [10, 11, 12],  # LinkedIn less active on weekends
        'weekday_hours': [7, 8, 9, 10, 12, 17, 18],
        'description': 'early morning (7-8 AM), mid-morning (9-10 AM), and lunch break (12 PM)'
    },
    'threads': {
        'peak_hours': [8, 9, 12, 17, 18, 19, 20, 21],
        'best_hours': [9, 18, 21],
        'weekend_hours': [10, 14, 18, 20, 21],
        'weekday_hours': [8, 9, 12, 17, 18, 19, 20, 21],
        'description': 'morning (8-9 AM), lunch (12 PM), and evening (5-9 PM)'
    },
    'pinterest': {
        'peak_hours': [14, 15, 20, 21, 22, 23],
        'best_hours': [15, 21],
        'weekend_hours': [14, 15, 20, 21, 22],
        'weekday_hours': [14, 15, 20, 21, 22, 23],
        'description': 'afternoon (2-3 PM) and late evening (8-11 PM)'
    },
}

# Minute offsets to stagger posts across platforms
PLATFORM_MINUTE_OFFSET = {
    "instagram": 0,
    "youtube": 5,
    "tiktok": 10,
    "facebook": 15,
    "linkedin": 20,
    "threads": 25,
    "pinterest": 30,
}


class TimeSlotService:
    """Service to determine optimal posting times based on user data or research."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.analytics_collection = db["user_post_analytics"]
    
    async def get_optimal_time_for_platform(
        self,
        platform: str,
        avoid_times: List[datetime] = None
    ) -> Tuple[datetime, str, str]:
        """
        Get the optimal posting time for a specific platform.
        
        Returns:
            Tuple of (scheduled_datetime, reason, data_source)
        """
        platform_lower = platform.lower()
        avoid_times = avoid_times or []
        
        # Check if user has enough FRESH data for this platform
        user_analytics, is_stale = await self._get_user_analytics(platform_lower)
        
        if user_analytics and user_analytics.get("post_count", 0) >= MIN_POSTS_FOR_USER_DATA:
            if is_stale:
                # Data exists but is stale - still use it but log warning
                logger.warning(f"Analytics for {platform_lower} are stale (>24h). Consider refreshing.")
            # Use user's historical data
            return await self._get_time_from_user_data(platform_lower, user_analytics, avoid_times)
        else:
            # Use AI research-based defaults
            return self._get_time_from_research_data(platform_lower, avoid_times)
    
    async def _get_user_analytics(self, platform: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Fetch cached user analytics for a platform.
        
        Returns:
            Tuple of (analytics_data, is_stale)
            - analytics_data: The stored analytics or None
            - is_stale: True if data is older than DATA_STALE_AFTER_HOURS
        """
        try:
            analytics = await self.analytics_collection.find_one({
                "userId": self.user_id,
                "platform": platform
            })
            
            if not analytics:
                return None, False
            
            # Check if analytics are stale
            is_stale = False
            last_updated = analytics.get("updatedAt")
            if last_updated:
                if isinstance(last_updated, datetime):
                    if last_updated.tzinfo is None:
                        last_updated = last_updated.replace(tzinfo=timezone.utc)
                    hours_since_update = (datetime.now(timezone.utc) - last_updated).total_seconds() / 3600
                    is_stale = hours_since_update > DATA_STALE_AFTER_HOURS
                        
            return analytics, is_stale
        except Exception as e:
            logger.error(f"Error fetching user analytics: {e}")
            return None, False
    
    async def _get_time_from_user_data(
        self,
        platform: str,
        analytics: Dict[str, Any],
        avoid_times: List[datetime]
    ) -> Tuple[datetime, str, str]:
        """
        Calculate optimal time from user's historical engagement data.
        Improvements:
        - Suggest best day+hour combo (not just hour)
        - Recency weighting: recent posts weigh more
        - Avoid repeating recent times
        - Platform-specific nuances (future)
        - A/B testing (future)
        """
        now = datetime.now(timezone.utc)
        time_slots = analytics.get("best_time_slots", [])
        recent_time_cutoff = now - timedelta(days=7)  # Avoid repeating slots from last 7 days
        # Gather recent post times to avoid
        recent_post_times = analytics.get("recent_post_times", []) if analytics.get("recent_post_times") else []
        avoid_times = avoid_times or []
        for t in recent_post_times:
            try:
                avoid_times.append(datetime.fromisoformat(t))
            except Exception:
                continue
        # Recency weighting: boost slots with recent high engagement
        weighted_slots = []
        for slot in time_slots:
            # Optionally, add recency_weight if present
            recency_weight = slot.get("recency_weight", 1.0)
            score = slot.get("engagement_score", 0) * recency_weight
            weighted_slots.append({**slot, "weighted_score": score})
        # Sort by weighted score
        weighted_slots.sort(key=lambda x: x["weighted_score"], reverse=True)
        # Try to find a slot that doesn't conflict with avoid_times
        import random
        ab_test_enabled = True  # Set to True to enable A/B testing
        ab_test_fraction = 0.3  # 30% of the time, pick a non-top slot for A/B
        ab_test_slots = []
        for slot in weighted_slots:
            day = slot.get("day")
            hour = slot.get("hour", 12)
            # Platform-specific nuances
            if platform == "linkedin" and day in ["Saturday", "Sunday"]:
                continue  # Avoid weekends for LinkedIn
            if platform == "instagram" and day in ["Saturday", "Sunday"] and hour < 17:
                continue  # Prefer evenings on weekends for Instagram
            # ...add more platform-specific rules as needed...
            ab_test_slots.append(slot)
        # A/B test: randomly select from top 3 slots for a fraction of posts
        selected_slot = None
        if ab_test_enabled and len(ab_test_slots) >= 2 and random.random() < ab_test_fraction:
            selected_slot = random.choice(ab_test_slots[:3])
            ab_test_reason = True
        else:
            selected_slot = ab_test_slots[0] if ab_test_slots else None
            ab_test_reason = False
        if selected_slot:
            day = selected_slot.get("day")
            hour = selected_slot.get("hour", 12)
            days_ahead = (list(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).index(day) - now.weekday()) % 7
            proposed_time = now.replace(hour=hour, minute=PLATFORM_MINUTE_OFFSET.get(platform, 0), second=0, microsecond=0) + timedelta(days=days_ahead)
            if proposed_time <= now:
                proposed_time += timedelta(days=7)
            if not self._time_conflicts(proposed_time, avoid_times):
                reason = (
                    f"Scheduled at {day} {self._format_hour(hour)} based on your best performing posts on {PLATFORM_NAME_MAP.get(platform, platform.capitalize())}. "
                    f"This time slot has historically achieved higher engagement. (Platform-specific rules applied)"
                )
                if ab_test_reason:
                    reason += " [A/B TEST: Randomized top slot]"
                return proposed_time, reason, "user_data"
        # Fallback to research data if no suitable slot found
        return self._get_time_from_research_data(platform, avoid_times)
    
    def _get_time_from_research_data(
        self,
        platform: str,
        avoid_times: List[datetime]
    ) -> Tuple[datetime, str, str]:
        """Get optimal time from AI research-based data."""
        now = datetime.now(timezone.utc)
        is_weekend = now.weekday() >= 5
        
        platform_config = PLATFORM_PEAK_HOURS.get(platform, PLATFORM_PEAK_HOURS.get('instagram'))
        
        if is_weekend:
            hours = platform_config.get('weekend_hours', platform_config.get('best_hours', [12]))
        else:
            hours = platform_config.get('weekday_hours', platform_config.get('best_hours', [12]))
        
        # Try each hour until we find one that doesn't conflict
        for hour in hours:
            proposed_time = now.replace(
                hour=hour,
                minute=PLATFORM_MINUTE_OFFSET.get(platform, 0),
                second=0,
                microsecond=0
            )
            
            if proposed_time <= now:
                proposed_time += timedelta(days=1)
            
            if not self._time_conflicts(proposed_time, avoid_times):
                reason = (
                    f"Scheduled at {self._format_hour(hour)} based on industry research - "
                    f"{platform_config.get('description', 'optimal engagement time')}. "
                    f"Post more to get personalized recommendations!"
                )
                return proposed_time, reason, "research_data"
        
        # Ultimate fallback: 1 hour from now
        fallback_time = now + timedelta(hours=1)
        return fallback_time, "Safe fallback scheduling time.", "fallback"
    
    def _time_conflicts(self, proposed: datetime, avoid_times: List[datetime]) -> bool:
        """Check if proposed time conflicts with any avoid times (within 30 min)."""
        for avoid_time in avoid_times:
            if abs((proposed - avoid_time).total_seconds()) < 1800:  # 30 minutes
                return True
        return False
    
    def _format_hour(self, hour: int) -> str:
        """Format hour as 12-hour time."""
        if hour == 0:
            return "12:00 AM"
        elif hour < 12:
            return f"{hour}:00 AM"
        elif hour == 12:
            return "12:00 PM"
        else:
            return f"{hour - 12}:00 PM"


async def save_user_post_analytics(
    user_id: str,
    platform: str,
    posts_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Save user's post analytics - DELETES OLD DATA FIRST then stores fresh data.
    
    IMPORTANT: 
    - DELETES old analytics data for this user+platform before storing
    - Only stores the LAST 25 posts (most recent)
    - This ensures data is always fresh and not mixed with old data
    
    This should be called when:
    1. User connects a new platform
    2. User opens dashboard/views posts
    3. Before auto-posting (automatic refresh)
    
    Args:
        user_id: The user's ID
        platform: Platform name (lowercase)
        posts_data: List of post data with engagement metrics and timestamps
    
    Returns:
        Analytics summary
    """
    analytics_collection = db["user_post_analytics"]
    
    try:
        # STEP 1: DELETE existing analytics for this user+platform
        delete_result = await analytics_collection.delete_one({
            "userId": user_id,
            "platform": platform.lower()
        })
        
        if delete_result.deleted_count > 0:
            logger.info(f"🗑️ Deleted old analytics for {user_id} on {platform}")
        
        # STEP 2: Sort posts by timestamp (newest first) and take only last MAX_POSTS_TO_STORE
        posts_with_time = []
        for post in posts_data:
            posted_at = post.get("timestamp") or post.get("created_time") or post.get("postedAt")
            if isinstance(posted_at, str):
                try:
                    posted_at = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
                    post["_parsed_time"] = posted_at
                    posts_with_time.append(post)
                except:
                    continue
            elif isinstance(posted_at, datetime):
                post["_parsed_time"] = posted_at
                posts_with_time.append(post)
        
        # Sort by time (newest first) and limit to MAX_POSTS_TO_STORE
        posts_with_time.sort(key=lambda x: x["_parsed_time"], reverse=True)
        recent_posts = posts_with_time[:MAX_POSTS_TO_STORE]
        
        logger.info(f"📊 Storing {len(recent_posts)} most recent posts (out of {len(posts_data)} total)")
        
        # STEP 3: Process posts to calculate best time slots with recency weighting
        time_slot_scores = {}  # {(day_of_week, hour): [engagement_scores]}
        slot_recency = {}  # {(day_of_week, hour): [days_ago]}
        recent_post_times = []
        for post in recent_posts:
            posted_at = post["_parsed_time"]
            recent_post_times.append(posted_at.isoformat())
            # Calculate engagement score
            likes = post.get("likes", 0) or post.get("like_count", 0) or 0
            comments = post.get("comments", 0) or post.get("comments_count", 0) or 0
            shares = post.get("shares", 0) or post.get("share_count", 0) or 0
            views = post.get("views", 0) or post.get("play_count", 0) or 0
            engagement_score = likes + (comments * 2) + (shares * 3) + (views * 0.01)
            day_of_week = posted_at.weekday()
            hour = posted_at.hour
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week]
            key = (day_name, hour)
            if key not in time_slot_scores:
                time_slot_scores[key] = []
                slot_recency[key] = []
            time_slot_scores[key].append(engagement_score)
            # Recency: how many days ago was this post?
            days_ago = (datetime.now(timezone.utc) - posted_at).days
            slot_recency[key].append(days_ago)
        # Calculate average scores per time slot, with recency weighting
        best_time_slots = []
        for (day, hour), scores in time_slot_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            # Recency weighting: more recent posts weigh more (e.g., 1/(1+days_ago))
            recency_weights = [1/(1+d) for d in slot_recency[(day, hour)]]
            recency_weight = sum(recency_weights) / len(recency_weights) if recency_weights else 1.0
            best_time_slots.append({
                "day": day,
                "hour": hour,
                "engagement_score": avg_score,
                "post_count": len(scores),
                "recency_weight": recency_weight
            })
        # Sort by weighted score (engagement * recency)
        best_time_slots.sort(key=lambda x: x["engagement_score"] * x["recency_weight"], reverse=True)
        
        # STEP 4: INSERT new analytics document (no upsert - we already deleted)
        now = datetime.now(timezone.utc)
        analytics_doc = {
            "userId": user_id,
            "platform": platform.lower(),
            "post_count": len(recent_posts),  # Count of stored posts (max 25)
            "total_posts_received": len(posts_data),  # Total posts from API
            "best_time_slots": best_time_slots[:10],  # Keep top 10 slots
            "recent_post_times": recent_post_times,
            "data_source": "meta_api",
            "createdAt": now,
            "updatedAt": now,
        }
        
        await analytics_collection.insert_one(analytics_doc)
        
        logger.info(f"✅ Saved FRESH analytics for {user_id} on {platform}: {len(recent_posts)} posts stored (old data deleted)")
        
        return {
            "success": True,
            "platform": platform,
            "posts_analyzed": len(recent_posts),
            "total_posts_received": len(posts_data),
            "best_time_slots": best_time_slots[:5],
            "message": f"Stored {len(recent_posts)} most recent posts (old data replaced)"
        }
        
    except Exception as e:
        logger.error(f"Error saving user analytics: {e}")
        return {"success": False, "error": str(e)}


async def get_optimal_times_for_platforms(
    user_id: str,
    platforms: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Get optimal posting times for multiple platforms.
    
    Args:
        user_id: The user's ID
        platforms: List of platform names (lowercase)
    
    Returns:
        Dict mapping platform to time info
    """
    service = TimeSlotService(user_id)
    results = {}
    scheduled_times = []  # Track to avoid conflicts
    
    for platform in platforms:
        scheduled_at, reason, data_source = await service.get_optimal_time_for_platform(
            platform,
            avoid_times=scheduled_times
        )
        
        scheduled_times.append(scheduled_at)
        
        results[platform] = {
            "scheduledAt": scheduled_at,
            "reason": reason,
            "dataSource": data_source
        }
    
    return results


# =============================================================================
# META API INTEGRATION - Fetch posts from Instagram/Facebook
# =============================================================================

async def fetch_posts_from_meta_api(
    access_token: str,
    account_id: str,
    platform: str = "instagram",
    limit: int = 25
) -> List[Dict[str, Any]]:
    """
    Fetch posts from Meta API (Instagram/Facebook).
    
    Args:
        access_token: User's Meta API access token
        account_id: Instagram Business Account ID or Facebook Page ID
        platform: 'instagram' or 'facebook'
        limit: Maximum posts to fetch (default 25)
    
    Returns:
        List of posts with engagement metrics
    """
    try:
        # Clean the token - remove any whitespace or hidden characters
        access_token = access_token.strip() if access_token else ""
        
        # Debug log token info (not the actual token!)
        logger.info(f"Token info: length={len(access_token)}, starts_with={access_token[:10] if access_token else 'EMPTY'}...")
        
        # Determine API base URL based on token type
        # Instagram User tokens (IGAB*) should use graph.instagram.com
        # Facebook Page tokens should use graph.facebook.com
        if access_token.startswith("IGAB") or access_token.startswith("IGQV"):
            # Instagram Basic Display API / Instagram Graph API token
            base_url = "https://graph.instagram.com"
            logger.info(f"Using Instagram Graph API (token type: Instagram User Token)")
        else:
            # Facebook Page Access Token
            base_url = "https://graph.facebook.com/v18.0"
            logger.info(f"Using Facebook Graph API")
        
        if platform.lower() == "instagram":
            # Instagram Graph API - fetch media with insights
            url = f"{base_url}/{account_id}/media"
            fields = "id,caption,media_type,timestamp,like_count,comments_count,permalink"
        else:  # facebook
            # Facebook Graph API - fetch page posts
            base_url = "https://graph.facebook.com/v18.0"  # Always use FB for Facebook
            url = f"{base_url}/{account_id}/posts"
            fields = "id,message,created_time,likes.summary(true),comments.summary(true),shares"
        
        params = {
            "access_token": access_token,
            "fields": fields,
            "limit": limit
        }
        
        logger.info(f"Calling Meta API: {url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Meta API error: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            posts = data.get("data", [])
            
            # Normalize the data format
            normalized_posts = []
            for post in posts:
                if platform.lower() == "instagram":
                    normalized_posts.append({
                        "id": post.get("id"),
                        "caption": post.get("caption", ""),
                        "media_type": post.get("media_type"),
                        "timestamp": post.get("timestamp"),
                        "likes": post.get("like_count", 0),
                        "comments": post.get("comments_count", 0),
                        "permalink": post.get("permalink"),
                    })
                else:  # facebook
                    likes_data = post.get("likes", {})
                    comments_data = post.get("comments", {})
                    shares_data = post.get("shares", {})
                    
                    normalized_posts.append({
                        "id": post.get("id"),
                        "caption": post.get("message", ""),
                        "timestamp": post.get("created_time"),
                        "likes": likes_data.get("summary", {}).get("total_count", 0) if isinstance(likes_data, dict) else 0,
                        "comments": comments_data.get("summary", {}).get("total_count", 0) if isinstance(comments_data, dict) else 0,
                        "shares": shares_data.get("count", 0) if isinstance(shares_data, dict) else 0,
                    })
            
            logger.info(f"✅ Fetched {len(normalized_posts)} posts from {platform.capitalize()} API")
            return normalized_posts
            
    except httpx.TimeoutException:
        logger.error(f"Meta API timeout for {platform}")
        return []
    except Exception as e:
        logger.error(f"Error fetching posts from Meta API: {e}")
        return []


async def auto_refresh_analytics_for_user(
    user_id: str,
    platforms: List[str]
) -> Dict[str, Any]:
    """
    Automatically refresh analytics data for a user from Meta API.
    This should be called BEFORE auto-posting to ensure fresh data.
    
    FLOW:
    1. Get user's OAuth credentials from database
    2. Fetch latest posts from Meta API (Instagram/Facebook)
    3. DELETE old analytics and store fresh data
    
    Args:
        user_id: The user's ID
        platforms: List of platform names to refresh
    
    Returns:
        Dict with refresh results per platform
    """
    oauth_collection = db["oauthcredentials"]
    results = {}
    
    for platform in platforms:
        platform_lower = platform.lower()
        
        # Only support Meta platforms (Instagram, Facebook) for now
        if platform_lower not in ["instagram", "facebook"]:
            results[platform] = {
                "success": True,
                "message": f"{platform} does not support auto-refresh (non-Meta platform)",
                "refreshed": False
            }
            continue
        
        try:
            # Get OAuth credentials for this platform
            # Use case-insensitive regex because DB stores "Instagram" not "instagram"
            auth = await oauth_collection.find_one({
                "userId": user_id,
                "platform": {"$regex": f"^{platform_lower}$", "$options": "i"}
            })
            
            if not auth:
                results[platform] = {
                    "success": False,
                    "message": f"No OAuth credentials found for {platform}",
                    "refreshed": False
                }
                continue
            
            access_token = auth.get("accessToken")
            account_id = auth.get("accountId")
            
            if not access_token or account_id is None:
                results[platform] = {
                    "success": False,
                    "message": f"Missing access token or account ID for {platform}",
                    "refreshed": False
                }
                continue
            
            # Fetch latest posts from Meta API
            logger.info(f"🔄 Auto-refreshing analytics for {user_id} on {platform}...")
            posts = await fetch_posts_from_meta_api(
                access_token=access_token,
                account_id=account_id,
                platform=platform_lower,
                limit=MAX_POSTS_TO_STORE
            )
            
            if not posts:
                results[platform] = {
                    "success": False,
                    "message": f"Failed to fetch posts from {platform} API (token may be expired)",
                    "refreshed": False
                }
                continue
            
            # Save analytics (this will DELETE old data first)
            save_result = await save_user_post_analytics(
                user_id=user_id,
                platform=platform_lower,
                posts_data=posts
            )
            
            results[platform] = {
                "success": save_result.get("success", False),
                "message": f"Refreshed {save_result.get('posts_analyzed', 0)} posts from {platform}",
                "refreshed": True,
                "posts_count": save_result.get("posts_analyzed", 0)
            }
            
        except Exception as e:
            logger.error(f"Error refreshing analytics for {platform}: {e}")
            results[platform] = {
                "success": False,
                "message": str(e),
                "refreshed": False
            }
    
    return results
