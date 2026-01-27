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

ENGAGEMENT OPTIMIZATION:
- Platform-specific peak hours based on industry research (2024-2025 data)
- Timezone-aware scheduling with user's local timezone
- Precise minute-level scheduling (not just hour)
- Avoids repeating times from recent posts
- Day-of-week optimization per platform
"""

import os
import random
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

# Default timezone if user doesn't have one set
DEFAULT_TIMEZONE = "UTC"

# Conflict window in minutes - posts scheduled within this window are considered conflicts
CONFLICT_WINDOW_MINUTES = 45

# Platform name mapping (lowercase -> proper case)
PLATFORM_NAME_MAP = {
    "instagram": "Instagram",
    "youtube": "Youtube",
    "tiktok": "TikTok",
    "facebook": "Facebook",
    "linkedin": "LinkedIn",
    "threads": "Threads",
    "pinterest": "Pinterest",
    "twitter": "Twitter/X",
    "x": "Twitter/X",
}

# =============================================================================
# PLATFORM-SPECIFIC PEAK HOURS - Based on 2024-2025 Industry Research
# Each platform has unique engagement patterns based on user behavior
# =============================================================================

PLATFORM_PEAK_HOURS = {
    'instagram': {
        # Instagram: Visual content - users scroll during commute, lunch, and evening relaxation
        # Day-wise best hours (engagement varies by day)
        'day_wise_hours': {
            'Monday': [8, 11, 17, 18, 19],
            'Tuesday': [7, 8, 11, 18, 19, 20],
            'Wednesday': [8, 11, 12, 18, 19, 20],
            'Thursday': [8, 11, 17, 18, 19],
            'Friday': [8, 11, 14, 17, 18, 19],
            'Saturday': [10, 11, 14, 17, 19, 20, 21],
            'Sunday': [10, 11, 14, 17, 18, 19, 20]
        },
        'peak_hours': [7, 8, 9, 11, 12, 17, 18, 19, 20, 21],
        'best_hours': [8, 11, 18, 20],
        'weekend_hours': [10, 11, 14, 17, 19, 20, 21],
        'weekday_hours': [7, 8, 11, 12, 17, 18, 19, 20],
        'optimal_minutes': [0, 15, 30, 45],
        'prime_minutes': [5, 20, 35, 50],
        'restrict_days': False,  # No day restriction - post any day
        'engagement_multiplier': {
            'Monday': 0.95, 'Tuesday': 1.1, 'Wednesday': 1.15,
            'Thursday': 1.0, 'Friday': 1.05, 'Saturday': 0.95, 'Sunday': 0.9
        },
        'description': 'Best at 8 AM commute, 11 AM break, 6-8 PM evening scroll.'
    },
    'youtube': {
        # YouTube: Long-form content - users watch during afternoon/evening leisure time
        'day_wise_hours': {
            'Monday': [12, 14, 15, 17, 18, 19],
            'Tuesday': [12, 14, 15, 17, 18, 19],
            'Wednesday': [12, 14, 15, 17, 18, 19, 20],
            'Thursday': [14, 15, 16, 17, 18, 19, 20],
            'Friday': [14, 15, 16, 17, 18, 19, 20, 21],
            'Saturday': [10, 11, 12, 14, 15, 19, 20, 21],
            'Sunday': [10, 11, 12, 14, 17, 18, 19, 20]
        },
        'peak_hours': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        'best_hours': [14, 15, 17, 20],
        'weekend_hours': [10, 11, 12, 14, 15, 19, 20, 21],
        'weekday_hours': [12, 14, 15, 16, 17, 18, 19, 20],
        'optimal_minutes': [0, 15, 30],
        'prime_minutes': [0, 15, 45],
        'restrict_days': False,
        'engagement_multiplier': {
            'Monday': 0.9, 'Tuesday': 0.95, 'Wednesday': 1.0,
            'Thursday': 1.1, 'Friday': 1.15, 'Saturday': 1.2, 'Sunday': 1.1
        },
        'description': 'Post 2-3 PM weekdays (before evening viewers). Weekend mornings great for binge-watching.'
    },
    'tiktok': {
        # TikTok: Short-form, addictive - users check frequently, evening/night peak
        'day_wise_hours': {
            'Monday': [7, 8, 12, 19, 20, 21, 22],
            'Tuesday': [6, 7, 12, 19, 20, 21, 22],
            'Wednesday': [7, 12, 15, 19, 20, 21],
            'Thursday': [7, 12, 15, 19, 20, 21, 22],
            'Friday': [7, 12, 15, 19, 20, 21, 22, 23],
            'Saturday': [9, 10, 11, 12, 15, 19, 20, 21, 22, 23],
            'Sunday': [9, 10, 11, 12, 15, 19, 20, 21, 22]
        },
        'peak_hours': [6, 7, 8, 10, 12, 15, 19, 20, 21, 22, 23],
        'best_hours': [7, 12, 19, 21],
        'weekend_hours': [9, 10, 11, 12, 15, 19, 20, 21, 22, 23],
        'weekday_hours': [6, 7, 8, 12, 15, 19, 20, 21, 22],
        'optimal_minutes': [0, 17, 33, 47],
        'prime_minutes': [3, 18, 33, 48],
        'restrict_days': False,
        'engagement_multiplier': {
            'Monday': 0.95, 'Tuesday': 1.1, 'Wednesday': 1.0,
            'Thursday': 1.1, 'Friday': 1.15, 'Saturday': 1.1, 'Sunday': 0.95
        },
        'description': 'Early AM (6-7), lunch break (12), prime time (7-9 PM), late night (10-11 PM).'
    },
    'facebook': {
        # Facebook: Broad demographics - morning news, lunch scroll, evening family time
        'day_wise_hours': {
            'Monday': [9, 10, 11, 13, 18, 19],
            'Tuesday': [9, 10, 11, 13, 18, 19],
            'Wednesday': [9, 10, 11, 12, 13, 18, 19, 20],
            'Thursday': [9, 10, 11, 13, 18, 19],
            'Friday': [9, 10, 11, 13, 14, 18, 19],
            'Saturday': [9, 10, 11, 12, 13, 17, 18, 19],
            'Sunday': [9, 10, 11, 12, 13, 17, 18, 19]
        },
        'peak_hours': [8, 9, 10, 11, 12, 13, 18, 19, 20],
        'best_hours': [9, 13, 19],
        'weekend_hours': [9, 10, 11, 12, 13, 17, 18, 19],
        'weekday_hours': [8, 9, 10, 11, 12, 13, 18, 19, 20],
        'optimal_minutes': [0, 15, 30, 45],
        'prime_minutes': [7, 22, 37, 52],
        'restrict_days': False,
        'engagement_multiplier': {
            'Monday': 0.9, 'Tuesday': 1.0, 'Wednesday': 1.1,
            'Thursday': 1.05, 'Friday': 1.0, 'Saturday': 0.9, 'Sunday': 0.85
        },
        'description': 'Mid-morning (9-11 AM), lunch (1 PM), early evening (7 PM).'
    },
    'linkedin': {
        # LinkedIn: Professional network - business hours ONLY, avoid weekends
        # This is the ONLY platform with day restrictions
        'day_wise_hours': {
            'Monday': [8, 9, 10, 12, 17, 18],
            'Tuesday': [7, 8, 9, 10, 12, 17, 18],
            'Wednesday': [7, 8, 9, 10, 11, 12, 17, 18],
            'Thursday': [7, 8, 9, 10, 12, 17, 18],
            'Friday': [8, 9, 10, 12],
            'Saturday': [],  # Avoid weekend
            'Sunday': []  # Avoid weekend
        },
        'peak_hours': [7, 8, 9, 10, 11, 12, 17, 18],
        'best_hours': [8, 10, 12, 17],
        'weekend_hours': [],  # LinkedIn is not active on weekends
        'weekday_hours': [7, 8, 9, 10, 11, 12, 17, 18],
        'optimal_minutes': [0, 15, 30, 45],
        'prime_minutes': [5, 20, 35, 50],
        'restrict_days': True,  # ONLY platform with day restriction
        'avoid_days': ['Saturday', 'Sunday'],  # Professional network - avoid weekends
        'engagement_multiplier': {
            'Monday': 0.9, 'Tuesday': 1.2, 'Wednesday': 1.25,
            'Thursday': 1.15, 'Friday': 0.85, 'Saturday': 0.3, 'Sunday': 0.25
        },
        'description': 'Business hours only (7 AM - 6 PM). Tue-Thu highest engagement. Avoid weekends.'
    },
    'threads': {
        # Threads: Text-based social - similar to Twitter patterns
        'day_wise_hours': {
            'Monday': [8, 9, 12, 17, 18, 19, 20],
            'Tuesday': [8, 9, 12, 17, 18, 19, 20, 21],
            'Wednesday': [8, 9, 12, 17, 18, 19, 20],
            'Thursday': [8, 9, 12, 17, 18, 19, 20],
            'Friday': [8, 9, 12, 17, 18, 19],
            'Saturday': [10, 11, 13, 18, 19, 20, 21],
            'Sunday': [10, 11, 13, 18, 19, 20, 21]
        },
        'peak_hours': [7, 8, 9, 12, 13, 17, 18, 19, 20, 21],
        'best_hours': [8, 12, 18, 20],
        'weekend_hours': [10, 11, 13, 18, 19, 20, 21],
        'weekday_hours': [7, 8, 9, 12, 13, 17, 18, 19, 20],
        'optimal_minutes': [0, 12, 24, 36, 48],
        'prime_minutes': [3, 15, 27, 39, 51],
        'restrict_days': False,
        'engagement_multiplier': {
            'Monday': 1.05, 'Tuesday': 1.1, 'Wednesday': 1.05,
            'Thursday': 1.0, 'Friday': 0.95, 'Saturday': 0.85, 'Sunday': 0.9
        },
        'description': 'Morning (8-9 AM), lunch (12 PM), evening (6-9 PM).'
    },
    'pinterest': {
        # Pinterest: Visual discovery - evening browsing, weekend planning
        'day_wise_hours': {
            'Monday': [12, 13, 14, 20, 21, 22],
            'Tuesday': [12, 13, 14, 20, 21, 22],
            'Wednesday': [12, 13, 14, 15, 20, 21, 22],
            'Thursday': [12, 13, 14, 20, 21, 22],
            'Friday': [12, 13, 14, 15, 20, 21, 22, 23],
            'Saturday': [10, 11, 14, 15, 16, 20, 21, 22],
            'Sunday': [10, 11, 14, 15, 16, 20, 21, 22]
        },
        'peak_hours': [12, 13, 14, 15, 20, 21, 22, 23],
        'best_hours': [14, 21, 22],
        'weekend_hours': [10, 11, 14, 15, 16, 20, 21, 22],
        'weekday_hours': [12, 13, 14, 15, 20, 21, 22, 23],
        'optimal_minutes': [0, 20, 40],
        'prime_minutes': [5, 25, 45],
        'restrict_days': False,
        'engagement_multiplier': {
            'Monday': 0.85, 'Tuesday': 0.9, 'Wednesday': 0.95,
            'Thursday': 0.95, 'Friday': 1.05, 'Saturday': 1.15, 'Sunday': 1.1
        },
        'description': 'Afternoon (2-3 PM), late evening (8-11 PM). Weekends great for planning/inspiration.'
    },
    'twitter': {
        # Twitter/X: Real-time news/conversation
        'day_wise_hours': {
            'Monday': [8, 9, 12, 13, 17, 18, 19],
            'Tuesday': [8, 9, 12, 13, 17, 18, 19, 20],
            'Wednesday': [8, 9, 12, 13, 17, 18, 19, 20],
            'Thursday': [8, 9, 12, 13, 17, 18, 19, 20],
            'Friday': [8, 9, 12, 13, 17, 18, 19],
            'Saturday': [9, 10, 12, 17, 18, 19],
            'Sunday': [9, 10, 12, 17, 18, 19]
        },
        'peak_hours': [7, 8, 9, 12, 13, 17, 18, 19, 20],
        'best_hours': [8, 12, 17, 19],
        'weekend_hours': [9, 10, 12, 17, 18, 19],
        'weekday_hours': [7, 8, 9, 12, 13, 17, 18, 19, 20],
        'optimal_minutes': [0, 10, 20, 30, 40, 50],
        'prime_minutes': [5, 15, 25, 35, 45, 55],
        'restrict_days': False,
        'engagement_multiplier': {
            'Monday': 0.95, 'Tuesday': 1.05, 'Wednesday': 1.1,
            'Thursday': 1.05, 'Friday': 1.0, 'Saturday': 0.85, 'Sunday': 0.8
        },
        'description': 'Morning (8-9 AM), lunch (12-1 PM), evening (5-7 PM).'
    },
    'x': {
        # Alias for Twitter
        'day_wise_hours': {
            'Monday': [8, 9, 12, 13, 17, 18, 19],
            'Tuesday': [8, 9, 12, 13, 17, 18, 19, 20],
            'Wednesday': [8, 9, 12, 13, 17, 18, 19, 20],
            'Thursday': [8, 9, 12, 13, 17, 18, 19, 20],
            'Friday': [8, 9, 12, 13, 17, 18, 19],
            'Saturday': [9, 10, 12, 17, 18, 19],
            'Sunday': [9, 10, 12, 17, 18, 19]
        },
        'peak_hours': [7, 8, 9, 12, 13, 17, 18, 19, 20],
        'best_hours': [8, 12, 17, 19],
        'weekend_hours': [9, 10, 12, 17, 18, 19],
        'weekday_hours': [7, 8, 9, 12, 13, 17, 18, 19, 20],
        'optimal_minutes': [0, 10, 20, 30, 40, 50],
        'prime_minutes': [5, 15, 25, 35, 45, 55],
        'restrict_days': False,
        'engagement_multiplier': {
            'Monday': 0.95, 'Tuesday': 1.05, 'Wednesday': 1.1,
            'Thursday': 1.05, 'Friday': 1.0, 'Saturday': 0.85, 'Sunday': 0.8
        },
        'description': 'Morning (8-9 AM), lunch (12-1 PM), evening (5-7 PM).'
    },
}

# Default fallback for unknown platforms
DEFAULT_PLATFORM_CONFIG = {
    'day_wise_hours': {
        'Monday': [9, 10, 11, 12, 17, 18, 19],
        'Tuesday': [9, 10, 11, 12, 17, 18, 19, 20],
        'Wednesday': [9, 10, 11, 12, 17, 18, 19, 20],
        'Thursday': [9, 10, 11, 12, 17, 18, 19],
        'Friday': [9, 10, 11, 12, 17, 18, 19],
        'Saturday': [10, 11, 14, 18, 19],
        'Sunday': [10, 11, 14, 18, 19]
    },
    'peak_hours': [9, 10, 11, 12, 17, 18, 19, 20],
    'best_hours': [10, 12, 18],
    'weekend_hours': [10, 11, 18, 19],
    'weekday_hours': [9, 10, 11, 12, 17, 18, 19],
    'optimal_minutes': [0, 15, 30, 45],
    'prime_minutes': [5, 20, 35, 50],
    'restrict_days': False,  # No day restriction by default
    'engagement_multiplier': {
        'Monday': 0.95, 'Tuesday': 1.05, 'Wednesday': 1.05,
        'Thursday': 1.0, 'Friday': 1.0, 'Saturday': 0.9, 'Sunday': 0.85
    },
    'description': 'Standard social media timing - mid-morning, lunch, and evening.'
}

# Platform-specific minute offsets for multi-platform posting (stagger posts)
PLATFORM_MINUTE_OFFSET = {
    "instagram": 0,
    "youtube": 7,
    "tiktok": 14,
    "facebook": 21,
    "linkedin": 28,
    "threads": 35,
    "pinterest": 42,
    "twitter": 49,
    "x": 49,
}


class TimeSlotService:
    """
    Service to determine optimal posting times based on user data or research.
    
    Features:
    - Platform-specific peak hours with minute-level precision
    - Timezone-aware scheduling
    - Avoids recent post times to prevent repetition
    - Day-of-week optimization
    - Engagement multiplier based on platform research
    """
    
    def __init__(self, user_id: str, user_timezone: str = None):
        self.user_id = user_id
        self.user_timezone = user_timezone or DEFAULT_TIMEZONE
        self.analytics_collection = db["user_post_analytics"]
        
        # Validate timezone
        try:
            self.tz = pytz.timezone(self.user_timezone)
        except Exception:
            logger.warning(f"Invalid timezone '{self.user_timezone}', falling back to UTC")
            self.tz = pytz.UTC
            self.user_timezone = "UTC"
    
    async def get_optimal_time_for_platform(
        self,
        platform: str,
        avoid_times: List[datetime] = None,
        last_post_time: datetime = None
    ) -> Tuple[datetime, str, str, Dict[str, Any]]:
        """
        Get the optimal posting time for a specific platform.
        
        Args:
            platform: Platform name (e.g., 'instagram', 'facebook')
            avoid_times: List of datetime objects to avoid scheduling near
            last_post_time: The most recent post time for this platform (to avoid repetition)
        
        Returns:
            Tuple of (scheduled_datetime, reason, data_source, metadata)
            - scheduled_datetime: The recommended posting time (timezone-aware)
            - reason: Human-readable explanation
            - data_source: 'user_data', 'research_data', or 'fallback'
            - metadata: Additional info (timezone, engagement_score, etc.)
        """
        platform_lower = platform.lower()
        avoid_times = avoid_times or []
        
        # Add last post time to avoid list if provided
        if last_post_time:
            avoid_times.append(last_post_time)
        
        # Fetch last post time from analytics if not provided
        if not last_post_time:
            last_post_time = await self._get_last_post_time(platform_lower)
            if last_post_time:
                avoid_times.append(last_post_time)
        
        # Check if user has enough FRESH data for this platform
        user_analytics, is_stale = await self._get_user_analytics(platform_lower)
        
        if user_analytics and user_analytics.get("post_count", 0) >= MIN_POSTS_FOR_USER_DATA:
            if is_stale:
                logger.warning(f"Analytics for {platform_lower} are stale (>24h). Consider refreshing.")
            # Use user's historical data
            return await self._get_time_from_user_data(platform_lower, user_analytics, avoid_times)
        else:
            # Use AI research-based defaults
            return self._get_time_from_research_data(platform_lower, avoid_times)
    
    async def _get_last_post_time(self, platform: str) -> Optional[datetime]:
        """Get the most recent post time for a platform from analytics."""
        try:
            analytics = await self.analytics_collection.find_one({
                "userId": self.user_id,
                "platform": platform
            })
            
            if analytics and analytics.get("recent_post_times"):
                recent_times = analytics.get("recent_post_times", [])
                if recent_times:
                    # Get the most recent time
                    latest = recent_times[0]  # Already sorted newest first
                    if isinstance(latest, str):
                        return datetime.fromisoformat(latest.replace("Z", "+00:00"))
                    elif isinstance(latest, datetime):
                        return latest
            return None
        except Exception as e:
            logger.error(f"Error getting last post time: {e}")
            return None
    
    async def _get_user_analytics(self, platform: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Fetch cached user analytics for a platform.
        
        Returns:
            Tuple of (analytics_data, is_stale)
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
    ) -> Tuple[datetime, str, str, Dict[str, Any]]:
        """
        Calculate optimal time from user's historical engagement data.
        
        KEY LOGIC:
        1. Get TODAY's day name (e.g., Tuesday)
        2. Filter user's historical data to find best hours FOR TODAY
        3. If user has engagement data for today, use the best hour for today
        4. Only if no data for today, fall back to tomorrow
        5. LinkedIn is exception - skip weekends
        
        This ensures we always try to schedule for TODAY first based on 
        the user's engagement patterns for this specific day of the week.
        """
        now = datetime.now(self.tz)
        now_utc = datetime.now(timezone.utc)
        current_day = now.strftime("%A")
        
        time_slots = analytics.get("best_time_slots", [])
        platform_config = PLATFORM_PEAK_HOURS.get(platform, DEFAULT_PLATFORM_CONFIG)
        
        # Check if LinkedIn on weekend - skip to Monday
        restrict_days = platform_config.get('restrict_days', False)
        avoid_days = platform_config.get('avoid_days', [])
        
        if restrict_days and current_day in avoid_days:
            # LinkedIn on weekend - use research data to find next weekday
            return self._get_time_from_research_data(platform, avoid_times)
        
        # Get recent post times to avoid
        recent_post_times = analytics.get("recent_post_times", [])
        for t in recent_post_times:
            try:
                parsed_time = datetime.fromisoformat(t.replace("Z", "+00:00")) if isinstance(t, str) else t
                if parsed_time not in avoid_times:
                    avoid_times.append(parsed_time)
            except Exception:
                continue
        
        # ============================================================
        # STEP 1: Filter slots for TODAY only
        # ============================================================
        today_slots = []
        other_day_slots = []
        
        for slot in time_slots:
            day = slot.get("day", "")
            if day == current_day:
                today_slots.append(slot)
            else:
                other_day_slots.append(slot)
        
        # Weight today's slots by engagement score and recency
        weighted_today_slots = []
        for slot in today_slots:
            recency_weight = slot.get("recency_weight", 1.0)
            engagement_score = slot.get("engagement_score", 0)
            
            # Calculate weighted score
            weighted_score = engagement_score * recency_weight
            
            weighted_today_slots.append({
                **slot,
                "weighted_score": weighted_score
            })
        
        # Sort by weighted score (highest first)
        weighted_today_slots.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # ============================================================
        # STEP 2: Try to schedule for TODAY using user's data
        # ============================================================
        for slot in weighted_today_slots:
            hour = slot.get("hour", 12)
            minute = self._get_optimal_minute(platform, hour)
            
            proposed_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Skip if time is in the past or too soon
            if proposed_time <= now + timedelta(minutes=30):
                continue
            
            proposed_time_utc = proposed_time.astimezone(timezone.utc)
            
            if not self._time_conflicts(proposed_time_utc, avoid_times):
                formatted_time = self._format_time_with_timezone(proposed_time, hour, minute)
                
                reason = (
                    f"Scheduled for today ({current_day}) at {formatted_time} based on your "
                    f"{current_day} engagement history on {PLATFORM_NAME_MAP.get(platform, platform.capitalize())}. "
                    f"This time slot achieves {slot.get('weighted_score', 0):.1f}x engagement on {current_day}s."
                )
                
                metadata = {
                    "timezone": self.user_timezone,
                    "local_time": proposed_time.isoformat(),
                    "utc_time": proposed_time_utc.isoformat(),
                    "engagement_score": slot.get("weighted_score", 0),
                    "day_of_week": current_day,
                    "hour": hour,
                    "minute": minute,
                    "scheduled_for_today": True,
                    "data_basis": f"user_{current_day}_history"
                }
                
                return proposed_time_utc, reason, "user_data", metadata
        
        # ============================================================
        # STEP 3: If no user data for today's hours available,
        #         use research data for TODAY (not jump to another day)
        # ============================================================
        logger.info(f"No user data slots available for {current_day}, using research data for today")
        
        # Get today's hours from research data
        day_wise_hours = platform_config.get('day_wise_hours', {})
        today_hours = day_wise_hours.get(current_day, [])
        
        if not today_hours:
            is_weekend = now.weekday() >= 5
            if is_weekend:
                today_hours = platform_config.get('weekend_hours', platform_config.get('best_hours', [12]))
            else:
                today_hours = platform_config.get('weekday_hours', platform_config.get('best_hours', [12]))
        
        best_hours = set(platform_config.get('best_hours', []))
        sorted_hours = sorted(today_hours, key=lambda h: (h not in best_hours, h))
        day_multiplier = platform_config.get('engagement_multiplier', {}).get(current_day, 1.0)
        
        for hour in sorted_hours:
            minute = self._get_optimal_minute(platform, hour)
            
            proposed_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if proposed_time <= now + timedelta(minutes=30):
                continue
            
            proposed_time_utc = proposed_time.astimezone(timezone.utc)
            
            if not self._time_conflicts(proposed_time_utc, avoid_times):
                formatted_time = self._format_time_with_timezone(proposed_time, hour, minute)
                
                reason = (
                    f"Scheduled for today ({current_day}) at {formatted_time} ({self.user_timezone}) - "
                    f"Using research data for {current_day}. {platform_config.get('description', '')}. "
                    f"Today's engagement: {day_multiplier:.2f}x average."
                )
                
                metadata = {
                    "timezone": self.user_timezone,
                    "local_time": proposed_time.isoformat(),
                    "utc_time": proposed_time_utc.isoformat(),
                    "engagement_multiplier": day_multiplier,
                    "day_of_week": current_day,
                    "hour": hour,
                    "minute": minute,
                    "is_peak_hour": hour in best_hours,
                    "scheduled_for_today": True,
                    "data_basis": "research_for_today"
                }
                
                return proposed_time_utc, reason, "user_data", metadata
        
        # ============================================================
        # STEP 4: If no slots available today at all, try tomorrow
        # ============================================================
        return self._schedule_for_tomorrow(platform, now, avoid_times, platform_config)
    
    def _get_time_from_research_data(
        self,
        platform: str,
        avoid_times: List[datetime]
    ) -> Tuple[datetime, str, str, Dict[str, Any]]:
        """
        Get optimal time from AI research-based data.
        
        LOGIC:
        1. Get TODAY's day name (e.g., Tuesday)
        2. Find the best hours FOR TODAY based on day_wise_hours
        3. Only LinkedIn has day restrictions (skip weekends)
        4. For other platforms, always try to schedule for TODAY first
        5. Use engagement multiplier to adjust scoring but not to skip days
        """
        now = datetime.now(self.tz)
        now_utc = datetime.now(timezone.utc)
        current_day = now.strftime("%A")
        is_weekend = now.weekday() >= 5
        
        platform_config = PLATFORM_PEAK_HOURS.get(platform, DEFAULT_PLATFORM_CONFIG)
        
        # Check if this platform has day restrictions (only LinkedIn)
        restrict_days = platform_config.get('restrict_days', False)
        avoid_days = platform_config.get('avoid_days', [])
        
        # For LinkedIn on weekends, schedule for Monday
        if restrict_days and current_day in avoid_days:
            return self._schedule_for_next_available_day(platform, now, avoid_times, platform_config)
        
        # Get day-wise hours for TODAY (this is the key improvement)
        day_wise_hours = platform_config.get('day_wise_hours', {})
        today_hours = day_wise_hours.get(current_day, [])
        
        # If no day-wise hours defined, fall back to weekend/weekday hours
        if not today_hours:
            if is_weekend:
                today_hours = platform_config.get('weekend_hours', platform_config.get('best_hours', [12]))
            else:
                today_hours = platform_config.get('weekday_hours', platform_config.get('best_hours', [12]))
        
        # Sort hours by best engagement (best_hours first)
        best_hours = set(platform_config.get('best_hours', []))
        sorted_hours = sorted(today_hours, key=lambda h: (h not in best_hours, h))
        
        # Get today's engagement multiplier
        day_multiplier = platform_config.get('engagement_multiplier', {}).get(current_day, 1.0)
        
        # Try each hour for TODAY
        for hour in sorted_hours:
            minute = self._get_optimal_minute(platform, hour)
            
            proposed_time = now.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0
            )
            
            # Skip if time is in the past or too soon (need at least 30 mins)
            if proposed_time <= now + timedelta(minutes=30):
                continue
            
            proposed_time_utc = proposed_time.astimezone(timezone.utc)
            
            if not self._time_conflicts(proposed_time_utc, avoid_times):
                formatted_time = self._format_time_with_timezone(proposed_time, hour, minute)
                
                reason = (
                    f"Scheduled for today ({current_day}) at {formatted_time} ({self.user_timezone}) - "
                    f"{platform_config.get('description', 'optimal engagement time')}. "
                    f"Today's engagement: {day_multiplier:.2f}x average. "
                    f"Post more to get personalized recommendations!"
                )
                
                metadata = {
                    "timezone": self.user_timezone,
                    "local_time": proposed_time.isoformat(),
                    "utc_time": proposed_time_utc.isoformat(),
                    "engagement_multiplier": day_multiplier,
                    "day_of_week": current_day,
                    "hour": hour,
                    "minute": minute,
                    "is_peak_hour": hour in best_hours,
                    "scheduled_for_today": True
                }
                
                return proposed_time_utc, reason, "research_data", metadata
        
        # If no slots available today, try tomorrow
        return self._schedule_for_tomorrow(platform, now, avoid_times, platform_config)
    
    def _schedule_for_tomorrow(
        self,
        platform: str,
        now: datetime,
        avoid_times: List[datetime],
        platform_config: Dict
    ) -> Tuple[datetime, str, str, Dict[str, Any]]:
        """Schedule for tomorrow when no slots available today."""
        tomorrow = now + timedelta(days=1)
        tomorrow_day = tomorrow.strftime("%A")
        
        # For LinkedIn, skip weekends
        restrict_days = platform_config.get('restrict_days', False)
        avoid_days = platform_config.get('avoid_days', [])
        
        if restrict_days and tomorrow_day in avoid_days:
            return self._schedule_for_next_available_day(platform, now, avoid_times, platform_config)
        
        # Get tomorrow's hours
        day_wise_hours = platform_config.get('day_wise_hours', {})
        tomorrow_hours = day_wise_hours.get(tomorrow_day, [])
        
        if not tomorrow_hours:
            is_weekend = tomorrow.weekday() >= 5
            if is_weekend:
                tomorrow_hours = platform_config.get('weekend_hours', platform_config.get('best_hours', [12]))
            else:
                tomorrow_hours = platform_config.get('weekday_hours', platform_config.get('best_hours', [12]))
        
        best_hours = set(platform_config.get('best_hours', []))
        sorted_hours = sorted(tomorrow_hours, key=lambda h: (h not in best_hours, h))
        
        day_multiplier = platform_config.get('engagement_multiplier', {}).get(tomorrow_day, 1.0)
        
        for hour in sorted_hours:
            minute = self._get_optimal_minute(platform, hour)
            
            proposed_time = tomorrow.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0
            )
            
            proposed_time_utc = proposed_time.astimezone(timezone.utc)
            
            if not self._time_conflicts(proposed_time_utc, avoid_times):
                formatted_time = self._format_time_with_timezone(proposed_time, hour, minute)
                
                reason = (
                    f"Scheduled for tomorrow ({tomorrow_day}) at {formatted_time} ({self.user_timezone}) - "
                    f"No optimal slots available today. {platform_config.get('description', '')}. "
                    f"Tomorrow's engagement: {day_multiplier:.2f}x average."
                )
                
                metadata = {
                    "timezone": self.user_timezone,
                    "local_time": proposed_time.isoformat(),
                    "utc_time": proposed_time_utc.isoformat(),
                    "engagement_multiplier": day_multiplier,
                    "day_of_week": tomorrow_day,
                    "hour": hour,
                    "minute": minute,
                    "is_peak_hour": hour in best_hours,
                    "scheduled_for_today": False
                }
                
                return proposed_time_utc, reason, "research_data", metadata
        
        # Ultimate fallback
        return self._get_fallback_time(platform, now)
    
    def _schedule_for_next_available_day(
        self,
        platform: str,
        now: datetime,
        avoid_times: List[datetime],
        platform_config: Dict
    ) -> Tuple[datetime, str, str, Dict[str, Any]]:
        """
        Schedule for the next available day (used for LinkedIn on weekends).
        """
        avoid_days = platform_config.get('avoid_days', [])
        best_hours = set(platform_config.get('best_hours', []))
        
        # Find the next non-avoided day
        for days_ahead in range(1, 8):
            target_date = now + timedelta(days=days_ahead)
            target_day = target_date.strftime("%A")
            
            if target_day not in avoid_days:
                day_wise_hours = platform_config.get('day_wise_hours', {})
                target_hours = day_wise_hours.get(target_day, platform_config.get('weekday_hours', []))
                
                if not target_hours:
                    target_hours = platform_config.get('weekday_hours', platform_config.get('best_hours', [12]))
                
                sorted_hours = sorted(target_hours, key=lambda h: (h not in best_hours, h))
                day_multiplier = platform_config.get('engagement_multiplier', {}).get(target_day, 1.0)
                
                for hour in sorted_hours:
                    minute = self._get_optimal_minute(platform, hour)
                    
                    proposed_time = target_date.replace(
                        hour=hour,
                        minute=minute,
                        second=0,
                        microsecond=0
                    )
                    
                    proposed_time_utc = proposed_time.astimezone(timezone.utc)
                    
                    if not self._time_conflicts(proposed_time_utc, avoid_times):
                        formatted_time = self._format_time_with_timezone(proposed_time, hour, minute)
                        
                        reason = (
                            f"Scheduled for {target_day} at {formatted_time} ({self.user_timezone}) - "
                            f"LinkedIn is a professional network, weekends have very low engagement. "
                            f"{target_day}'s engagement: {day_multiplier:.2f}x average."
                        )
                        
                        metadata = {
                            "timezone": self.user_timezone,
                            "local_time": proposed_time.isoformat(),
                            "utc_time": proposed_time_utc.isoformat(),
                            "engagement_multiplier": day_multiplier,
                            "day_of_week": target_day,
                            "hour": hour,
                            "minute": minute,
                            "is_peak_hour": hour in best_hours,
                            "scheduled_for_today": False,
                            "weekend_avoided": True
                        }
                        
                        return proposed_time_utc, reason, "research_data", metadata
        
        # Ultimate fallback
        return self._get_fallback_time(platform, now)
    
    def _get_fallback_time(
        self,
        platform: str,
        now: datetime
    ) -> Tuple[datetime, str, str, Dict[str, Any]]:
        """Get a fallback time when no optimal slots are available."""
        fallback_minute = PLATFORM_MINUTE_OFFSET.get(platform, 0)
        fallback_time = (now + timedelta(hours=1)).replace(minute=fallback_minute, second=0, microsecond=0)
        fallback_time_utc = fallback_time.astimezone(timezone.utc)
        
        metadata = {
            "timezone": self.user_timezone,
            "local_time": fallback_time.isoformat(),
            "utc_time": fallback_time_utc.isoformat(),
            "fallback": True
        }
        
        return fallback_time_utc, f"Scheduled for {self._format_time_with_timezone(fallback_time, fallback_time.hour, fallback_minute)} ({self.user_timezone}) - Safe scheduling time.", "fallback", metadata
    
    def _get_days_to_try(self, now: datetime, platform_config: Dict) -> List[Tuple[int, str]]:
        """
        Get a list of (days_offset, day_name) tuples.
        For most platforms, prioritize TODAY. Only LinkedIn skips weekends.
        """
        days_to_try = []
        restrict_days = platform_config.get('restrict_days', False)
        avoid_days = platform_config.get('avoid_days', [])
        
        # Always try today first (unless restricted)
        today_name = now.strftime("%A")
        if not (restrict_days and today_name in avoid_days):
            days_to_try.append((0, today_name))
        
        # Then tomorrow
        for offset in range(1, 8):
            future_date = now + timedelta(days=offset)
            day_name = future_date.strftime("%A")
            if not (restrict_days and day_name in avoid_days):
                days_to_try.append((offset, day_name))
        
        return days_to_try
    
    def _get_optimal_minute(self, platform: str, hour: int) -> int:
        """
        Calculate the optimal minute for posting based on platform and hour.
        
        Uses platform-specific minute patterns and adds slight randomization
        to avoid posting at exactly the same time as competitors.
        """
        platform_config = PLATFORM_PEAK_HOURS.get(platform, DEFAULT_PLATFORM_CONFIG)
        
        # Use prime minutes (slightly offset from common times)
        prime_minutes = platform_config.get('prime_minutes', [5, 20, 35, 50])
        
        # Add platform-specific offset to stagger multi-platform posts
        platform_offset = PLATFORM_MINUTE_OFFSET.get(platform, 0) % 15
        
        # Select a minute based on hour (different hours get different minutes)
        minute_index = hour % len(prime_minutes)
        base_minute = prime_minutes[minute_index]
        
        # Add small random variation (±3 minutes) to avoid exact scheduling collisions
        random_offset = random.randint(-3, 3)
        final_minute = (base_minute + platform_offset + random_offset) % 60
        
        # Ensure minute is within valid range
        return max(0, min(59, final_minute))
    
    def _time_conflicts(self, proposed: datetime, avoid_times: List[datetime]) -> bool:
        """Check if proposed time conflicts with any avoid times."""
        # Ensure proposed is timezone-aware
        if proposed.tzinfo is None:
            proposed = proposed.replace(tzinfo=timezone.utc)
        
        for avoid_time in avoid_times:
            if avoid_time is None:
                continue
            
            # Ensure avoid_time is timezone-aware
            if avoid_time.tzinfo is None:
                avoid_time = avoid_time.replace(tzinfo=timezone.utc)
            
            time_diff = abs((proposed - avoid_time).total_seconds())
            if time_diff < CONFLICT_WINDOW_MINUTES * 60:  # Convert to seconds
                return True
        
        return False
    
    def _format_time_with_timezone(self, dt: datetime, hour: int, minute: int) -> str:
        """Format time with hour, minute, and AM/PM."""
        if hour == 0:
            hour_str = "12"
            period = "AM"
        elif hour < 12:
            hour_str = str(hour)
            period = "AM"
        elif hour == 12:
            hour_str = "12"
            period = "PM"
        else:
            hour_str = str(hour - 12)
            period = "PM"
        
        return f"{hour_str}:{minute:02d} {period}"
    
    def _format_hour(self, hour: int) -> str:
        """Format hour as 12-hour time (legacy compatibility)."""
        return self._format_time_with_timezone(None, hour, 0)


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
    - Now includes minute-level timing data for precise scheduling
    
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
        
        # STEP 3: Process posts to calculate best time slots with minute-level precision
        time_slot_scores = {}  # {(day_of_week, hour, minute_bucket): [engagement_scores]}
        slot_recency = {}  # {(day_of_week, hour, minute_bucket): [days_ago]}
        recent_post_times = []
        
        # Minute buckets: 0-14, 15-29, 30-44, 45-59
        def get_minute_bucket(minute: int) -> int:
            return (minute // 15) * 15
        
        for post in recent_posts:
            posted_at = post["_parsed_time"]
            recent_post_times.append(posted_at.isoformat())
            
            # Calculate engagement score with weighted metrics
            likes = post.get("likes", 0) or post.get("like_count", 0) or 0
            comments = post.get("comments", 0) or post.get("comments_count", 0) or 0
            shares = post.get("shares", 0) or post.get("share_count", 0) or 0
            views = post.get("views", 0) or post.get("play_count", 0) or 0
            saves = post.get("saves", 0) or post.get("saved", 0) or 0
            
            # Weighted engagement score
            # Shares/Saves are most valuable (shows intent to return)
            # Comments are highly valuable (shows active engagement)
            # Likes are standard engagement
            # Views are weakest signal
            engagement_score = (
                likes * 1.0 +
                comments * 3.0 +
                shares * 4.0 +
                saves * 5.0 +
                views * 0.01
            )
            
            day_of_week = posted_at.weekday()
            hour = posted_at.hour
            minute = posted_at.minute
            minute_bucket = get_minute_bucket(minute)
            
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week]
            
            # Store both hour-level and minute-level data
            key_hour = (day_name, hour)
            key_minute = (day_name, hour, minute_bucket)
            
            # Hour-level tracking
            if key_hour not in time_slot_scores:
                time_slot_scores[key_hour] = []
                slot_recency[key_hour] = []
            time_slot_scores[key_hour].append(engagement_score)
            
            # Minute-level tracking
            if key_minute not in time_slot_scores:
                time_slot_scores[key_minute] = []
                slot_recency[key_minute] = []
            time_slot_scores[key_minute].append(engagement_score)
            
            # Recency: how many days ago was this post?
            days_ago = (datetime.now(timezone.utc) - posted_at).days
            slot_recency[key_hour].append(days_ago)
            slot_recency[key_minute].append(days_ago)
        
        # Calculate average scores per time slot, with recency weighting
        best_time_slots = []
        
        for key, scores in time_slot_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Recency weighting: more recent posts weigh more (exponential decay)
            recency_weights = [1 / (1 + d * 0.1) for d in slot_recency.get(key, [0])]
            recency_weight = sum(recency_weights) / len(recency_weights) if recency_weights else 1.0
            
            if len(key) == 2:  # Hour-level slot (day, hour)
                day, hour = key
                best_time_slots.append({
                    "day": day,
                    "hour": hour,
                    "minute": None,  # Hour-level slot
                    "engagement_score": avg_score,
                    "post_count": len(scores),
                    "recency_weight": recency_weight,
                    "slot_type": "hour"
                })
            elif len(key) == 3:  # Minute-level slot (day, hour, minute_bucket)
                day, hour, minute_bucket = key
                best_time_slots.append({
                    "day": day,
                    "hour": hour,
                    "minute": minute_bucket,
                    "engagement_score": avg_score,
                    "post_count": len(scores),
                    "recency_weight": recency_weight,
                    "slot_type": "minute"
                })
        
        # Sort by weighted score (engagement * recency)
        best_time_slots.sort(key=lambda x: x["engagement_score"] * x["recency_weight"], reverse=True)
        
        # Separate hour-level and minute-level slots for storage
        hour_slots = [s for s in best_time_slots if s["slot_type"] == "hour"][:10]
        minute_slots = [s for s in best_time_slots if s["slot_type"] == "minute"][:15]
        
        # STEP 4: INSERT new analytics document
        now = datetime.now(timezone.utc)
        analytics_doc = {
            "userId": user_id,
            "platform": platform.lower(),
            "post_count": len(recent_posts),
            "total_posts_received": len(posts_data),
            "best_time_slots": hour_slots,  # Keep backward compatibility
            "minute_slots": minute_slots,  # New minute-level data
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
            "best_time_slots": hour_slots[:5],
            "minute_slots": minute_slots[:5],
            "message": f"Stored {len(recent_posts)} most recent posts (old data replaced)"
        }
        
    except Exception as e:
        logger.error(f"Error saving user analytics: {e}")
        return {"success": False, "error": str(e)}


async def get_optimal_times_for_platforms(
    user_id: str,
    platforms: List[str],
    user_timezone: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get optimal posting times for multiple platforms.
    
    Features:
    - Timezone-aware scheduling
    - Avoids conflicts between platforms
    - Returns detailed metadata for each platform
    - Ensures no two platforms post at the same time
    
    Args:
        user_id: The user's ID
        platforms: List of platform names (lowercase)
        user_timezone: User's timezone (e.g., 'America/New_York', 'Asia/Kolkata')
    
    Returns:
        Dict mapping platform to time info with detailed metadata
    """
    # Try to get user's timezone from database if not provided
    if not user_timezone:
        try:
            users_collection = db["users"]
            user = await users_collection.find_one({"_id": user_id})
            if user:
                user_timezone = user.get("timezone") or user.get("settings", {}).get("timezone")
        except Exception as e:
            logger.warning(f"Could not fetch user timezone: {e}")
    
    user_timezone = user_timezone or DEFAULT_TIMEZONE
    
    service = TimeSlotService(user_id, user_timezone)
    results = {}
    scheduled_times = []  # Track to avoid conflicts between platforms
    
    # Sort platforms by priority (platforms with stricter timing requirements first)
    platform_priority = {
        'linkedin': 1,  # Most time-sensitive (business hours)
        'twitter': 2,
        'x': 2,
        'facebook': 3,
        'threads': 4,
        'instagram': 5,
        'tiktok': 6,
        'youtube': 7,  # Least time-sensitive (content lives longer)
        'pinterest': 8,
    }
    
    sorted_platforms = sorted(platforms, key=lambda p: platform_priority.get(p.lower(), 99))
    
    for platform in sorted_platforms:
        try:
            scheduled_at, reason, data_source, metadata = await service.get_optimal_time_for_platform(
                platform,
                avoid_times=scheduled_times.copy()
            )
            
            scheduled_times.append(scheduled_at)
            
            results[platform] = {
                "scheduledAt": scheduled_at,
                "scheduledAtISO": scheduled_at.isoformat(),
                "localTime": metadata.get("local_time"),
                "timezone": user_timezone,
                "reason": reason,
                "dataSource": data_source,
                "dayOfWeek": metadata.get("day_of_week"),
                "hour": metadata.get("hour"),
                "minute": metadata.get("minute"),
                "engagementScore": metadata.get("engagement_score") or metadata.get("engagement_multiplier"),
                "isPeakHour": metadata.get("is_peak_hour", False),
                "isBestDay": metadata.get("is_best_day", False),
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error getting optimal time for {platform}: {e}")
            # Provide a fallback time
            fallback_time = datetime.now(timezone.utc) + timedelta(hours=1)
            results[platform] = {
                "scheduledAt": fallback_time,
                "scheduledAtISO": fallback_time.isoformat(),
                "timezone": user_timezone,
                "reason": f"Fallback time due to error: {str(e)}",
                "dataSource": "fallback",
                "error": str(e)
            }
    
    return results


async def get_single_optimal_time(
    user_id: str,
    platform: str,
    user_timezone: str = None,
    last_post_time: datetime = None,
    avoid_times: List[datetime] = None
) -> Dict[str, Any]:
    """
    Get optimal posting time for a single platform.
    
    This is a convenience function for getting one platform's optimal time.
    
    Args:
        user_id: The user's ID
        platform: Platform name (e.g., 'instagram')
        user_timezone: User's timezone
        last_post_time: Most recent post time to avoid
        avoid_times: Additional times to avoid
    
    Returns:
        Dict with scheduling info and metadata
    """
    user_timezone = user_timezone or DEFAULT_TIMEZONE
    service = TimeSlotService(user_id, user_timezone)
    
    scheduled_at, reason, data_source, metadata = await service.get_optimal_time_for_platform(
        platform,
        avoid_times=avoid_times,
        last_post_time=last_post_time
    )
    
    return {
        "scheduledAt": scheduled_at,
        "scheduledAtISO": scheduled_at.isoformat(),
        "localTime": metadata.get("local_time"),
        "timezone": user_timezone,
        "reason": reason,
        "dataSource": data_source,
        "dayOfWeek": metadata.get("day_of_week"),
        "hour": metadata.get("hour"),
        "minute": metadata.get("minute"),
        "engagementScore": metadata.get("engagement_score") or metadata.get("engagement_multiplier"),
        "isPeakHour": metadata.get("is_peak_hour", False),
        "isBestDay": metadata.get("is_best_day", False),
        "metadata": metadata
    }


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
