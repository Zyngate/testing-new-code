# services/recommendation_plan_integration.py
"""
Recommendation Engine ↔ Plan My Week Integration Service

This module bridges the recommendation engine's data (optimal posting times,
platform performance, content analytics) with the Plan My Week feature,
ensuring weekly plans generated from goals align with recommendation insights.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Set, Optional
from config import logger

# Import platform peak hours from time_slot_service (recommendation engine data)
from services.time_slot_service import PLATFORM_PEAK_HOURS

# Supported social media platforms for goal detection
SOCIAL_MEDIA_PLATFORMS = [
    'instagram', 'linkedin', 'facebook', 'threads', 
    'youtube', 'tiktok', 'twitter', 'x', 'pinterest'
]

# Keywords that indicate a social media related goal
SOCIAL_MEDIA_KEYWORDS = [
    'social media', 'content creator', 'influencer', 'followers', 'engagement',
    'posting', 'reels', 'stories', 'posts per week', 'content strategy',
    'grow audience', 'brand awareness', 'content calendar', 'hashtag',
    'viral', 'subscribers', 'views', 'likes', 'comments', 'shares',
    'content plan', 'posting schedule', 'social presence', 'online presence',
    'digital marketing', 'social marketing', 'content marketing'
]

# Platform name aliases
PLATFORM_ALIASES = {
    'x': 'twitter',
    'twitter': 'twitter',
    'ig': 'instagram',
    'fb': 'facebook',
    'yt': 'youtube',
    'tt': 'tiktok',
    'li': 'linkedin',
}


def detect_social_media_platforms(text: str) -> Set[str]:
    """
    Detect social media platforms mentioned in the given text.
    Returns a set of normalized platform names.
    """
    text_lower = text.lower()
    detected = set()
    
    for platform in SOCIAL_MEDIA_PLATFORMS:
        if platform in text_lower:
            detected.add(PLATFORM_ALIASES.get(platform, platform))
    
    # Check aliases
    for alias, platform in PLATFORM_ALIASES.items():
        if alias in text_lower.split():  # Only match whole words for short aliases
            detected.add(platform)
    
    return detected


def is_social_media_goal(goal_text: str) -> bool:
    """Check if a goal text is related to social media."""
    text_lower = goal_text.lower()
    
    # Check for platform names
    for platform in SOCIAL_MEDIA_PLATFORMS:
        if platform in text_lower:
            return True
    
    # Check for social media keywords
    for keyword in SOCIAL_MEDIA_KEYWORDS:
        if keyword in text_lower:
            return True
    
    return False


def build_platform_schedule_context(platforms: Set[str]) -> str:
    """
    Build a formatted context string with platform posting schedules
    from the recommendation engine's research data.
    """
    if not platforms:
        return ""
    
    context_parts = []
    context_parts.append("\n📅 PLATFORM OPTIMAL POSTING SCHEDULE (from recommendation engine):")
    
    for platform in platforms:
        config = PLATFORM_PEAK_HOURS.get(platform.lower())
        if config:
            best_hours = config.get('best_hours', [])
            best_days = config.get('best_days', [])
            worst_days = config.get('worst_days', [])
            posts_per_week = config.get('posts_per_week', 3)
            description = config.get('description', '')
            peak_multiplier = config.get('engagement_multiplier', {}).get('peak', 1.0)
            
            best_hours_str = ', '.join([
                f"{(h % 12) or 12}:00 {'AM' if h < 12 else 'PM'}" for h in best_hours
            ])
            
            context_parts.append(
                f"  🔹 {platform.upper()}:\n"
                f"     - Recommended posts/week: {posts_per_week}\n"
                f"     - Best posting times: {best_hours_str}\n"
                f"     - Best times description: {description}\n"
                f"     - Best days: {', '.join(best_days) if best_days else 'Any weekday'}\n"
                f"     - Worst days (avoid): {', '.join(worst_days) if worst_days else 'None'}\n"
                f"     - Peak engagement multiplier: {peak_multiplier}x"
            )
    
    return "\n".join(context_parts)


async def build_user_analytics_context(user_id: str, platforms: Set[str], analytics_collection) -> str:
    """
    Fetch user-specific analytics from the recommendation engine and build context string.
    Uses the user_post_analytics collection populated by the recommendation engine.
    """
    if not platforms or not analytics_collection:
        return ""
    
    context_parts = []
    has_data = False
    
    for platform in platforms:
        try:
            analytics = await analytics_collection.find_one({
                "user_id": user_id,
                "platform": platform.lower()
            })
            
            if analytics and analytics.get("post_count", 0) >= 15:
                has_data = True
                post_count = analytics.get('post_count', 0)
                best_slots = analytics.get("best_time_slots", [])
                top_categories = analytics.get("top_categories", [])
                avg_engagement = analytics.get("avg_engagement_rate", 0)
                
                # Format best time slots
                top_slots_str = "Not enough data"
                if best_slots:
                    slot_descriptions = []
                    for slot in best_slots[:3]:
                        hour = slot.get("hour", 0)
                        day = slot.get("day", "Unknown")
                        score = slot.get("engagement_score", 0)
                        suffix = "AM" if hour < 12 else "PM"
                        h12 = hour % 12 or 12
                        slot_descriptions.append(
                            f"{day} at {h12}:00 {suffix} (engagement: {score:.1f})"
                        )
                    top_slots_str = "; ".join(slot_descriptions)
                
                context_parts.append(
                    f"  🎯 {platform.upper()} (Based on YOUR {post_count} analyzed posts):\n"
                    f"     - Your best performing time slots: {top_slots_str}\n"
                    f"     - Your top content categories: {', '.join(top_categories) if top_categories else 'N/A'}\n"
                    f"     - Your avg engagement rate: {avg_engagement:.2f}%\n"
                    f"     - Data source: YOUR post analytics (prioritize over general data)"
                )
        except Exception as e:
            logger.warning(f"Could not fetch analytics for {platform}: {e}")
    
    if has_data:
        return "\n🎯 USER-SPECIFIC ANALYTICS (from recommendation engine):\n" + "\n".join(context_parts)
    return ""


async def get_recommendation_context_for_plan(
    user_id: str, 
    goal_text: str, 
    goals_data: Optional[List[Dict[str, Any]]] = None,
    analytics_collection=None
) -> str:
    """
    Main entry point: Builds full recommendation engine context for plan generation.
    
    Can be used with:
    - A single goal_text string (plan_routes.py path)
    - A list of goals_data dicts (goal_routes.py path)
    
    Returns a formatted context string to inject into the AI plan generation prompt.
    """
    # 1. Detect platforms from goal text
    detected_platforms = detect_social_media_platforms(goal_text)
    
    # Also check goals_data if provided
    if goals_data:
        for goal in goals_data:
            combined_text = (
                goal.get('title', '') + ' ' + 
                goal.get('description', '') + ' ' +
                ' '.join([t.get('title', '') for t in goal.get('tasks', [])])
            )
            detected_platforms.update(detect_social_media_platforms(combined_text))
    
    # Check for generic social media keywords
    all_text = goal_text
    if goals_data:
        all_text += ' ' + ' '.join([
            g.get('title', '') + ' ' + g.get('description', '') 
            for g in goals_data
        ])
    
    has_social_keywords = any(kw in all_text.lower() for kw in SOCIAL_MEDIA_KEYWORDS)
    
    # If no platforms detected but social media keywords present, use common platforms
    if has_social_keywords and not detected_platforms:
        detected_platforms = {'instagram', 'linkedin', 'facebook'}
    
    # If nothing social media related, return empty
    if not detected_platforms:
        return ""
    
    context_parts = []
    context_parts.append("\n\n📊 RECOMMENDATION ENGINE DATA (Use this to align the plan):")
    context_parts.append("=" * 60)
    
    # 2. Add user-specific analytics if available
    if analytics_collection and user_id:
        user_context = await build_user_analytics_context(
            user_id, detected_platforms, analytics_collection
        )
        if user_context:
            context_parts.append(user_context)
    
    # 3. Add platform peak hours from recommendation engine
    schedule_context = build_platform_schedule_context(detected_platforms)
    if schedule_context:
        context_parts.append(schedule_context)
    
    # 4. Add alignment instructions
    platforms_list = ', '.join([p.upper() for p in detected_platforms])
    context_parts.append(f"\n⚡ PLAN-RECOMMENDATION ALIGNMENT RULES:")
    context_parts.append(f"  Detected social media platforms: {platforms_list}")
    context_parts.append(f"  1. Schedule social media tasks at the OPTIMAL POSTING TIMES listed above.")
    context_parts.append(f"  2. Match the recommended POSTING FREQUENCY (posts_per_week) for content tasks.")
    context_parts.append(f"  3. Prioritize tasks on BEST DAYS for each platform.")
    context_parts.append(f"  4. Avoid scheduling key social media tasks on WORST DAYS.")
    context_parts.append(f"  5. Include the recommended posting time in task descriptions.")
    context_parts.append(f"  6. Prepare content BEFORE the optimal posting window.")
    context_parts.append(f"  7. If user analytics exist, PRIORITIZE user data over general research.")
    context_parts.append(f"  8. Each social media sub_task should mention the specific recommended time.")
    context_parts.append("=" * 60)
    
    return "\n".join(context_parts)


def get_recommendation_context_sync(goal_text: str) -> str:
    """
    Synchronous version: Builds recommendation context from goal text only.
    Uses platform peak hours data (no user analytics since those require async DB calls).
    Used by plan_service.py synchronous planner.
    """
    detected_platforms = detect_social_media_platforms(goal_text)
    
    # Check for generic social media keywords
    has_social_keywords = any(kw in goal_text.lower() for kw in SOCIAL_MEDIA_KEYWORDS)
    
    if has_social_keywords and not detected_platforms:
        detected_platforms = {'instagram', 'linkedin', 'facebook'}
    
    if not detected_platforms:
        return ""
    
    context_parts = []
    context_parts.append("\n\n📊 RECOMMENDATION ENGINE DATA (Use this to align the plan):")
    context_parts.append("=" * 60)
    
    # Add platform schedule context
    schedule_context = build_platform_schedule_context(detected_platforms)
    if schedule_context:
        context_parts.append(schedule_context)
    
    # Add alignment instructions
    platforms_list = ', '.join([p.upper() for p in detected_platforms])
    context_parts.append(f"\n⚡ PLAN-RECOMMENDATION ALIGNMENT RULES:")
    context_parts.append(f"  Detected platforms: {platforms_list}")
    context_parts.append(f"  1. Schedule social media tasks at OPTIMAL POSTING TIMES listed above.")
    context_parts.append(f"  2. Match recommended POSTING FREQUENCY for content tasks.")
    context_parts.append(f"  3. Prioritize BEST DAYS, avoid WORST DAYS for each platform.")
    context_parts.append(f"  4. Include recommended posting time in task descriptions and subtasks.")
    context_parts.append(f"  5. Schedule content creation BEFORE optimal posting windows.")
    context_parts.append("=" * 60)
    
    return "\n".join(context_parts)
