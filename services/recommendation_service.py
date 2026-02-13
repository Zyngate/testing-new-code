# services/recommendation_service.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import pytz
from scipy import stats
import math
import re

load_dotenv()  # Load .env file

def clean_nan_inf(obj):
    """Recursively clean NaN/Inf for JSON"""
    if isinstance(obj, dict):
        return {k: clean_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_inf(item) for item in obj]
    elif isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    elif pd.isna(obj):
        return 0.0
    return obj


# Utility functions
def extract_platform_from_url(url: str) -> str:
    """Extract the social media platform from a URL"""
    url_lower = url.lower()
    if "instagram.com" in url_lower: return "instagram"
    elif "youtube.com" in url_lower or "youtu.be" in url_lower: return "youtube"
    elif "linkedin.com" in url_lower: return "linkedin"
    elif "threads.net" in url_lower: return "threads"
    elif "tiktok.com" in url_lower: return "tiktok"
    elif "facebook.com" in url_lower or "fb.com" in url_lower: return "facebook"
    else: return "unknown"

def format_hour_12h(hour: int) -> str:
    """Convert 0-23 hour to 12h format like 8:00 am or 5:00 pm."""
    suffix = "am" if hour < 12 else "pm"
    hour12 = hour % 12
    if hour12 == 0: hour12 = 12
    return f"{hour12}:00 {suffix}"

def format_time_range(hour: int) -> str:
    """Convert hour to a time range like '3:00 PM - 5:00 PM' (1 hour before and after)."""
    start_hour = max(0, hour - 1)
    end_hour = min(23, hour + 1)
    
    def to_12h(h):
        suffix = "AM" if h < 12 else "PM"
        h12 = h % 12
        if h12 == 0: h12 = 12
        return f"{h12}:00 {suffix}"
    
    return f"{to_12h(start_hour)} - {to_12h(end_hour)}"

# Platform-specific research data for best posting times and frequencies
PLATFORM_PEAK_HOURS = {
    'instagram': {
        # PRE-PEAK Strategy: Meta's algorithm needs 2-5 hours to distribute content.
        # Post BEFORE peak traffic, not AT peak. NEVER post at 5 PM, 7 PM, 9:30 PM.
        # Proven: 6:37 AM = 144K likes, 10:35 AM = 102K likes
        'peak_hours': [6, 7, 9, 10, 14, 15, 16],  # Pre-peak hours (2-3h before traffic peaks)
        'best_hours': [6, 7, 10, 14],  # Proven best pre-peak times
        'posts_per_week': 14,  # 2 posts daily minimum for growth
        'avoid_hours': [17, 18, 19, 20, 21],  # NEVER post during high traffic
        'description': 'PRE-PEAK: Post 2-3 hours before traffic peaks. Best at 6-7 AM (before morning rush), 9-10 AM (before lunch peak), 2-4 PM (before evening peak). NEVER post at 5-9 PM - Meta needs time to distribute.'
    },
    'twitter': {
        'peak_hours': [8, 9, 10, 12, 13, 17, 18],  # Business hours
        'best_hours': [9, 12, 17],
        'posts_per_week': 21,  # 3 posts daily
        'description': 'morning (8-10 AM), lunch (12-1 PM), and end of workday (5-6 PM)'
    },
    'x': {
        'peak_hours': [8, 9, 10, 12, 13, 17, 18],  # Same as Twitter
        'best_hours': [9, 12, 17],
        'posts_per_week': 21,
        'description': 'morning (8-10 AM), lunch (12-1 PM), and end of workday (5-6 PM)'
    },
    'facebook': {
        'peak_hours': [9, 10, 11, 13, 14, 18, 19],  # Mid-morning and early evening
        'best_hours': [10, 13, 19],
        'posts_per_week': 4,  # 3-4 times per week
        'description': 'mid-morning (9-11 AM), early afternoon (1-2 PM), and evening (6-7 PM)'
    },
    'youtube': {
        'peak_hours': [12, 13, 14, 15, 16, 17, 18, 19, 20],  # Afternoon to evening
        'best_hours': [14, 15, 17],
        'posts_per_week': 3,  # 2-3 videos per week
        'description': 'early afternoon (2-3 PM) and evening (5-8 PM) when viewers have leisure time'
    },
    'linkedin': {
        'peak_hours': [7, 8, 9, 10, 12, 17, 18],  # Business hours
        'best_hours': [8, 10, 12],
        'posts_per_week': 5,  # 1 per weekday
        'description': 'early morning (7-8 AM), mid-morning (9-10 AM), and lunch break (12 PM)'
    },
    'tiktok': {
        'peak_hours': [7, 8, 9, 12, 15, 19, 20, 21, 22],  # Morning, afternoon, and evening
        'best_hours': [9, 12, 19, 21],
        'posts_per_week': 14,  # 1-3 posts daily
        'description': 'morning (7-9 AM), lunch (12 PM), and especially evenings (7-10 PM)'
    },
    'threads': {
        'peak_hours': [8, 9, 12, 13, 18, 19, 20, 21],  # Similar to Twitter
        'best_hours': [9, 12, 19],
        'posts_per_week': 14,  # 2 posts daily
        'description': 'morning (8-9 AM), lunch (12-1 PM), and evenings (6-9 PM)'
    }
}

# Minimum posts threshold for using user data vs research data
MIN_POSTS_FOR_USER_DATA = 15

def get_timezone_display_name(tz_code: str) -> tuple:
    """Convert timezone code to display name and pytz timezone string"""
    tz_code_upper = tz_code.upper().strip()
    timezone_map = {
        "EST": ("Eastern Time", "America/New_York"),
        "EDT": ("Eastern Time", "America/New_York"),
        "CST": ("Central Time", "America/Chicago"),
        "CDT": ("Central Time", "America/Chicago"),
        "MST": ("Mountain Time", "America/Denver"),
        "MDT": ("Mountain Time", "America/Denver"),
        "PST": ("Pacific Time", "America/Los_Angeles"),
        "PDT": ("Pacific Time", "America/Los_Angeles"),
        "IST": ("India Time", "Asia/Kolkata"),
        "GMT": ("GMT", "Europe/London"),
        "UTC": ("UTC", "UTC"),
        "BST": ("British Time", "Europe/London"),
        "CET": ("Central European Time", "Europe/Paris"),
        "JST": ("Japan Time", "Asia/Tokyo"),
        "AEST": ("Australian Eastern Time", "Australia/Sydney"),
    }
    if tz_code_upper in timezone_map: return timezone_map[tz_code_upper]
    if "/" in tz_code:
        try:
            pytz.timezone(tz_code)
            city = tz_code.split("/")[-1].replace("_", " ")
            return (f"{city} Time", tz_code)
        except: pass
    return ("UTC", "UTC")

# Data Models
class InstagramPost(BaseModel):
    link: str
    type: str
    likes: int
    comments: int
    shares: int
    saved: int
    interactions: int
    views: int
    reach: int
    posting_time: str
    caption: str
    time_zone: str = "UTC"

class YouTubePost(BaseModel):
    link: str
    type: str
    likes: int
    comments: int
    shares: int
    views: int
    posting_time: str
    caption: str
    time_zone: str = "UTC"

class LinkedInPost(BaseModel):
    link: str
    type: str
    likes: int
    comments: int
    shares: int
    views: int
    posting_time: str
    caption: str
    time_zone: str = "UTC"

class ThreadsPost(BaseModel):
    link: str
    type: str
    likes: int
    replies: int
    reposts: int
    views: int
    posting_time: str
    caption: str
    time_zone: str = "UTC"

class TikTokPost(BaseModel):
    link: str
    type: str
    likes: int
    comments: int
    shares: int
    views: int
    posting_time: str
    caption: str
    time_zone: str = "UTC"

class FacebookPost(BaseModel):
    link: str
    type: str
    likes: int
    comments: int
    shares: int
    saved: int
    interactions: int
    views: int
    reach: int
    posting_time: str
    caption: str
    time_zone: str = "UTC"

PostData = Union[InstagramPost, YouTubePost, LinkedInPost, ThreadsPost, TikTokPost, FacebookPost]

class ContentAnalysis(BaseModel):
    post_link: str
    caption: str
    category: str
    content_theme: str
    content_sentiment: str
    engagement_prediction: str
    suggested_hashtags: List[str]
    content_subcategory: str
    target_audience: str
    content_strengths: List[str]
    improvement_suggestions: List[str]
    platform: str

class PlatformPerformance(BaseModel):
    instagram_engagement_score: float
    instagram_engagement_rate: float
    youtube_engagement_score: float
    youtube_engagement_rate: float
    linkedin_engagement_score: float
    linkedin_engagement_rate: float
    threads_engagement_score: float
    threads_engagement_rate: float
    tiktok_engagement_score: float
    tiktok_engagement_rate: float
    facebook_engagement_score: float
    facebook_engagement_rate: float
    best_platform: str
    best_platform_by_rate: str
    best_platform_by_score: str
    average_engagement_rate: float = 0.0

class OptimalTimeSlot(BaseModel):
    day: str
    hour: int
    engagement_score: float
    engagement_rate: float
    confidence: str
    confidence_score: float

class ContentCategoryInsight(BaseModel):
    category: str
    post_count: int
    avg_engagement: float
    avg_engagement_rate: float
    engagement_score_rank: int
    engagement_rate_rank: int
    consistency_score: float
    overall_score: float
    percentage: float

class PlatformTimeSlot(BaseModel):
    day: str
    time_range: str
    peak_hour: int
    engagement_score: float
    data_source: str

class PlatformSchedule(BaseModel):
    platform: str
    platform_display: str
    posts_per_week: int
    time_slots: List[PlatformTimeSlot]
    peak_hours_description: str
    data_source: str
    user_post_count: int

class PlatformSpecificSchedule(BaseModel):
    platform_schedules: Dict[str, PlatformSchedule]
    data_source: str
    use_user_data: bool
    total_posts_analyzed: int
    min_posts_threshold: int

class InsightsSection(BaseModel):
    timing: List[str]
    engagement: List[str]
    content: List[str]
    growth: List[str]

class RecommendationsSection(BaseModel):
    timing: List[Dict[str, Any]]
    engagement: List[Dict[str, Any]]
    content: List[Dict[str, Any]]
    growth: List[Dict[str, Any]]

class RecommendationResponse(BaseModel):
    total_posts_analyzed: int
    user_timezone: str
    timezone_display_name: str
    posts_by_platform: Dict[str, int]
    content_performance: Dict[str, float]
    content_performance_by_rate: Dict[str, float]
    time_performance: Dict[str, Any]
    time_performance_by_platform: Dict[str, Dict[str, Any]]  # NEW: Platform-specific time analysis
    platform_performance: PlatformPerformance
    top_performing_posts: List[Dict[str, Any]]
    insights: Dict[str, List[str]]  # Categorized insights only
    recommendations: Dict[str, List[Dict[str, Any]]]  # Categorized recommendations only
    confidence_levels: Dict[str, str]
    content_analysis: List[ContentAnalysis]
    content_category_breakdown: Dict[str, Any]
    content_category_insights: List[ContentCategoryInsight]
    optimal_posting_schedule: List[OptimalTimeSlot]
    platform_analysis: Dict[str, Any]
    platform_specific_schedule: Dict[str, Any]

class ContentAnalyzer:
    def __init__(self, api_key: str = None):
        self.cache = {}
        self.api_key = api_key or os.getenv("GROQ_API_KEY_RECOMMENDATION")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY_RECOMMENDATION not found in .env")
        self.client = Groq(api_key=self.api_key)

    def _fallback_category_detection(self, caption: str) -> str:
        """Fallback method to detect category based on keywords"""
        caption_lower = caption.lower()
        category_keywords = {
            "Funny": ["funny", "lol", "humor", "hilarious", "joke", "comedy", "laugh", "😂", "🤣"],
            "Educational": ["learn", "tutorial", "how to", "guide", "education", "teach", "lesson", "tip", "hack"],
            "Technology": ["tech", "ai", "software", "app", "digital", "coding", "programming", "innovation"],
            "News": ["news", "breaking", "update", "report", "announced", "viral news"],
            "Inspirational": ["inspire", "motivation", "motivational", "success", "dream", "believe", "achieve"],
            "Food": ["food", "recipe", "cooking", "delicious", "tasty", "chef", "meal", "restaurant"],
            "Fitness": ["fitness", "workout", "exercise", "gym", "training", "health", "muscle"],
            "Travel": ["travel", "trip", "vacation", "destination", "explore", "adventure", "journey"],
            "Fashion": ["fashion", "style", "outfit", "trending", "look", "wear"],
            "Beauty": ["beauty", "makeup", "skincare", "cosmetic", "hair"],
            "Music": ["music", "song", "artist", "album", "concert", "sing"],
            "Art": ["art", "artist", "painting", "drawing", "creative", "design"],
            "Entertainment": ["entertainment", "celebrity", "movie", "show", "series", "watch"],
            "Promotional": ["sale", "discount", "offer", "buy", "shop", "deal", "promo", "limited"],
            "Personal": ["personal", "my life", "my day", "feeling", "thoughts"],
            "Lifestyle": ["lifestyle", "living", "daily", "routine", "life"],
        }
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in caption_lower)
            if score > 0:
                category_scores[category] = score
        return max(category_scores, key=category_scores.get) if category_scores else "Other"

    def extract_category_from_caption(self, caption: str, platform: str) -> str:
        """Extract category from caption using LLM"""
        cache_key = f"{platform}_{hash(caption)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""Analyze this social media caption and determine the primary content category.

Platform: {platform}
Caption: "{caption}"

Respond with ONLY ONE WORD from this exact list:
Educational, Inspirational, Funny, Promotional, News, Personal, Entertainment, Lifestyle, Travel, Food, Fitness, Technology, Art, Music, Fashion, Beauty, Other

Choose the single most relevant category. Do not add any explanation or extra text."""

        valid_categories = [
            "Educational", "Inspirational", "Funny", "Promotional", "News",
            "Personal", "Entertainment", "Lifestyle", "Travel", "Food",
            "Fitness", "Technology", "Art", "Music", "Fashion", "Beauty", "Other"
        ]

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a social media content analyzer. Respond with only ONE category word from the provided list, nothing else."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            category = response.choices[0].message.content.strip()
            category_cleaned = ''.join(filter(str.isalpha, category))
            
            for valid_cat in valid_categories:
                if category_cleaned.lower() == valid_cat.lower():
                    category = valid_cat
                    break
            else:
                category_lower = category.lower()
                matched = False
                for valid_cat in valid_categories:
                    if valid_cat.lower() in category_lower:
                        category = valid_cat
                        matched = True
                        break
                if not matched:
                    category = self._fallback_category_detection(caption)

            self.cache[cache_key] = category
            return category
        except Exception as e:
            print(f"Error extracting category: {e}")
            return self._fallback_category_detection(caption)

    def analyze_content(self, post_link: str, caption: str, platform: str) -> ContentAnalysis:
        """Analyze content using Groq LLM"""
        category = self.extract_category_from_caption(caption, platform)
        cache_key = f"{post_link}_{category}_{hash(caption)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""Analyze this social media post and provide detailed insights.

Platform: {platform}
Caption: "{caption}"
Category: {category}

Provide a JSON response with:
1. content_theme: Main theme (1-2 words)
2. content_sentiment: positive/negative/neutral
3. engagement_prediction: high/medium/low
4. suggested_hashtags: Array of 3 relevant hashtags (without # symbol)
5. content_subcategory: More specific subcategory
6. target_audience: Who this content targets
7. content_strengths: Array of 3 strengths
8. improvement_suggestions: Array of 3 suggestions

Return ONLY valid JSON, no markdown formatting."""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a social media content analyzer. Return ONLY valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text[3:].lstrip()
                if result_text.startswith("json"):
                    result_text = result_text[4:].lstrip()
                if result_text.endswith("```"):
                    result_text = result_text[:-3].rstrip()
            
            result_text = result_text.strip()
            result = json.loads(result_text)

            analysis = ContentAnalysis(
                post_link=post_link,
                caption=caption,
                category=category,
                content_theme=result.get('content_theme', 'General'),
                content_sentiment=result.get('content_sentiment', 'neutral'),
                engagement_prediction=result.get('engagement_prediction', 'medium'),
                suggested_hashtags=result.get('suggested_hashtags', []),
                content_subcategory=result.get('content_subcategory', category),
                target_audience=result.get('target_audience', 'General audience'),
                content_strengths=result.get('content_strengths', []),
                improvement_suggestions=result.get('improvement_suggestions', []),
                platform=platform
            )

            self.cache[cache_key] = analysis
            return analysis
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return ContentAnalysis(
                post_link=post_link,
                caption=caption,
                category=category,
                content_theme="General",
                content_sentiment="neutral",
                engagement_prediction="medium",
                suggested_hashtags=[],
                content_subcategory=category,
                target_audience="General audience",
                content_strengths=[],
                improvement_suggestions=[],
                platform=platform
            )
class RecommendationEngine:
    def __init__(self, api_key: str = None):
        self.content_analyzer = ContentAnalyzer(api_key)
        self.processed_data = None
        self.user_timezone = "UTC"
        self.timezone_display_name = "UTC"
        self.analysis_start_date = None
        self.analysis_end_date = None

    def process_data(self, posts: List[PostData]) -> pd.DataFrame:
        """Process the raw post data into a structured DataFrame"""
        processed_posts = []
        utc_tz = pytz.UTC

        if len(posts) > 0:
            first_post = posts[0].model_dump()
            tz_code = first_post.get('time_zone', 'UTC')
            self.timezone_display_name, pytz_tz_string = get_timezone_display_name(tz_code)
            self.user_timezone = pytz_tz_string

            try:
                user_tz = pytz.timezone(self.user_timezone)
            except:
                user_tz = utc_tz
                self.timezone_display_name = "UTC"
                self.user_timezone = "UTC"

        for post in posts:
            post_dict = post.model_dump()
            platform = extract_platform_from_url(post_dict['link'])

            likes = post_dict.get('likes', 0)
            comments = post_dict.get('comments', 0)
            shares = post_dict.get('shares', 0)
            views = post_dict.get('views', 0)
            reach = post_dict.get('reach', 0)
            saved = post_dict.get('saved', 0)
            interactions = post_dict.get('interactions', 0)

            if platform == 'threads':
                comments = post_dict.get('replies', 0)
                shares = post_dict.get('reposts', 0)

            try:
                posting_time_str = post_dict['posting_time']
                posting_time_utc = None
                
                # Try ISO format first (e.g., "2025-11-11T13:15:00+0000")
                try:
                    posting_time_utc = datetime.fromisoformat(posting_time_str.replace('Z', '+00:00'))
                except:
                    pass
                
                # Try alternative formats if ISO failed
                if posting_time_utc is None:
                    # Remove timezone suffix (IST, UTC, etc.)
                    time_part = posting_time_str.rsplit(' ', 1)[0] if ' ' in posting_time_str else posting_time_str
                    
                    # Try multiple date format patterns
                    formats_to_try = [
                        "%d %b %Y, %I:%M %p",      # "30 Jan 2026, 03:37 am"
                        "%d %b %Y, %I:%M%p",       # "30 Jan 2026, 03:37am" (no space before am/pm)
                        "%Y-%m-%d %H:%M:%S",       # "2026-01-30 03:37:00"
                        "%Y-%m-%d",                # "2026-01-30"
                        "%d/%m/%Y %I:%M %p",       # "30/01/2026 03:37 am"
                        "%d-%m-%Y %I:%M %p",       # "30-01-2026 03:37 am"
                    ]
                    
                    for fmt in formats_to_try:
                        try:
                            posting_time_utc = datetime.strptime(time_part.strip(), fmt)
                            break
                        except:
                            continue
                    
                    # If still not parsed, use current time as fallback
                    if posting_time_utc is None:
                        posting_time_utc = datetime.now(utc_tz)
                        print(f"Warning: Could not parse timestamp '{posting_time_str}', using current time")
                
                # Ensure timezone awareness
                if posting_time_utc.tzinfo is None:
                    posting_time_utc = utc_tz.localize(posting_time_utc)
                
                # Convert to user timezone
                posting_time_user = posting_time_utc.astimezone(user_tz)
                
            except Exception as e:
                print(f"Error processing posting_time '{post_dict.get('posting_time', 'N/A')}': {e}")
                posting_time_user = datetime.now(user_tz)

            engagement_score = likes + (comments * 2) + (shares * 3) + saved
            engagement_rate = (engagement_score / max(views, reach, 1)) * 100

            category = self.content_analyzer.extract_category_from_caption(post_dict['caption'], platform)

            processed_posts.append({
                'link': post_dict['link'],
                'platform': platform,
                'content_type': post_dict['type'].lower(),
                'likes': likes,
                'comments': comments,
                'replies': post_dict.get('replies', 0) if platform == 'threads' else 0,
                'shares': shares,
                'reposts': post_dict.get('reposts', 0) if platform == 'threads' else 0,
                'saved': saved,
                'interactions': interactions,
                'views': views,
                'reach': reach,
                'engagement_score': engagement_score,
                'engagement_rate': engagement_rate,
                'posting_time': posting_time_user,
                'hour': posting_time_user.hour,
                'day_of_week': posting_time_user.weekday(),
                'day': posting_time_user.day,
                'month': posting_time_user.month,
                'caption': post_dict['caption'],
                'category': category
            })

        df = pd.DataFrame(processed_posts).sort_values('posting_time')
        if len(df) > 0:
            self.analysis_start_date = df['posting_time'].min()
            self.analysis_end_date = df['posting_time'].max()
        self.processed_data = df
        return df

    def analyze_content_performance(self) -> Dict[str, float]:
        """Analyze content performance by category (engagement score)"""
        return self.processed_data.groupby('category')['engagement_score'].mean().to_dict()

    def analyze_content_performance_by_rate(self) -> Dict[str, float]:
        """Analyze content performance by category (engagement rate)"""
        return self.processed_data.groupby('category')['engagement_rate'].mean().to_dict()

    def analyze_time_performance(self) -> Dict[str, Any]:
        """Analyze posting time performance"""
        hourly_perf = self.processed_data.groupby('hour')['engagement_score'].mean().to_dict()
        daily_perf = self.processed_data.groupby('day_of_week')['engagement_score'].mean().to_dict()
        hourly_rate_perf = self.processed_data.groupby('hour')['engagement_rate'].mean().to_dict()
        daily_rate_perf = self.processed_data.groupby('day_of_week')['engagement_rate'].mean().to_dict()

        hourly_count = self.processed_data.groupby('hour').size().to_dict()
        daily_count = self.processed_data.groupby('day_of_week').size().to_dict()

        hourly_confidence_scores = {}
        for hour, count in hourly_count.items():
            if count >= 5:
                hourly_confidence_scores[hour] = min(1.0, count / 10)
            else:
                hourly_confidence_scores[hour] = count / 5

        daily_confidence_scores = {}
        for day, count in daily_count.items():
            if count >= 3:
                daily_confidence_scores[day] = min(1.0, count / 7)
            else:
                daily_confidence_scores[day] = count / 3

        best_hour = max(hourly_perf, key=hourly_perf.get) if hourly_perf else 12
        best_day = max(daily_perf, key=daily_perf.get) if daily_perf else 0
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        best_day_name = day_names[best_day] if 0 <= best_day < 7 else 'Monday'

        hour_confidence = "High" if hourly_confidence_scores.get(best_hour, 0) >= 0.7 else \
                         "Medium" if hourly_confidence_scores.get(best_hour, 0) >= 0.4 else "Low"
        day_confidence = "High" if daily_confidence_scores.get(best_day, 0) >= 0.7 else \
                        "Medium" if daily_confidence_scores.get(best_day, 0) >= 0.4 else "Low"

        return {
            'hourly_performance': hourly_perf,
            'daily_performance': daily_perf,
            'hourly_rate_performance': hourly_rate_perf,
            'daily_rate_performance': daily_rate_perf,
            'best_hour': best_hour,
            'best_day': best_day,
            'best_day_name': best_day_name,
            'hour_confidence': hour_confidence,
            'day_confidence': day_confidence,
            'hourly_count': hourly_count,
            'daily_count': daily_count,
            'hourly_confidence_scores': hourly_confidence_scores,
            'daily_confidence_scores': daily_confidence_scores,
            'optimal_posting_schedule': self.get_optimal_posting_schedule()
        }

    def analyze_time_performance_by_platform(self) -> Dict[str, Dict[str, Any]]:
        """Analyze posting time performance for each platform separately"""
        platforms = self.processed_data['platform'].unique().tolist()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        platform_time_analysis = {}
        
        for platform in platforms:
            platform_df = self.processed_data[self.processed_data['platform'] == platform]
            
            if len(platform_df) == 0:
                continue
            
            # Hourly and daily performance for this platform
            hourly_perf = platform_df.groupby('hour')['engagement_score'].mean().to_dict()
            daily_perf = platform_df.groupby('day_of_week')['engagement_score'].mean().to_dict()
            hourly_rate_perf = platform_df.groupby('hour')['engagement_rate'].mean().to_dict()
            daily_rate_perf = platform_df.groupby('day_of_week')['engagement_rate'].mean().to_dict()
            
            hourly_count = platform_df.groupby('hour').size().to_dict()
            daily_count = platform_df.groupby('day_of_week').size().to_dict()
            
            # Confidence scores
            hourly_confidence_scores = {}
            for hour, count in hourly_count.items():
                if count >= 5:
                    hourly_confidence_scores[hour] = min(1.0, count / 10)
                else:
                    hourly_confidence_scores[hour] = count / 5
            
            daily_confidence_scores = {}
            for day, count in daily_count.items():
                if count >= 3:
                    daily_confidence_scores[day] = min(1.0, count / 7)
                else:
                    daily_confidence_scores[day] = count / 3
            
            # Best hour and day for this platform
            best_hour = max(hourly_perf, key=hourly_perf.get) if hourly_perf else 12
            best_day = max(daily_perf, key=daily_perf.get) if daily_perf else 0
            best_day_name = day_names[best_day] if 0 <= best_day < 7 else 'Monday'
            
            # Best by engagement rate (sometimes more useful)
            best_hour_by_rate = max(hourly_rate_perf, key=hourly_rate_perf.get) if hourly_rate_perf else 12
            best_day_by_rate = max(daily_rate_perf, key=daily_rate_perf.get) if daily_rate_perf else 0
            best_day_name_by_rate = day_names[best_day_by_rate] if 0 <= best_day_by_rate < 7 else 'Monday'
            
            # Confidence levels
            hour_confidence = "High" if hourly_confidence_scores.get(best_hour, 0) >= 0.7 else \
                             "Medium" if hourly_confidence_scores.get(best_hour, 0) >= 0.4 else "Low"
            day_confidence = "High" if daily_confidence_scores.get(best_day, 0) >= 0.7 else \
                            "Medium" if daily_confidence_scores.get(best_day, 0) >= 0.4 else "Low"
            
            # Get optimal posting schedule for this platform
            time_groups = platform_df.groupby(['day_of_week', 'hour']).agg({
                'engagement_score': 'mean',
                'engagement_rate': 'mean'
            }).reset_index()
            
            if len(time_groups) > 0:
                time_groups['count'] = platform_df.groupby(['day_of_week', 'hour']).size().values
                time_groups['confidence_score'] = time_groups['count'].apply(
                    lambda x: min(1.0, x / 3) if x >= 2 else x / 3
                )
                time_groups['confidence'] = time_groups['confidence_score'].apply(
                    lambda x: "High" if x >= 0.7 else "Medium" if x >= 0.4 else "Low"
                )
                time_groups = time_groups.sort_values('engagement_score', ascending=False)
                
                optimal_slots = []
                for _, row in time_groups.head(3).iterrows():  # Top 3 slots per platform
                    optimal_slots.append({
                        'day': day_names[int(row['day_of_week'])],
                        'hour': int(row['hour']),
                        'time_display': self._format_hour(int(row['hour'])),
                        'engagement_score': float(row['engagement_score']),
                        'engagement_rate': float(row['engagement_rate']),
                        'confidence': row['confidence'],
                        'confidence_score': float(row['confidence_score'])
                    })
            else:
                optimal_slots = []
            
            platform_time_analysis[platform] = {
                'platform': platform,
                'platform_display': platform.capitalize(),
                'post_count': len(platform_df),
                'hourly_performance': hourly_perf,
                'daily_performance': daily_perf,
                'hourly_rate_performance': hourly_rate_perf,
                'daily_rate_performance': daily_rate_perf,
                'best_hour': best_hour,
                'best_hour_display': self._format_hour(best_hour),
                'best_day': best_day,
                'best_day_name': best_day_name,
                'best_hour_by_rate': best_hour_by_rate,
                'best_day_by_rate': best_day_by_rate,
                'best_day_name_by_rate': best_day_name_by_rate,
                'hour_confidence': hour_confidence,
                'day_confidence': day_confidence,
                'hourly_count': hourly_count,
                'daily_count': daily_count,
                'hourly_confidence_scores': hourly_confidence_scores,
                'daily_confidence_scores': daily_confidence_scores,
                'optimal_posting_schedule': optimal_slots
            }
        
        return platform_time_analysis
    
    def _format_hour(self, hour: int) -> str:
        """Format hour to readable time string"""
        if hour == 0:
            return "12:00 AM"
        elif hour < 12:
            return f"{hour}:00 AM"
        elif hour == 12:
            return "12:00 PM"
        else:
            return f"{hour - 12}:00 PM"

    def get_optimal_posting_schedule(self) -> List[OptimalTimeSlot]:
        """Get optimal posting schedule with confidence scores"""
        time_groups = self.processed_data.groupby(['day_of_week', 'hour']).agg({
            'engagement_score': 'mean',
            'engagement_rate': 'mean'
        }).reset_index()
        time_groups['count'] = self.processed_data.groupby(['day_of_week', 'hour']).size().values

        time_groups['confidence_score'] = time_groups['count'].apply(
            lambda x: min(1.0, x / 3) if x >= 2 else x / 3
        )

        time_groups['confidence'] = time_groups['confidence_score'].apply(
            lambda x: "High" if x >= 0.7 else "Medium" if x >= 0.4 else "Low"
        )

        time_groups = time_groups.sort_values('engagement_score', ascending=False)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        optimal_slots = []
        for _, row in time_groups.head(2).iterrows():
            optimal_slots.append(OptimalTimeSlot(
                day=day_names[int(row['day_of_week'])],
                hour=int(row['hour']),
                engagement_score=float(row['engagement_score']),
                engagement_rate=float(row['engagement_rate']),
                confidence=row['confidence'],
                confidence_score=float(row['confidence_score'])
            ))
        return optimal_slots

    def get_platform_specific_schedule(self) -> Dict[str, Any]:
        """
        Generate platform-specific posting schedule.
        Uses user data if >= 15 posts in last 30 days, otherwise uses research-based data.
        Returns recommended time slots per platform with posting frequency.
        """
        df = self.processed_data
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Get platforms the user is active on
        user_platforms = df['platform'].unique().tolist()
        
        # Count posts in last 30 days to determine data source
        try:
            # Get timezone-aware current time
            current_time = datetime.now(pytz.UTC)
            thirty_days_ago = current_time - timedelta(days=30)
            
            # Make sure posting_time is timezone-aware for comparison
            if 'posting_time' in df.columns and len(df) > 0:
                # Filter posts from last 30 days
                recent_mask = df['posting_time'].apply(
                    lambda x: x.replace(tzinfo=pytz.UTC) if x.tzinfo is None else x
                ) >= thirty_days_ago
                recent_posts = df[recent_mask]
                total_recent_posts = len(recent_posts)
            else:
                total_recent_posts = len(df)
        except Exception:
            # Fallback: use all posts
            total_recent_posts = len(df)
        
        # Determine if we should use user data or research data
        use_user_data = total_recent_posts >= MIN_POSTS_FOR_USER_DATA
        data_source = "your posting history" if use_user_data else "industry research and best practices"
        
        platform_schedules = {}
        
        for platform in user_platforms:
            platform_lower = platform.lower()
            platform_df = df[df['platform'] == platform_lower]
            platform_post_count = len(platform_df)
            
            # Get platform config (default to instagram if not found)
            platform_config = PLATFORM_PEAK_HOURS.get(platform_lower, PLATFORM_PEAK_HOURS.get('instagram'))
            
            # Determine recommended posts per week for this platform
            posts_per_week = platform_config['posts_per_week']
            
            # Calculate number of time slots to recommend (match the posting frequency)
            num_slots = min(posts_per_week, 14)  # Cap at 14 slots (2 per day max in output)
            
            time_slots = []
            
            if use_user_data and platform_post_count >= 5:
                # Use user's own data for this platform
                platform_time_groups = platform_df.groupby(['day_of_week', 'hour']).agg({
                    'engagement_score': 'mean',
                    'engagement_rate': 'mean'
                }).reset_index()
                
                if len(platform_time_groups) > 0:
                    platform_time_groups['count'] = platform_df.groupby(['day_of_week', 'hour']).size().values
                    platform_time_groups = platform_time_groups.sort_values('engagement_score', ascending=False)
                    
                    # Get platform-specific avoid hours (e.g., Instagram avoids 5-9 PM)
                    avoid_hours = platform_config.get('avoid_hours', [])
                    
                    # Get top performing time slots (filtering out avoid_hours)
                    for _, row in platform_time_groups.head(num_slots * 2).iterrows():
                        hour = int(row['hour'])
                        day_idx = int(row['day_of_week'])
                        
                        # Skip platform-specific avoid hours
                        # (e.g., Instagram: Meta needs 2-5h to distribute, so 5-9 PM is too late)
                        if hour in avoid_hours:
                            continue
                        
                        time_slots.append({
                            'day': day_names[day_idx],
                            'time_range': format_time_range(hour),
                            'peak_hour': hour,
                            'engagement_score': float(row['engagement_score']),
                            'data_source': 'user_data'
                        })
                        
                        if len(time_slots) >= num_slots:
                            break
            
            # If not enough user data, use research-based recommendations
            if len(time_slots) < num_slots:
                best_hours = platform_config['best_hours']
                peak_hours = platform_config['peak_hours']
                
                # Combine best and peak hours, removing duplicates
                all_hours = best_hours + [h for h in peak_hours if h not in best_hours]
                
                # Distribute across days of the week
                slots_needed = num_slots - len(time_slots)
                used_combinations = set((s['day'], s['peak_hour']) for s in time_slots)
                
                # Calculate optimal day spacing for better distribution
                # Instead of sequential days, spread evenly across the week
                if slots_needed <= 7:
                    # For 1-7 posts, spread evenly across the week
                    # Calculate step size to distribute days evenly
                    step = 7 / slots_needed if slots_needed > 0 else 1
                    distributed_days = []
                    for i in range(slots_needed):
                        day_idx = int(i * step) % 7
                        distributed_days.append(day_idx)
                    
                    # Adjust for better distribution patterns
                    if slots_needed == 2:
                        distributed_days = [0, 4]  # Monday, Friday
                    elif slots_needed == 3:
                        distributed_days = [0, 3, 5]  # Monday, Thursday, Saturday
                    elif slots_needed == 4:
                        distributed_days = [0, 2, 4, 6]  # Monday, Wednesday, Friday, Sunday
                    elif slots_needed == 5:
                        distributed_days = [0, 1, 3, 4, 6]  # Mon, Tue, Thu, Fri, Sun
                    elif slots_needed == 6:
                        distributed_days = [0, 1, 2, 4, 5, 6]  # All except Wednesday
                    elif slots_needed == 7:
                        distributed_days = list(range(7))  # All days
                else:
                    # For more than 7 posts, use multiple slots per day but still spread
                    posts_per_day = max(1, (slots_needed + 6) // 7)
                    distributed_days = []
                    day_spacing_order = [0, 3, 6, 1, 4, 2, 5]  # Spread pattern: Mon, Thu, Sun, Tue, Fri, Wed, Sat
                    for day_idx in day_spacing_order:
                        for _ in range(posts_per_day):
                            if len(distributed_days) < slots_needed:
                                distributed_days.append(day_idx)
                
                # Rotate through hours for variety
                hour_index = 0
                for day_idx in distributed_days:
                    if slots_needed <= 0:
                        break
                    
                    # Try to find an unused hour for this day
                    attempts = 0
                    while attempts < len(all_hours):
                        hour = all_hours[hour_index % len(all_hours)]
                        
                        if (day_names[day_idx], hour) not in used_combinations:
                            time_slots.append({
                                'day': day_names[day_idx],
                                'time_range': format_time_range(hour),
                                'peak_hour': hour,
                                'engagement_score': 0,
                                'data_source': 'research_data'
                            })
                            used_combinations.add((day_names[day_idx], hour))
                            slots_needed -= 1
                            hour_index += 1
                            break
                        
                        hour_index += 1
                        attempts += 1
            
            # Sort by day and time
            day_order = {name: i for i, name in enumerate(day_names)}
            time_slots.sort(key=lambda x: (day_order.get(x['day'], 0), x['peak_hour']))
            
            platform_schedules[platform_lower] = {
                'platform': platform_lower,
                'platform_display': platform.capitalize(),
                'posts_per_week': posts_per_week,
                'time_slots': time_slots[:num_slots],
                'peak_hours_description': platform_config['description'],
                'data_source': 'user_data' if (use_user_data and platform_post_count >= 5) else 'research_data',
                'user_post_count': platform_post_count
            }
        
        return {
            'platform_schedules': platform_schedules,
            'data_source': data_source,
            'use_user_data': use_user_data,
            'total_posts_analyzed': total_recent_posts,
            'min_posts_threshold': MIN_POSTS_FOR_USER_DATA
        }

    def analyze_platform_performance(self) -> PlatformPerformance:
        """Analyze performance across different platforms"""
        platform_data = self.processed_data.groupby('platform').agg({
            'engagement_score': 'mean',
            'engagement_rate': 'mean'
        })

        platform_metrics = {}
        for platform in ['instagram', 'youtube', 'linkedin', 'threads', 'tiktok', 'facebook']:
            if platform in platform_data.index:
                engagement_rate = platform_data.loc[platform, 'engagement_rate']
                engagement_score = platform_data.loc[platform, 'engagement_score']
                engagement_rate = 0.0 if pd.isna(engagement_rate) else engagement_rate
                engagement_score = 0.0 if pd.isna(engagement_score) else engagement_score
                platform_metrics[platform] = {
                    'engagement_rate': engagement_rate,
                    'engagement_score': engagement_score
                }
            else:
                platform_metrics[platform] = {
                    'engagement_rate': 0.0,
                    'engagement_score': 0.0
                }

        best_platform_by_rate = max(platform_metrics.keys(), key=lambda p: platform_metrics[p]['engagement_rate'])
        best_platform_by_score = max(platform_metrics.keys(), key=lambda p: platform_metrics[p]['engagement_score'])

        # Calculate average engagement rate across all posts
        avg_engagement_rate = self.processed_data['engagement_rate'].mean()
        if pd.isna(avg_engagement_rate):
            avg_engagement_rate = 0.0

        return PlatformPerformance(
            instagram_engagement_score=float(platform_metrics.get('instagram', {}).get('engagement_score', 0)),
            instagram_engagement_rate=float(platform_metrics.get('instagram', {}).get('engagement_rate', 0)),
            youtube_engagement_score=float(platform_metrics.get('youtube', {}).get('engagement_score', 0)),
            youtube_engagement_rate=float(platform_metrics.get('youtube', {}).get('engagement_rate', 0)),
            linkedin_engagement_score=float(platform_metrics.get('linkedin', {}).get('engagement_score', 0)),
            linkedin_engagement_rate=float(platform_metrics.get('linkedin', {}).get('engagement_rate', 0)),
            threads_engagement_score=float(platform_metrics.get('threads', {}).get('engagement_score', 0)),
            threads_engagement_rate=float(platform_metrics.get('threads', {}).get('engagement_rate', 0)),
            tiktok_engagement_score=float(platform_metrics.get('tiktok', {}).get('engagement_score', 0)),
            tiktok_engagement_rate=float(platform_metrics.get('tiktok', {}).get('engagement_rate', 0)),
            facebook_engagement_score=float(platform_metrics.get('facebook', {}).get('engagement_score', 0)),
            facebook_engagement_rate=float(platform_metrics.get('facebook', {}).get('engagement_rate', 0)),
            best_platform=best_platform_by_rate,
            best_platform_by_rate=best_platform_by_rate,
            best_platform_by_score=best_platform_by_score,
            average_engagement_rate=float(avg_engagement_rate)
        )

    def analyze_platform_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of posts across platforms"""
        platform_counts = self.processed_data['platform'].value_counts().to_dict()
        platform_percentages = {platform: (count / len(self.processed_data)) * 100
                              for platform, count in platform_counts.items()}
        platform_engagement = self.processed_data.groupby('platform')['engagement_score'].mean().to_dict()
        platform_engagement_rate = self.processed_data.groupby('platform')['engagement_rate'].mean().to_dict()

        return {
            'platform_counts': platform_counts,
            'platform_percentages': platform_percentages,
            'platform_engagement': platform_engagement,
            'platform_engagement_rate': platform_engagement_rate
        }

    def get_top_performing_posts(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top n performing posts"""
        top_posts = self.processed_data.nlargest(n, 'engagement_score')
        result = top_posts.to_dict('records')

        for post in result:
            for key, value in post.items():
                if isinstance(value, np.integer):
                    post[key] = int(value)
                elif isinstance(value, np.floating):
                    post[key] = float(value)
                elif isinstance(value, np.bool_):
                    post[key] = bool(value)
        return result

    def get_content_category_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of content categories"""
        category_counts = self.processed_data['category'].value_counts().to_dict()
        category_engagement = self.processed_data.groupby('category')['engagement_score'].mean().to_dict()
        category_rate = self.processed_data.groupby('category')['engagement_rate'].mean().to_dict()
        category_std = self.processed_data.groupby('category')['engagement_score'].std().to_dict()

        breakdown = {}
        for category in category_counts:
            # Handle NaN values from std() when category has only 1 post
            std_value = category_std.get(category, 0)
            if pd.isna(std_value):
                std_value = 0.0
            
            breakdown[category] = {
                'post_count': int(category_counts[category]),
                'avg_engagement': float(category_engagement.get(category, 0)),
                'avg_engagement_rate': float(category_rate.get(category, 0)),
                'std_deviation': float(std_value),
                'percentage': float((category_counts[category] / len(self.processed_data)) * 100)
            }
        return breakdown

    def get_content_category_insights(self) -> List[ContentCategoryInsight]:
        """Get nuanced content category insights with multiple metrics"""
        breakdown = self.get_content_category_breakdown()
        categories = list(breakdown.keys())
        engagement_scores = [breakdown[cat]['avg_engagement'] for cat in categories]
        engagement_rates = [breakdown[cat]['avg_engagement_rate'] for cat in categories]

        score_rank = {cat: i+1 for i, cat in enumerate(sorted(categories, key=lambda x: engagement_scores[categories.index(x)], reverse=True))}
        rate_rank = {cat: i+1 for i, cat in enumerate(sorted(categories, key=lambda x: engagement_rates[categories.index(x)], reverse=True))}

        insights = []
        for i, category in enumerate(categories):
            insights.append(ContentCategoryInsight(
                category=category,
                post_count=breakdown[category]['post_count'],
                avg_engagement=breakdown[category]['avg_engagement'],
                avg_engagement_rate=breakdown[category]['avg_engagement_rate'],
                engagement_score_rank=score_rank.get(category, 1),
                engagement_rate_rank=rate_rank.get(category, 1),
                consistency_score=0.8,
                overall_score=0.85,
                percentage=breakdown[category]['percentage']
            ))
        return insights

    def generate_key_insights(self) -> Dict[str, Any]:
        """Generate dynamic, categorized key insights from the data"""
        df = self.processed_data
        
        timing_insights = self._generate_timing_insights(df)
        engagement_insights = self._generate_engagement_insights(df)
        content_insights = self._generate_content_insights(df)
        growth_insights = self._generate_growth_insights(df)
        
        # Combine all insights for backward compatibility (key_insights field)
        all_insights = timing_insights + engagement_insights + content_insights + growth_insights
        
        return {
            'all_insights': all_insights,
            'insights_by_category': {
                'timing': timing_insights,
                'engagement': engagement_insights,
                'content': content_insights,
                'growth': growth_insights
            }
        }
    
    def _generate_timing_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate timing-specific insights"""
        insights = []
        time_perf = self.analyze_time_performance()
        
        # Best posting hour insight
        best_hour = time_perf.get('best_hour', 12)
        hourly_perf = time_perf.get('hourly_performance', {})
        best_hour_engagement = hourly_perf.get(best_hour, 0)
        hour_formatted = datetime.strptime(str(best_hour), '%H').strftime('%I %p').lstrip('0')
        
        # Calculate how much better the best hour is
        avg_hourly = sum(hourly_perf.values()) / len(hourly_perf) if hourly_perf else 0
        hour_boost = ((best_hour_engagement - avg_hourly) / avg_hourly * 100) if avg_hourly > 0 else 0
        
        if hour_boost > 20:
            insights.append(
                f"⏰ Your sweet spot is around {hour_formatted}! Posts at this time get {hour_boost:.0f}% more engagement than your average. "
                f"That's a huge difference — try to schedule your most important content around this time."
            )
        elif best_hour_engagement > 0:
            insights.append(
                f"⏰ Posts around {hour_formatted} tend to perform well for you, averaging {best_hour_engagement:,.0f} interactions. "
                f"This could be when your audience is most active!"
            )
        
        # Best day insight with comparison
        best_day = time_perf.get('best_day_name', 'Monday')
        daily_perf = time_perf.get('daily_performance', {})
        best_day_num = time_perf.get('best_day', 0)
        best_day_engagement = daily_perf.get(best_day_num, 0)
        
        # Find worst day for comparison
        if len(daily_perf) > 1:
            worst_day_num = min(daily_perf, key=daily_perf.get)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            worst_day = day_names[worst_day_num] if worst_day_num < 7 else 'Monday'
            worst_engagement = daily_perf.get(worst_day_num, 0)
            day_diff = ((best_day_engagement - worst_engagement) / worst_engagement * 100) if worst_engagement > 0 else 0
            
            if day_diff > 30:
                insights.append(
                    f"📅 {best_day} is your magic day — posts get {day_diff:.0f}% more engagement than {worst_day}s! "
                    f"Consider saving your best content for {best_day}s."
                )
            else:
                insights.append(
                    f"📅 Your content performs fairly consistently throughout the week, with {best_day} having a slight edge "
                    f"({best_day_engagement:,.0f} avg interactions)."
                )
        
        # Weekend vs weekday insight
        weekday_engagement = sum([daily_perf.get(i, 0) for i in range(5)]) / 5 if len(daily_perf) > 0 else 0
        weekend_engagement = sum([daily_perf.get(i, 0) for i in [5, 6]]) / 2 if len(daily_perf) > 0 else 0
        
        if weekend_engagement > weekday_engagement * 1.2:
            insights.append(
                f"🎉 Your audience is more active on weekends! Weekend posts get about {((weekend_engagement - weekday_engagement) / weekday_engagement * 100):.0f}% more engagement. "
                f"Don't let those days go to waste."
            )
        elif weekday_engagement > weekend_engagement * 1.2 and weekend_engagement > 0:
            insights.append(
                f"💼 Your audience is more engaged during the workweek. Weekday posts outperform weekends by about "
                f"{((weekday_engagement - weekend_engagement) / weekend_engagement * 100):.0f}%. Focus your energy Monday through Friday!"
            )
        
        # Posting consistency insight
        hourly_counts = time_perf.get('hourly_count', {})
        if len(hourly_counts) > 0:
            most_posted_hour = max(hourly_counts, key=hourly_counts.get)
            most_posted_count = hourly_counts[most_posted_hour]
            hour_str = datetime.strptime(str(most_posted_hour), '%H').strftime('%I %p').lstrip('0')
            
            if most_posted_hour != best_hour and most_posted_count > 3:
                best_hour_str = datetime.strptime(str(best_hour), '%H').strftime('%I %p').lstrip('0')
                insights.append(
                    f"💡 Interesting pattern: You usually post around {hour_str}, but your content actually performs better around {best_hour_str}. "
                    f"Try shifting your posting schedule to catch that engagement boost!"
                )
        
        return insights
    
    def _generate_engagement_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate engagement-specific insights"""
        insights = []
        
        # Overall engagement stats
        total_engagement = df['engagement_score'].sum()
        total_likes = df['likes'].sum() if 'likes' in df.columns else 0
        total_comments = df['comments'].sum() if 'comments' in df.columns else 0
        total_shares = df['shares'].sum() if 'shares' in df.columns else 0
        avg_rate = df['engagement_rate'].mean()
        
        # Engagement rate quality assessment
        if avg_rate >= 10:
            insights.append(
                f"🔥 Wow! Your average engagement rate is {avg_rate:.1f}% — that's exceptional! "
                f"Your content is really resonating with your audience. Keep doing what you're doing!"
            )
        elif avg_rate >= 5:
            insights.append(
                f"💪 Your {avg_rate:.1f}% engagement rate is above average! Your followers are actively connecting with your content. "
                f"You're building a loyal community."
            )
        elif avg_rate >= 2:
            insights.append(
                f"📊 Your engagement rate of {avg_rate:.1f}% is solid. There's room to grow, but you're on the right track. "
                f"Focus on content that sparks conversations!"
            )
        else:
            insights.append(
                f"📈 Your current engagement rate is {avg_rate:.1f}%. Let's work on boosting that! "
                f"The key is creating content that makes people want to like, comment, and share."
            )
        
        # Comment-to-like ratio insight
        if total_likes > 0 and total_comments > 0:
            comment_ratio = (total_comments / max(total_likes, 1)) * 100
            if comment_ratio > 10:
                insights.append(
                    f"💬 Your audience loves to chat! You're getting {comment_ratio:.1f} comments for every 100 likes — "
                    f"that's a sign of a highly engaged community. Comments are gold for the algorithm!"
                )
            elif comment_ratio < 3:
                insights.append(
                    f"💬 You're getting lots of likes, but fewer comments ({comment_ratio:.1f} per 100 likes). "
                    f"Try ending your captions with a question to encourage more conversation!"
                )
        
        # Share performance insight
        if total_shares > 0:
            share_ratio = (total_shares / len(df))
            if share_ratio > 100:
                insights.append(
                    f"🚀 People are actively sharing your content! Averaging {share_ratio:.0f} shares per post means your content is "
                    f"worth passing along — that's the best kind of organic growth."
                )
            elif share_ratio > 30:
                insights.append(
                    f"📤 Your posts average {share_ratio:.0f} shares each. That's solid! Shareable content extends your reach beyond your followers."
                )
        
        # High engagement posts analysis
        if len(df) >= 5:
            top_20_pct = df.nlargest(max(1, len(df) // 5), 'engagement_score')
            bottom_20_pct = df.nsmallest(max(1, len(df) // 5), 'engagement_score')
            
            top_avg = top_20_pct['engagement_score'].mean()
            bottom_avg = bottom_20_pct['engagement_score'].mean()
            
            # Protect against NaN values
            if pd.isna(top_avg):
                top_avg = 0
            if pd.isna(bottom_avg):
                bottom_avg = 0
            
            if bottom_avg > 0 and top_avg > bottom_avg * 5:
                insights.append(
                    f"⭐ Your top-performing posts get {(top_avg / bottom_avg):.1f}x more engagement than your lowest performers. "
                    f"Study what made those posts work — the topic, the format, the caption style — and do more of that!"
                )
        
        # Platform-specific engagement insight
        platform_rates = df.groupby('platform')['engagement_rate'].mean()
        if len(platform_rates) > 1:
            best_platform = platform_rates.idxmax()
            worst_platform = platform_rates.idxmin()
            best_rate = platform_rates[best_platform]
            worst_rate = platform_rates[worst_platform]
            
            # Protect against NaN values
            if pd.isna(best_rate):
                best_rate = 0
            if pd.isna(worst_rate):
                worst_rate = 0
            
            if best_rate > worst_rate * 1.5:
                insights.append(
                    f"📱 Your {best_platform.capitalize()} content gets {best_rate:.1f}% engagement vs {worst_rate:.1f}% on {worst_platform.capitalize()}. "
                    f"Your {best_platform.capitalize()} audience is really connecting with you — consider doubling down there!"
                )
        
        return insights
    
    def _generate_content_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate content-specific insights"""
        insights = []
        category_perf = self.analyze_content_performance()
        
        if len(category_perf) > 0:
            # Best category insight
            best_category = max(category_perf, key=category_perf.get)
            best_engagement = category_perf[best_category]
            avg_engagement = sum(category_perf.values()) / len(category_perf) if len(category_perf) > 0 else 0
            
            # Protect against NaN
            if pd.isna(best_engagement):
                best_engagement = 0
            if pd.isna(avg_engagement):
                avg_engagement = 0
            
            boost_pct = ((best_engagement - avg_engagement) / avg_engagement * 100) if avg_engagement > 0 else 0
            
            if boost_pct > 50:
                insights.append(
                    f"🏆 Your {best_category} content is crushing it! It performs {boost_pct:.0f}% better than your average. "
                    f"This is clearly your strongest content type — your audience wants more!"
                )
            elif boost_pct > 20:
                insights.append(
                    f"🏆 {best_category} content is your top performer, getting {boost_pct:.0f}% more engagement than average. "
                    f"Keep this as a core part of your content mix."
                )
            
            # Underperforming category insight
            if len(category_perf) > 2:
                worst_category = min(category_perf, key=category_perf.get)
                worst_engagement = category_perf[worst_category]
                worst_pct = ((avg_engagement - worst_engagement) / avg_engagement * 100) if avg_engagement > 0 else 0
                
                worst_count = len(df[df['category'] == worst_category])
                if worst_count > 1 and worst_pct > 30:
                    insights.append(
                        f"📉 Your {worst_category} content isn't hitting the same marks — it's {worst_pct:.0f}% below average. "
                        f"Consider either refreshing your approach or focusing on content types that perform better."
                    )
            
            # Content diversity insight
            unique_categories = len(category_perf)
            if unique_categories >= 5:
                insights.append(
                    f"🎨 You're creating diverse content across {unique_categories} categories! This helps you reach different audiences. "
                    f"Just make sure to prioritize your top performers."
                )
            elif unique_categories <= 2:
                insights.append(
                    f"🎯 You're focused on {unique_categories} content categories. That's okay for building expertise, "
                    f"but experimenting with new content types could help you discover untapped potential!"
                )
        
        # Caption length insight (if we have caption data)
        if 'caption' in df.columns:
            df['caption_length'] = df['caption'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
            
            # Compare engagement for long vs short captions
            median_length = df['caption_length'].median()
            short_captions = df[df['caption_length'] <= median_length]
            long_captions = df[df['caption_length'] > median_length]
            
            if len(short_captions) > 2 and len(long_captions) > 2:
                short_engagement = short_captions['engagement_score'].mean()
                long_engagement = long_captions['engagement_score'].mean()
                
                # Protect against NaN values
                if pd.isna(short_engagement):
                    short_engagement = 0
                if pd.isna(long_engagement):
                    long_engagement = 0
                
                if short_engagement > 0 and long_engagement > short_engagement * 1.3:
                    insights.append(
                        f"📝 Longer captions are working for you! Posts with detailed captions get about "
                        f"{((long_engagement - short_engagement) / short_engagement * 100):.0f}% more engagement. Your audience appreciates the depth!"
                    )
                elif long_engagement > 0 and short_engagement > long_engagement * 1.3:
                    insights.append(
                        f"📝 Short and punchy works! Your concise captions outperform longer ones by "
                        f"{((short_engagement - long_engagement) / long_engagement * 100):.0f}%. Keep it snappy!"
                    )
        
        return insights
    
    def _generate_growth_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate growth-related insights"""
        insights = []
        
        # Summary stats
        total_posts = len(df)
        total_engagement = df['engagement_score'].sum()
        
        insights.append(
            f"📊 Across your {total_posts} posts, you've generated {total_engagement:,.0f} total interactions. "
            f"That's an average of {(total_engagement / total_posts):,.0f} per post — let's work on pushing that higher!"
        )
        
        # Trend analysis
        if len(df) >= 5:
            split_point = int(len(df) * 0.6)
            early_df = df.iloc[:split_point]
            recent_df = df.iloc[split_point:]
            
            early_engagement = early_df['engagement_score'].mean()
            recent_engagement = recent_df['engagement_score'].mean()
            
            # Protect against NaN values
            if pd.isna(early_engagement):
                early_engagement = 0
            if pd.isna(recent_engagement):
                recent_engagement = 0
            
            change_pct = ((recent_engagement - early_engagement) / early_engagement * 100) if early_engagement > 0 else 0
            
            if change_pct > 30:
                insights.append(
                    f"🚀 Amazing momentum! Your recent posts are performing {change_pct:.0f}% better than your earlier ones. "
                    f"Whatever you changed is working — keep that energy going!"
                )
            elif change_pct > 10:
                insights.append(
                    f"📈 You're trending upward! Recent content is getting {change_pct:.0f}% more engagement. "
                    f"You're finding your groove."
                )
            elif change_pct < -20:
                insights.append(
                    f"📉 Heads up: your recent posts are down {abs(change_pct):.0f}% from earlier performance. "
                    f"This happens to everyone — let's look at what was working before and get back on track!"
                )
            else:
                insights.append(
                    f"➡️ Your engagement is holding steady. Consistency is great, but let's experiment with new approaches "
                    f"to break through to the next level!"
                )
        
        # Posting frequency insight
        if 'posting_time' in df.columns and len(df) > 1:
            try:
                df_with_dates = df.copy()
                df_with_dates['posting_time'] = pd.to_datetime(df_with_dates['posting_time'])
                date_range = (df_with_dates['posting_time'].max() - df_with_dates['posting_time'].min()).days
                
                if date_range > 0:
                    posts_per_week = (total_posts / date_range) * 7
                    
                    if posts_per_week >= 7:
                        insights.append(
                            f"📅 You're posting about {posts_per_week:.1f}x per week — that's excellent consistency! "
                            f"Regular posting keeps you in the algorithm's good graces."
                        )
                    elif posts_per_week >= 3:
                        insights.append(
                            f"📅 At {posts_per_week:.1f} posts per week, you've got a decent rhythm. "
                            f"Bumping up to 5-7 posts weekly could give your reach a significant boost."
                        )
                    elif posts_per_week < 2:
                        insights.append(
                            f"📅 You're averaging about {posts_per_week:.1f} posts per week. "
                            f"Increasing to 3-4 posts weekly could help you stay top of mind with your audience."
                        )
            except Exception:
                pass
        
        # Platform presence insight
        platform_counts = df['platform'].value_counts()
        if len(platform_counts) >= 3:
            platforms_list = ", ".join([p.capitalize() for p in platform_counts.index[:3]])
            insights.append(
                f"🌐 You're active on {len(platform_counts)} platforms ({platforms_list}). "
                f"Multi-platform presence helps you reach different audiences and reduces algorithm dependency!"
            )
        elif len(platform_counts) == 1:
            single_platform = platform_counts.index[0].capitalize()
            insights.append(
                f"🎯 You're focused on {single_platform}. That's great for mastering one platform, "
                f"but expanding to 2-3 platforms could significantly grow your overall reach."
            )
        
        return insights

    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate categorized, dynamic actionable recommendations"""
        df = self.processed_data
        category_perf = self.analyze_content_performance()
        time_perf = self.analyze_time_performance()
        
        timing_recommendations = self._generate_timing_recommendations(df, time_perf)
        engagement_recommendations = self._generate_engagement_recommendations(df)
        content_recommendations = self._generate_content_recommendations(df, category_perf)
        growth_recommendations = self._generate_growth_recommendations(df)
        
        # Combine all recommendations for backward compatibility
        all_recommendations = timing_recommendations + engagement_recommendations + content_recommendations + growth_recommendations
        
        return {
            'all_recommendations': all_recommendations,
            'recommendations_by_category': {
                'timing': timing_recommendations,
                'engagement': engagement_recommendations,
                'content': content_recommendations,
                'growth': growth_recommendations
            }
        }
    
    def _generate_timing_recommendations(self, df: pd.DataFrame, time_perf: Dict) -> List[Dict[str, Any]]:
        """Generate timing-specific recommendations"""
        recommendations = []
        
        # Platform-specific timing recommendation
        platform_schedule = self.get_platform_specific_schedule()
        platform_schedules = platform_schedule.get('platform_schedules', {})
        data_source = platform_schedule.get('data_source', 'research data')
        use_user_data = platform_schedule.get('use_user_data', False)
        
        if len(platform_schedules) > 0:
            schedule_parts = []
            total_weekly_posts = 0
            
            for platform, schedule in platform_schedules.items():
                platform_name = schedule['platform_display']
                posts_per_week = schedule['posts_per_week']
                time_slots = schedule['time_slots']
                total_weekly_posts += posts_per_week
                
                if len(time_slots) > 0:
                    days_with_times = {}
                    for slot in time_slots:
                        day = slot['day']
                        time_range = slot['time_range']
                        if day not in days_with_times:
                            days_with_times[day] = []
                        days_with_times[day].append(time_range)
                    
                    day_schedules = []
                    for day, times in days_with_times.items():
                        if len(times) == 1:
                            day_schedules.append(f"{day} between {times[0]}")
                        else:
                            times_text = " and ".join(times[:2])
                            day_schedules.append(f"{day} between {times_text}")
                    
                    if len(day_schedules) > 4:
                        day_text = ", ".join(day_schedules[:3]) + f", and {len(day_schedules) - 3} more days"
                    elif len(day_schedules) > 1:
                        day_text = ", ".join(day_schedules[:-1]) + f", and {day_schedules[-1]}"
                    else:
                        day_text = day_schedules[0] if day_schedules else ""
                    
                    schedule_parts.append({
                        'platform': platform_name,
                        'posts_per_week': posts_per_week,
                        'schedule_text': day_text,
                        'peak_description': schedule['peak_hours_description']
                    })
            
            if use_user_data:
                intro_text = (
                    "Based on how your audience has responded to your posts, I've put together a personalized "
                    "posting schedule for each platform you use. These times are when your followers are most active."
                )
            else:
                intro_text = (
                    "Since you're still building up your posting history, I've pulled together the best times to post "
                    "based on what works for most creators on each platform. As you post more, I'll fine-tune these "
                    "recommendations using your actual results."
                )
            
            action_lines = [f"Here's your weekly posting schedule:"]
            for part in schedule_parts:
                action_lines.append(f"• {part['platform']}: Post {part['posts_per_week']}x per week — try {part['schedule_text']}")
            
            action_text = "\n".join(action_lines)
            
            recommendations.append({
                'priority': 'high',
                'category': 'Posting Schedule',
                'action': action_text,
                'reason': intro_text,
                'expected_outcome': (
                    f"Following this schedule can boost your reach by 30-50%. Each platform has its own rhythm — "
                    f"by posting when your audience is most active, you're giving your content the best chance to be seen and shared."
                ),
                'platform_schedules': platform_schedules,
                'data_source': data_source,
                'total_weekly_posts': total_weekly_posts
            })
        
        # Best hour optimization recommendation
        best_hour = time_perf.get('best_hour', 12)
        hourly_perf = time_perf.get('hourly_performance', {})
        hourly_counts = time_perf.get('hourly_count', {})
        
        if len(hourly_counts) > 0:
            most_posted_hour = max(hourly_counts, key=hourly_counts.get)
            if most_posted_hour != best_hour:
                best_hour_str = datetime.strptime(str(best_hour), '%H').strftime('%I %p').lstrip('0')
                current_hour_str = datetime.strptime(str(most_posted_hour), '%H').strftime('%I %p').lstrip('0')
                
                best_engagement = hourly_perf.get(best_hour, 0)
                current_engagement = hourly_perf.get(most_posted_hour, 0)
                improvement = ((best_engagement - current_engagement) / current_engagement * 100) if current_engagement > 0 else 0
                
                if improvement > 15:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'Timing Optimization',
                        'action': f"Shift your posting time from {current_hour_str} to around {best_hour_str}",
                        'reason': (
                            f"You typically post around {current_hour_str}, but your content performs {improvement:.0f}% better "
                            f"when posted around {best_hour_str}. Your audience is more active at that time!"
                        ),
                        'expected_outcome': (
                            f"This simple timing shift could boost your engagement by {improvement:.0f}% without changing anything about your content. "
                            f"It's one of the easiest wins in social media!"
                        )
                    })
        
        # Weekend strategy recommendation
        daily_perf = time_perf.get('daily_performance', {})
        daily_counts = time_perf.get('daily_count', {})
        
        if len(daily_perf) >= 5:
            weekend_engagement = (daily_perf.get(5, 0) + daily_perf.get(6, 0)) / 2
            weekday_engagement = sum([daily_perf.get(i, 0) for i in range(5)]) / 5
            
            weekend_posts = daily_counts.get(5, 0) + daily_counts.get(6, 0)
            weekday_posts = sum([daily_counts.get(i, 0) for i in range(5)])
            
            if weekend_engagement > weekday_engagement * 1.3 and weekend_posts < weekday_posts * 0.3:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Weekend Strategy',
                    'action': "Add more weekend posts — your audience is 30%+ more active on Saturday and Sunday!",
                    'reason': (
                        f"Right now only {weekend_posts} of your {len(df)} posts are on weekends, but weekend posts "
                        f"get significantly higher engagement. You're missing out on your audience's leisure time!"
                    ),
                    'expected_outcome': (
                        "Posting 2-3 times on weekends could capture this untapped engagement. "
                        "People have more time to scroll and engage when they're relaxing!"
                    )
                })
            elif weekday_engagement > weekend_engagement * 1.3 and weekday_posts < weekend_posts * 0.5:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Weekday Strategy',
                    'action': "Focus more on weekday posts — that's when your audience is most engaged!",
                    'reason': (
                        f"Your weekday posts significantly outperform weekends. "
                        f"Your audience might be more active during work breaks or commutes."
                    ),
                    'expected_outcome': (
                        "Shifting more content to Monday-Friday could boost your overall engagement by 25-40%."
                    )
                })
        
        return recommendations
    
    def _generate_engagement_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate dynamic engagement recommendations based on actual user data"""
        recommendations = []
        
        total_comments = df['comments'].sum() if 'comments' in df.columns else 0
        total_likes = df['likes'].sum() if 'likes' in df.columns else 0
        total_shares = df['shares'].sum() if 'shares' in df.columns else 0
        total_posts = len(df)
        avg_comments = total_comments / total_posts if total_posts > 0 else 0
        avg_engagement_rate = df['engagement_rate'].mean()
        
        # Dynamic engagement strategy based on comment-to-like ratio
        if total_likes > 0:
            comment_ratio = (total_comments / total_likes) * 100
            
            if comment_ratio < 3:
                # Low comments - focus on conversation starters
                recommendations.append({
                    'priority': 'high',
                    'category': 'Engagement Strategy',
                    'action': (
                        "Turn lurkers into commenters with these caption hooks:\n"
                        "• Ask specific questions: \"What's YOUR go-to [topic]?\" instead of generic \"What do you think?\"\n"
                        "• Create light debates: \"Hot take: [opinion]. Agree or disagree?\"\n"
                        "• Use fill-in-the-blanks: \"My morning can't start without ___\"\n"
                        "• Ask for recommendations: \"Drop your favorite [related item] in the comments!\""
                    ),
                    'reason': (
                        f"You're getting {total_likes:,} likes but only {total_comments:,} comments — that's about "
                        f"{comment_ratio:.1f} comments per 100 likes. Comments are worth 5x more than likes for algorithm reach! "
                        f"Your audience likes your content but isn't talking about it yet."
                    ),
                    'expected_outcome': (
                        "Posts with good comment sections get 2-3x more reach. By asking better questions, you can "
                        f"turn those {total_likes:,} passive likes into active conversations!"
                    )
                })
            elif comment_ratio >= 10:
                # High comments - focus on building community
                recommendations.append({
                    'priority': 'high',
                    'category': 'Community Building',
                    'action': (
                        "Your comment game is strong! Level up with these tactics:\n"
                        "• Reply to comments with follow-up questions to keep threads going\n"
                        "• Highlight great comments in your Stories with credit\n"
                        "• Create content based on common questions from your comments\n"
                        "• Pin your best comment responses to encourage more discussion"
                    ),
                    'reason': (
                        f"You're averaging {avg_comments:.0f} comments per post — that's excellent! "
                        f"You've built an engaged community. Now let's turn those commenters into superfans."
                    ),
                    'expected_outcome': (
                        "Active community management can increase comment frequency by 40% and boost your "
                        "overall reach. Your most engaged followers become your best ambassadors!"
                    )
                })
            else:
                # Medium comments - focus on response speed
                recommendations.append({
                    'priority': 'high',
                    'category': 'Engagement Strategy',
                    'action': (
                        "Boost your engagement momentum with the 60-minute rule:\n"
                        "• Reply to every comment within the first hour of posting\n"
                        "• Set 15-minute check-ins for the first hour after posting\n"
                        "• Use voice notes or video replies for standout responses\n"
                        "• React to comments even if you can't type a full reply"
                    ),
                    'reason': (
                        f"With {avg_comments:.0f} comments per post, you've got a foundation to build on. "
                        f"The first hour after posting is critical — early engagement signals tell the algorithm your post is worth showing more."
                    ),
                    'expected_outcome': (
                        "Creators who engage in the first hour typically see 2x the reach. "
                        "Fast responses create a snowball effect as more people see the active conversation."
                    )
                })
        
        # Share optimization (if share data exists and is low)
        if total_shares > 0 and total_likes > 0:
            share_ratio = (total_shares / total_likes) * 100
            
            if share_ratio < 5:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Shareability',
                    'action': (
                        "Make your content more shareable:\n"
                        "• Create \"tag a friend who...\" moments in your content\n"
                        "• Share useful tips or hacks people will want to save and send\n"
                        "• Add relatable content that makes people say \"this is so me!\"\n"
                        "• Create carousel posts with valuable takeaways worth saving"
                    ),
                    'reason': (
                        f"You're getting {total_shares:,} shares across {total_posts} posts — about {(total_shares/total_posts):.0f} per post. "
                        f"Shares are the ultimate compliment: people are putting your content in front of their own followers!"
                    ),
                    'expected_outcome': (
                        "Each share exposes your content to a whole new audience. Increasing shareability can "
                        "dramatically expand your organic reach beyond your existing followers."
                    )
                })
        
        # Story/cross-promotion based on engagement patterns
        if 'platform' in df.columns:
            platforms = df['platform'].unique()
            if 'instagram' in platforms or 'facebook' in platforms:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Cross-Promotion',
                    'action': (
                        "Amplify your posts with Stories:\n"
                        "• Tease new posts in your Stories with \"New post alert!\" stickers\n"
                        "• Share behind-the-scenes of your content creation\n"
                        "• Use countdown stickers for upcoming content\n"
                        "• Reshare posts to Stories after 24 hours for a second wave of engagement"
                    ),
                    'reason': (
                        "Stories get 3-5x more visibility than feed posts for many accounts. "
                        "Using Stories to drive people to your posts creates a powerful engagement loop."
                    ),
                    'expected_outcome': (
                        "Creators who promote feed posts in Stories see 20-40% higher engagement. "
                        "It's like giving your content a second chance to be discovered!"
                    )
                })
        
        # Engagement rate specific recommendations
        if avg_engagement_rate < 2:
            recommendations.append({
                'priority': 'high',
                'category': 'Audience Connection',
                'action': (
                    "Reconnect with your audience:\n"
                    "• Post more personal/behind-the-scenes content to build connection\n"
                    "• Go live to interact in real-time and boost visibility\n"
                    "• Use polls and quizzes in Stories to increase interaction\n"
                    "• Reply to DMs and story mentions promptly"
                ),
                'reason': (
                    f"Your engagement rate is {avg_engagement_rate:.1f}%, which suggests your content might not be "
                    f"reaching your full audience or they're scrolling past. Let's work on making deeper connections!"
                ),
                'expected_outcome': (
                    "Showing more authenticity typically boosts engagement rates by 30-50%. "
                    "People follow people, not just content — let them see the real you!"
                )
            })
        elif avg_engagement_rate >= 8:
            recommendations.append({
                'priority': 'medium',
                'category': 'Engagement Leverage',
                'action': (
                    "Your engagement is amazing — now leverage it:\n"
                    "• Partner with creators who have similar engagement rates\n"
                    "• Consider launching a community (Discord, newsletter, etc.)\n"
                    "• Test monetization opportunities with your engaged audience\n"
                    "• Create exclusive content for your most engaged followers"
                ),
                'reason': (
                    f"With a {avg_engagement_rate:.1f}% engagement rate, you're in the top tier of creators! "
                    f"Your audience is highly invested in what you share. This opens doors to partnerships and monetization."
                ),
                'expected_outcome': (
                    "Highly engaged audiences are worth 10x more than large passive followings. "
                    "Brands specifically look for engagement rates like yours!"
                )
            })
        
        return recommendations
    
    def _generate_content_recommendations(self, df: pd.DataFrame, category_perf: Dict) -> List[Dict[str, Any]]:
        """Generate content-specific recommendations"""
        recommendations = []
        
        if len(category_perf) > 0:
            # Top performing content recommendation
            best_category = max(category_perf, key=category_perf.get)
            best_engagement = category_perf[best_category]
            category_posts = len(df[df['category'] == best_category])
            category_pct = (category_posts / len(df)) * 100
            
            total_posts = len(df)
            target_percentage = 45
            needed_posts = max(3, int((target_percentage / 100) * total_posts) - category_posts)
            
            recommendations.append({
                'priority': 'high',
                'category': 'Content Strategy',
                'action': f"Double down on {best_category} content — share {needed_posts} more this week!",
                'reason': (
                    f"Your {best_category} content is a hit! It averages {best_engagement:,.0f} interactions per post. "
                    f"Currently only {category_pct:.0f}% of your content is {best_category} — there's room to grow!"
                ),
                'expected_outcome': (
                    f"Increasing {best_category} content to 40-50% of your posts could significantly boost overall engagement. "
                    f"Your audience is clearly telling you what they want!"
                )
            })
            
            # Second best category recommendation (if exists)
            if len(category_perf) >= 2:
                sorted_categories = sorted(category_perf.items(), key=lambda x: x[1], reverse=True)
                second_best = sorted_categories[1][0]
                second_engagement = sorted_categories[1][1]
                second_posts = len(df[df['category'] == second_best])
                
                if second_posts < 3:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'Content Diversification',
                        'action': f"Experiment more with {second_best} content — it's showing promise!",
                        'reason': (
                            f"Your {second_best} posts average {second_engagement:,.0f} engagement, but you've only posted {second_posts}. "
                            f"There might be untapped potential here!"
                        ),
                        'expected_outcome': (
                            f"Testing more {second_best} content will help you understand if it can become "
                            f"a consistent performer in your content mix."
                        )
                    })
            
            # Underperforming content adjustment
            if len(category_perf) >= 3:
                worst_category = min(category_perf, key=category_perf.get)
                worst_engagement = category_perf[worst_category]
                avg_engagement = sum(category_perf.values()) / len(category_perf) if len(category_perf) > 0 else 0
                
                # Protect against NaN
                if pd.isna(worst_engagement):
                    worst_engagement = 0
                if pd.isna(avg_engagement):
                    avg_engagement = 0
                
                worst_posts = len(df[df['category'] == worst_category])
                if avg_engagement > 0 and worst_posts >= 2 and worst_engagement < avg_engagement * 0.6:
                    recommendations.append({
                        'priority': 'low',
                        'category': 'Content Optimization',
                        'action': f"Rethink your {worst_category} approach or reduce it",
                        'reason': (
                            f"Your {worst_category} content is performing {((avg_engagement - worst_engagement) / avg_engagement * 100):.0f}% "
                            f"below average. It might not resonate with your current audience."
                        ),
                        'expected_outcome': (
                            f"Either refresh how you do {worst_category} content (different format, angle, or style) "
                            f"or reallocate that effort to your top performers."
                        )
                    })
        
        # Content format recommendations based on platform
        if 'content_type' in df.columns:
            content_types = df['content_type'].value_counts()
            if 'video' in content_types.index:
                video_engagement = df[df['content_type'] == 'video']['engagement_score'].mean()
                other_engagement = df[df['content_type'] != 'video']['engagement_score'].mean()
                
                # Protect against NaN values
                if pd.isna(video_engagement):
                    video_engagement = 0
                if pd.isna(other_engagement):
                    other_engagement = 0
                
                if other_engagement > 0 and video_engagement > other_engagement * 1.3:
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'Content Format',
                        'action': "Lean into video — it's your engagement superpower!",
                        'reason': (
                            f"Your video content gets {((video_engagement - other_engagement) / other_engagement * 100):.0f}% more engagement "
                            f"than other formats. Platforms are also pushing video content more in 2024."
                        ),
                        'expected_outcome': (
                            "Increasing video content from current levels could boost overall engagement significantly. "
                            "Even simple talking-head videos can outperform static posts!"
                        )
                    })
        
        return recommendations
    
    def _generate_growth_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate growth-focused recommendations"""
        recommendations = []
        total_posts = len(df)
        
        # Posting frequency recommendation
        if total_posts < 15:
            posts_needed = 20 - total_posts
            recommendations.append({
                'priority': 'medium',
                'category': 'Posting Consistency',
                'action': f"Build momentum with {posts_needed} more posts over the next 2 weeks",
                'reason': (
                    f"With {total_posts} posts, you're still building your presence. "
                    f"Consistency is key for algorithm favor — platforms reward creators who show up regularly."
                ),
                'expected_outcome': (
                    "Posting 3-5 times per week for 2 weeks straight can increase your reach by 40-60%. "
                    "The algorithm starts recognizing you as an active creator!"
                )
            })
        
        # Platform diversification
        platform_counts = df['platform'].value_counts()
        if len(platform_counts) > 0:
            top_platform = platform_counts.index[0]
            top_platform_pct = (platform_counts.iloc[0] / total_posts) * 100
            
            if top_platform_pct > 70 and len(platform_counts) < 3:
                suggestions = {
                    'instagram': ['TikTok', 'YouTube Shorts'],
                    'tiktok': ['Instagram Reels', 'YouTube Shorts'],
                    'youtube': ['Instagram', 'TikTok'],
                    'facebook': ['Instagram', 'LinkedIn'],
                    'linkedin': ['Twitter/X', 'YouTube'],
                    'threads': ['Instagram', 'Twitter/X']
                }
                suggested = suggestions.get(top_platform, ['Instagram', 'TikTok'])
                
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Platform Expansion',
                    'action': f"Expand to {suggested[0]} — your content would do well there!",
                    'reason': (
                        f"You're {top_platform_pct:.0f}% focused on {top_platform.capitalize()}. "
                        f"Diversifying protects you from algorithm changes and reaches new audiences."
                    ),
                    'expected_outcome': (
                        f"Repurposing your best {top_platform.capitalize()} content for {suggested[0]} "
                        f"can grow your total audience by 30-50% with minimal extra effort."
                    )
                })
        
        # Trend/engagement analysis for growth
        if total_posts >= 5:
            split_point = int(total_posts * 0.6)
            recent_df = df.iloc[split_point:]
            early_df = df.iloc[:split_point]
            
            recent_engagement = recent_df['engagement_score'].mean()
            early_engagement = early_df['engagement_score'].mean()
            
            if recent_engagement < early_engagement * 0.8:
                recommendations.append({
                    'priority': 'high',
                    'category': 'Trend Reversal',
                    'action': (
                        "Your recent posts are down — here's how to bounce back:\n"
                        "• Revisit what made your top-performing posts work\n"
                        "• Try posting at different times to test new audiences\n"
                        "• Engage heavily with your audience this week (comments, DMs, Stories)\n"
                        "• Consider a bold, attention-grabbing post to reset momentum"
                    ),
                    'reason': (
                        f"Your recent posts are averaging {((early_engagement - recent_engagement) / early_engagement * 100):.0f}% "
                        f"less engagement than earlier. This happens to everyone — the key is to adapt quickly!"
                    ),
                    'expected_outcome': (
                        "Most engagement dips are temporary. Taking action within 1-2 weeks typically "
                        "helps creators recover and sometimes even surpass previous levels."
                    )
                })
            elif recent_engagement > early_engagement * 1.3:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Momentum Capture',
                    'action': (
                        "You're on a hot streak — capitalize on it:\n"
                        "• Post more frequently this week to ride the wave\n"
                        "• Double down on the content style that's working\n"
                        "• Engage extra heavily with new followers\n"
                        "• Consider going live to connect with your growing audience"
                    ),
                    'reason': (
                        f"Your recent content is performing {((recent_engagement - early_engagement) / early_engagement * 100):.0f}% "
                        f"better than before! The algorithm is favoring you right now."
                    ),
                    'expected_outcome': (
                        "Creators who capitalize on momentum often see sustained growth. "
                        "This is the time to push — you have algorithmic wind at your back!"
                    )
                })
        
        return recommendations


class RecommendationService:
    def __init__(self, api_key: str = None):
        self.engine = RecommendationEngine(api_key)

    def generate_recommendations(self, posts: List[PostData]) -> Dict[str, Any]:
        """Main entry point - generates full recommendation response"""
        df = self.engine.process_data(posts)

        content_perf = self.engine.analyze_content_performance()
        content_perf_rate = self.engine.analyze_content_performance_by_rate()
        time_perf = self.engine.analyze_time_performance()
        time_perf_by_platform = self.engine.analyze_time_performance_by_platform()  # NEW
        platform_perf = self.engine.analyze_platform_performance()
        platform_dist = self.engine.analyze_platform_distribution()
        top_posts = self.engine.get_top_performing_posts(5)
        category_breakdown = self.engine.get_content_category_breakdown()
        category_insights = self.engine.get_content_category_insights()
        
        # Get categorized insights and recommendations (no duplicates)
        insights_data = self.engine.generate_key_insights()
        recommendations_data = self.engine.generate_recommendations()
        
        # Only use categorized versions (no duplicates)
        insights = insights_data.get('insights_by_category', {})
        recommendations = recommendations_data.get('recommendations_by_category', {})

        posts_by_platform = df["platform"].value_counts().to_dict()
        content_analysis = []
        for _, row in df.iterrows():
            content_analysis.append(
                self.engine.content_analyzer.analyze_content(
                    row["link"], row["caption"], row["platform"]
                )
            )

        confidence_levels = {
            "time_analysis": time_perf.get("hour_confidence", "Medium"),
            "day_analysis": time_perf.get("day_confidence", "Medium"),
        }
        optimal_slots = time_perf.get("optimal_posting_schedule", [])
        
        # Get platform-specific posting schedule
        platform_specific_schedule = self.engine.get_platform_specific_schedule()

        # ✅ FIX: Clean all data for JSON serialization
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            elif isinstance(data, float):
                if math.isnan(data) or math.isinf(data):
                    return 0.0
                return data
            elif isinstance(data, (np.floating, np.integer)):
                val = float(data)
                return 0.0 if math.isnan(val) or math.isinf(val) else val
            return data

        response_data = RecommendationResponse(
            total_posts_analyzed=len(df),
            user_timezone=self.engine.user_timezone,
            timezone_display_name=self.engine.timezone_display_name,
            posts_by_platform=clean_for_json(posts_by_platform),
            content_performance=clean_for_json(content_perf),
            content_performance_by_rate=clean_for_json(content_perf_rate),
            time_performance=clean_for_json(time_perf),
            time_performance_by_platform=clean_for_json(time_perf_by_platform),  # NEW
            platform_performance=platform_perf,
            top_performing_posts=clean_for_json(top_posts),
            insights=clean_for_json(insights),
            recommendations=clean_for_json(recommendations),
            confidence_levels=confidence_levels,
            content_analysis=content_analysis,
            content_category_breakdown=clean_for_json(category_breakdown),
            content_category_insights=category_insights,
            optimal_posting_schedule=optimal_slots,
            platform_analysis=clean_for_json(platform_dist),
            platform_specific_schedule=clean_for_json(platform_specific_schedule),
        )
        
        data_dict = response_data.model_dump()
        cleaned_data = clean_nan_inf(data_dict)

        return {
            "status": "success",
            "data": cleaned_data
        }
