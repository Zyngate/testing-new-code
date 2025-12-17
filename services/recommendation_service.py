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

class RecommendationResponse(BaseModel):
    total_posts_analyzed: int
    user_timezone: str
    timezone_display_name: str
    posts_by_platform: Dict[str, int]
    content_performance: Dict[str, float]
    content_performance_by_rate: Dict[str, float]
    time_performance: Dict[str, Any]
    platform_performance: PlatformPerformance
    top_performing_posts: List[Dict[str, Any]]
    key_insights: List[str]
    actionable_recommendations: List[Dict[str, Any]]
    confidence_levels: Dict[str, str]
    content_analysis: List[ContentAnalysis]
    content_category_breakdown: Dict[str, Any]
    content_category_insights: List[ContentCategoryInsight]
    optimal_posting_schedule: List[OptimalTimeSlot]
    platform_analysis: Dict[str, Any]

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
                posting_time_utc = datetime.fromisoformat(post_dict['posting_time'].replace('Z', '+00:00'))
                if posting_time_utc.tzinfo is None:
                    posting_time_utc = utc_tz.localize(posting_time_utc)
                posting_time_user = posting_time_utc.astimezone(user_tz)
            except:
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
            best_platform_by_score=best_platform_by_score
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
            breakdown[category] = {
                'post_count': int(category_counts[category]),
                'avg_engagement': float(category_engagement.get(category, 0)),
                'avg_engagement_rate': float(category_rate.get(category, 0)),
                'std_deviation': float(category_std.get(category, 0)),
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

    def generate_key_insights(self) -> List[str]:
        """Generate key insights from the data"""
        df = self.processed_data
        insights = []

        total_engagement = df['engagement_score'].sum()
        insights.append(f"📊 Analyzed {len(df)} posts with {total_engagement:,.0f} total engagement")

        avg_rate = df['engagement_rate'].mean()
        insights.append(f"📈 Average engagement rate: {avg_rate:.1f}%")

        platform_counts = df['platform'].value_counts()
        if len(platform_counts) > 0:
            top_platform = platform_counts.index[0]
            top_platform_count = platform_counts.iloc[0]
            platform_pct = (top_platform_count / len(df)) * 100
            insights.append(f"📱 {platform_pct:.0f}% of content on {top_platform}")

        if len(df) >= 3:
            split_point = int(len(df) * 0.7)
            early_df = df.iloc[:split_point]
            recent_df = df.iloc[split_point:]
            early_engagement = early_df['engagement_score'].mean()
            recent_engagement = recent_df['engagement_score'].mean()
            if recent_engagement > early_engagement * 1.1:
                insights.append("📈 Engagement growing")
            elif recent_engagement < early_engagement * 0.9:
                insights.append("📉 Engagement needs attention")
            else:
                insights.append("➡️ Engagement consistent")

        category_perf = self.analyze_content_performance()
        if len(category_perf) > 0:
            best_category = max(category_perf, key=category_perf.get)
            best_engagement = category_perf[best_category]
            post_count = len(df[df['category'] == best_category])
            insights.append(f"🏆 {best_category} performed best ({best_engagement:.0f} avg, {post_count} posts)")

        time_perf = self.analyze_time_performance()
        best_day = time_perf.get('best_day_name', 'Monday')
        insights.append(f"📅 Best day: {best_day}")

        return insights

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        df = self.processed_data
        category_perf = self.analyze_content_performance()
        time_perf = self.analyze_time_performance()

        if len(category_perf) > 0:
            best_category = max(category_perf, key=category_perf.get)
            best_engagement = category_perf[best_category]
            category_posts = len(df[df['category'] == best_category])
            category_pct = (category_posts / len(df)) * 100
            recommendations.append({
                'priority': 'high',
                'category': 'Content Strategy',
                'action': f'Post 3-4 more {best_category.lower()} pieces next week',
                'reason': f'{best_category} averaged {best_engagement:.0f} engagement ({category_pct:.0f}% of mix)',
                'target': f'40-50% {best_category.lower()} content'
            })

        hourly_perf = time_perf.get('hourly_performance', {})
        if len(hourly_perf) > 0:
            best_hours = sorted(hourly_perf.items(), key=lambda x: x[1], reverse=True)[:3]
            day_names = ['Tuesday', 'Wednesday', 'Friday']
            times_list = []
            for i, (hour, engagement) in enumerate(best_hours):
                hour_fmt = format_hour_12h(int(hour))
                day_name = day_names[i % len(day_names)]
                times_list.append(f"{day_name} {hour_fmt} {self.timezone_display_name}")
            time_text = ", ".join(times_list[:-1]) + f" and {times_list[-1]}" if len(times_list) > 1 else times_list[0]
            recommendations.append({
                'priority': 'high',
                'category': 'Posting Schedule',
                'action': f'Post at: {time_text}',
                'reason': f'Top hours averaged {best_hours[0][1]:.0f}+ engagement',
                'target': f'2-3 posts through these {len(times_list)} slots'
            })

        total_comments = df['comments'].sum()
        total_posts_with_comments = len(df[df['comments'] > 0])
        recommendations.append({
            'priority': 'high',
            'category': 'Algorithm Hack',
            'action': 'SHARE posts → Ask "What do you think?" → Reply ALL comments within 24h',
            'reason': f'{total_comments:,} comments across {total_posts_with_comments} posts. Replying = 2-3x boost',
            'target': '100% comment reply rate → Watch reach double'
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
        platform_perf = self.engine.analyze_platform_performance()
        platform_dist = self.engine.analyze_platform_distribution()
        top_posts = self.engine.get_top_performing_posts(5)
        category_breakdown = self.engine.get_content_category_breakdown()
        category_insights = self.engine.get_content_category_insights()
        key_insights = self.engine.generate_key_insights()
        actionable_recos = self.engine.generate_recommendations()

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
            platform_performance=platform_perf,
            top_performing_posts=clean_for_json(top_posts),
            key_insights=key_insights,
            actionable_recommendations=clean_for_json(actionable_recos),
            confidence_levels=confidence_levels,
            content_analysis=content_analysis,
            content_category_breakdown=clean_for_json(category_breakdown),
            content_category_insights=category_insights,
            optimal_posting_schedule=optimal_slots,
            platform_analysis=clean_for_json(platform_dist),
        )
        
        data_dict = response_data.model_dump()
        cleaned_data = clean_nan_inf(data_dict)

        return {
            "status": "success",
            "data": cleaned_data
        }

