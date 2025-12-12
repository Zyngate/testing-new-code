# stelle_backend/services/recommendation_service.py

import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from groq import Groq
from typing import List
import pytz

# ---------------------------------------
# Utility Functions (moved from combined file)
# ---------------------------------------

def extract_platform_from_url(url: str) -> str:
    if "instagram.com" in url: return "instagram"
    if "youtube.com" in url: return "youtube"
    if "linkedin.com" in url: return "linkedin"
    if "threads.net" in url: return "threads"
    if "tiktok.com" in url: return "tiktok"
    if "facebook.com" in url: return "facebook"
    return "unknown"

def format_hour_12h(hour: int):
    suffix = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    return f"{hour12}:00 {suffix}"

def get_timezone_display_name(tz_code: str):
    try:
        pytz.timezone(tz_code)
        name = tz_code.split("/")[-1].replace("_", " ")
        return name + " Time", tz_code
    except:
        return "UTC", "UTC"


# ---------------------------------------
# Content Analyzer Class
# ---------------------------------------

class ContentAnalysis(BaseModel):
    post_link: str
    caption: str
    category: str
    content_theme: str
    content_sentiment: str
    engagement_prediction: str
    suggested_hashtags: list
    content_subcategory: str
    target_audience: str
    content_strengths: list
    improvement_suggestions: list
    platform: str


class ContentAnalyzer:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.cache = {}

    def extract_category_from_caption(self, caption: str, platform: str) -> str:
        cache_key = f"{platform}_{hash(caption)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
Return one category:
Educational, Inspirational, Funny, Promotional, News, Personal,
Entertainment, Lifestyle, Travel, Food, Fitness, Technology,
Art, Music, Fashion, Beauty, Other

Caption: "{caption}"
Platform: {platform}"
"""

        valid = {
            "educational", "inspirational", "funny", "promotional", "news",
            "personal", "entertainment", "lifestyle", "travel", "food",
            "fitness", "technology", "art", "music", "fashion", "beauty", "other"
        }

        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Return only one word."},
                    {"role": "user", "content": prompt},
                ]
            )
            category = resp.choices[0].message.content.strip().lower()
            category = re.sub(r"[^a-z]", "", category)
            if category not in valid:
                category = "other"
        except:
            category = "other"

        category = category.capitalize()
        self.cache[cache_key] = category
        return category

    def analyze_content(self, link: str, caption: str, platform: str):
        category = self.extract_category_from_caption(caption, platform)
        cache_key = f"{link}_{category}_{hash(caption)}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
Analyze this post and return only JSON:
content_theme,
content_sentiment,
engagement_prediction,
suggested_hashtags,
content_subcategory,
target_audience,
content_strengths,
improvement_suggestions

Caption: "{caption}"
Platform: {platform}"
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return JSON only"},
                    {"role": "user", "content": prompt},
                ]
            )
            data = json.loads(response.choices[0].message.content)

        except:
            data = {
                "content_theme": "unknown",
                "content_sentiment": "neutral",
                "engagement_prediction": "medium",
                "suggested_hashtags": [],
                "content_subcategory": "general",
                "target_audience": "general",
                "content_strengths": [],
                "improvement_suggestions": [],
            }

        analysis = ContentAnalysis(
            post_link=link,
            caption=caption,
            category=category,
            platform=platform,
            **data,
        )

        self.cache[cache_key] = analysis
        return analysis


# ---------------------------------------
# Recommendation Engine Service
# ---------------------------------------

class RecommendationService:
    def __init__(self, api_key: str):
        self.analyzer = ContentAnalyzer(api_key)
        self.timezone_display = "UTC"
        self.user_timezone = "UTC"
        self.df = None

    def process_posts(self, posts: List[dict]):
        processed = []

        if len(posts) > 0:
            tz_name, tz_code = get_timezone_display_name(posts[0]["time_zone"])
            self.timezone_display = tz_name
            self.user_timezone = tz_code

        tz = pytz.timezone(self.user_timezone)

        for post in posts:
            dt = datetime.fromisoformat(post["posting_time"].replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            dt = dt.astimezone(tz)

            platform = extract_platform_from_url(post["link"])
            analysis = self.analyzer.analyze_content(post["link"], post["caption"], platform)

            score = post["likes"] + 2 * post["comments"] + 3 * post["shares"]
            rate = (score / max(post.get("views", 1), post.get("reach", 1))) * 100

            processed.append({
                "platform": platform,
                "hour": dt.hour,
                "day": dt.weekday(),
                "category": analysis.category,
                "engagement_score": score,
                "engagement_rate": rate,
            })

        self.df = pd.DataFrame(processed)
        return self.df

def generate_recommendations(self, posts):
    df = self.process_posts(posts)

    # Basic winners
    best_hour = df.groupby("hour")["engagement_score"].mean().idxmax()
    best_day = df.groupby("day_of_week")["engagement_score"].mean().idxmax()
    best_category = df.groupby("category")["engagement_score"].mean().idxmax()

    # NEW — Best performing platform
    best_platform = df.groupby("platform")["engagement_score"].mean().idxmax()

    # --- Build Recommendations ---
    recommendations = {
        "when_to_post": f"Post around {format_hour_12h(best_hour)} on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][best_day]}.",
        "what_to_post": f"{best_category} content performs best for your audience.",
        "where_to_post": f"{best_platform.capitalize()} gives you the highest engagement."
    }

    # --- Build Insights ---
    insights = []
    insights.append(f"{best_category} posts perform best overall.")
    insights.append(f"Your audience responds strongest on {best_platform.capitalize()}.")
    insights.append(
        f"Engagement spikes around {format_hour_12h(best_hour)} on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][best_day]}."
    )

    return {
        "timezone": self.timezone_display_name,
        "best_hour": format_hour_12h(best_hour),
        "best_day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][best_day],
        "best_category": best_category,
        "best_platform": best_platform.capitalize(),
        "recommendations": recommendations,
        "insights": insights,
        "total_posts": len(df)
    }
