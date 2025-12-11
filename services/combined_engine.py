# combined_engine.py

import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from collections import Counter
from pydantic import BaseModel
from groq import Groq
import pytz

# ======================================================
#                UTILS (Merged utils.py)
# ======================================================

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


# ======================================================
#                CONTENT ANALYZER (Merged)
# ======================================================

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
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.cache = {}

    # -----------------------------------
    # CATEGORY EXTRACTION
    # -----------------------------------
    def extract_category_from_caption(self, caption: str, platform: str) -> str:
        cache_key = f"{platform}_{hash(caption)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
Analyze this caption and return ONLY ONE category from:

Educational, Inspirational, Funny, Promotional, News, Personal, 
Entertainment, Lifestyle, Travel, Food, Fitness, Technology, 
Art, Music, Fashion, Beauty, Other

Platform: {platform}
Caption: "{caption}"
"""

        valid = {
            "educational", "inspirational", "funny", "promotional", "news",
            "personal", "entertainment", "lifestyle", "travel", "food",
            "fitness", "technology", "art", "music", "fashion", "beauty", "other"
        }

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Return only one word category"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            category = response.choices[0].message.content.strip().lower()
            category = re.sub(r"[^a-zA-Z]", "", category)

            if category not in valid:
                category = "other"

            category = category.capitalize()
            self.cache[cache_key] = category
            return category

        except Exception:
            return "Other"

    # -----------------------------------
    # FULL CAPTION ANALYSIS
    # -----------------------------------
    def analyze_content(self, link: str, caption: str, platform: str) -> ContentAnalysis:
        category = self.extract_category_from_caption(caption, platform)
        cache_key = f"{link}_{category}_{hash(caption)}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""
Analyze this post and RETURN ONLY JSON with:
content_theme,
content_sentiment,
engagement_prediction,
suggested_hashtags,
content_subcategory,
target_audience,
content_strengths,
improvement_suggestions

Caption: "{caption}"
Platform: {platform}
Category: {category}

Return pure JSON only.
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return JSON only"},
                    {"role": "user", "content": prompt}
                ]
            )

            data = json.loads(response.choices[0].message.content)

        except Exception:
            data = {
                "content_theme": "unknown",
                "content_sentiment": "neutral",
                "engagement_prediction": "medium",
                "suggested_hashtags": [],
                "content_subcategory": "general",
                "target_audience": "general",
                "content_strengths": [],
                "improvement_suggestions": []
            }

        analysis = ContentAnalysis(
            post_link=link,
            caption=caption,
            category=category,
            platform=platform,
            **data
        )

        self.cache[cache_key] = analysis
        return analysis


# ======================================================
#        RECOMMENDATION ENGINE (Merged)
# ======================================================

class RecommendationEngine:
    def __init__(self, groq_api_key):
        self.content_analyzer = ContentAnalyzer(groq_api_key)
        self.processed_data = None
        self.user_timezone = "UTC"
        self.timezone_display_name = "UTC"

    # ---------------------------------------------------
    #                PROCESS POSTS
    # ---------------------------------------------------
    def process_posts(self, posts):
        """
        Converts raw post objects into a normalized DataFrame.
        (This replaces your earlier process_data function.)
        """
        processed = []

        # detect timezone once
        if len(posts) > 0:
            tz_display, tz_real = get_timezone_display_name(posts[0].time_zone)
            self.timezone_display_name = tz_display
            self.user_timezone = tz_real

        user_tz = pytz.timezone(self.user_timezone)

        for post in posts:
            p = post.dict()
            platform = extract_platform_from_url(p["link"])

            # timezone convert
            try:
                dt = datetime.fromisoformat(p["posting_time"].replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = pytz.UTC.localize(dt)
                dt = dt.astimezone(user_tz)
            except:
                dt = datetime.now(user_tz)

            # engagement calculations (simple)
            likes = p.get("likes", 0)
            comments = p.get("comments", 0)
            shares = p.get("shares", 0)
            views = p.get("views", 0)
            reach = p.get("reach", 0)

            engagement_score = likes + (2 * comments) + (3 * shares)
            engagement_rate = (engagement_score / max(views, reach, 1)) * 100

            # content analysis
            c = self.content_analyzer.analyze_content(p["link"], p["caption"], platform)

            processed.append({
                "link": p["link"],
                "platform": platform,
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "views": views,
                "reach": reach,
                "caption": p["caption"],

                "engagement_score": engagement_score,
                "engagement_rate": engagement_rate,

                "posting_time": dt,
                "hour": dt.hour,
                "day_of_week": dt.weekday(),

                # from LLM
                "category": c.category,
                "content_theme": c.content_theme,
                "content_sentiment": c.content_sentiment,
                "target_audience": c.target_audience
            })

        df = pd.DataFrame(processed)
        self.processed_data = df
        return df

    # ---------------------------------------------------
    #         BASIC PERFORMANCE ANALYTICS
    # ---------------------------------------------------

    def best_category(self):
        df = self.processed_data
        return df.groupby("category")["engagement_score"].mean().idxmax()

    def best_hour(self):
        df = self.processed_data
        return df.groupby("hour")["engagement_score"].mean().idxmax()

    def best_day(self):
        df = self.processed_data
        return df.groupby("day_of_week")["engagement_score"].mean().idxmax()

    # ---------------------------------------------------
    #         FINAL RECOMMENDATION GENERATOR
    # ---------------------------------------------------

    def generate_recommendations(self, posts):
        df = self.process_posts(posts)

        best_cat = self.best_category()
        best_hr = self.best_hour()
        best_dy = self.best_day()

        return {
            "timezone": self.timezone_display_name,
            "total_posts_analyzed": len(df),
            "best_category": best_cat,
            "best_hour": format_hour_12h(best_hr),
            "best_day": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][best_dy]
        }
