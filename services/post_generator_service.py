# stelle_backend/services/post_generator_service.py

from typing import List, Dict, Any
from groq import AsyncGroq
from enum import Enum
from config import logger
from services.ai_service import (
    rate_limited_groq_call,
    query_internet_via_groq,
    groq_generate_text
)

MODEL = "llama-3.3-70b-versatile"

class Platforms(str, Enum):
    Instagram = "instagram"
    Twitter = "twitter"
    LinkedIn = "linkedin"
    Facebook = "facebook"
    Reddit = "reddit"

    platform_list = [Instagram, Twitter, LinkedIn, Facebook, Reddit]

PLATFORM_STYLES = {
    "instagram": "Write a sassy, trendy, aesthetic Gen-Z caption.",
    "linkedin": "Write a corporate, professional, confident caption.",
    "facebook": "Write a warm, friendly, conversational caption.",
    "reddit": "Write an informative, community-style caption.",
    "twitter": "Write a short, punchy, high-impact tweet."
}

# --------------------------------------------------------
# KEYWORDS
# --------------------------------------------------------
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    prompt = f"Generate exactly 3 short keywords for: {query}. Only keywords, comma-separated."
    try:
        r = await rate_limited_groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        keywords = [x.strip() for x in r.choices[0].message.content.split(",")]
        return (keywords + ["", "", ""])[:3]
    except:
        return ["", "", ""]

# --------------------------------------------------------
# HASHTAGS
# --------------------------------------------------------
async def fetch_trending_hashtags_post(client: AsyncGroq, seed_keywords, _platforms):
    tags = []
    for kw in seed_keywords:
        p = f"Give trending hashtags for: {kw}. Only hashtags separated by spaces."
        try:
            content = await query_internet_via_groq(p)
            tags.extend([x.replace("#", "") for x in content.split()])
        except:
            pass
    unique = list(dict.fromkeys(tags))
    return [f"#{t}" for t in unique[:15]]

# --------------------------------------------------------
# SEO
# --------------------------------------------------------
async def fetch_seo_keywords_post(client: AsyncGroq, seed_keywords):
    out = []
    for kw in seed_keywords:
        p = f"Give 5 SEO keywords for: {kw}. Only keywords, comma-separated."
        try:
            c = await query_internet_via_groq(p)
            out.extend([x.strip() for x in c.split(",")])
        except:
            pass
    return list(dict.fromkeys(out))[:10]

# --------------------------------------------------------
# MULTI-PLATFORM CAPTION GENERATOR (FINAL)
# --------------------------------------------------------
async def generate_caption_post(query: str, seed_keywords: list, hashtags: list, platforms: list):
    results = {}

    for platform in platforms:
        p = platform.lower()
        style = PLATFORM_STYLES.get(p, "Write a clean, creative caption.")

        prompt = f"""
Generate a caption for: {p}

Context: {query}

Style: {style}

Rules:
- NO hashtags in caption.
- NO keywords mentioned.
- ONE final clean caption.
"""

        try:
            caption = await groq_generate_text(MODEL, prompt)
        except Exception as e:
            logger.error(f"Caption generation failed for {p}: {e}")
            caption = query

        results[p] = caption.strip()

    return {"captions": results, "hashtags": hashtags}

# --------------------------------------------------------
# CLASSIFIER
# --------------------------------------------------------
async def classify_post_type(client: AsyncGroq, prompt: str):
    p = f"Classify this post: {prompt}. Options: Informative, Inspirational, Promotional, Tutorial. Only the word."
    try:
        r = await rate_limited_groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": p}]
        )
        return r.choices[0].message.content.strip().capitalize()
    except:
        return "Informative"
