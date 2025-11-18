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

# ============================================================
#  SUPPORTED PLATFORMS (7 platforms you asked for)
# ============================================================

class Platforms(str, Enum):
    Instagram = "instagram"
    Facebook = "facebook"
    LinkedIn = "linkedin"
    Pinterest = "pinterest"
    Threads = "threads"
    TikTok = "tiktok"
    YouTube = "youtube"

    platform_list = [
        Instagram, Facebook, LinkedIn,
        Pinterest, Threads, TikTok, YouTube
    ]


# ============================================================
#  PLATFORM STYLES
# ============================================================

PLATFORM_STYLES = {
    "instagram": "Write a sassy, trendy, aesthetic Gen-Z caption.",
    "facebook": "Write a warm, friendly, conversational caption.",
    "linkedin": "Write a professional, corporate, polished caption.",
    "pinterest": "Write an aesthetic, dreamy, soft-vibe caption.",
    "threads": "Write a spicy, short Gen-Z style caption.",
    "tiktok": "Write a chaotic, hook-first, Gen-Z engaging caption.",
    "youtube": "Write a YouTube description-style caption with SEO tone."
}


# ============================================================
#  KEYWORDS
# ============================================================

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


# ============================================================
#  PLATFORM-WISE HASHTAGS  (MOST IMPORTANT PART)
# ============================================================

async def fetch_platform_hashtags(
    client: AsyncGroq,
    seed_keywords: list,
    platform: str
) -> List[str]:

    prompt = f"""
Generate 15 trending hashtags for the platform: {platform}.
Base them on keywords: {", ".join(seed_keywords)}.
Rules:
- ONLY output hashtags (space-separated)
- No sentences or explanations
"""

    try:
        raw = await query_internet_via_groq(prompt)
        cleaned = [t.replace("#", "").strip() for t in raw.split() if t.strip()]
    except:
        cleaned = []

    cleaned = list(dict.fromkeys(cleaned))
    return [f"#{t}" for t in cleaned[:15]]


# ============================================================
#  GLOBAL HASHTAGS (old one, still used by chat)
# ============================================================

async def fetch_trending_hashtags_post(client: AsyncGroq, seed_keywords, _platforms):
    tags = []
    for kw in seed_keywords:
        p = f"Give trending hashtags for: {kw}. Only hashtags separated by spaces."
        try:
            raw = await query_internet_via_groq(p)
            tags.extend([t.replace("#", "") for t in raw.split()])
        except:
            pass
    unique = list(dict.fromkeys(tags))
    return [f"#{t}" for t in unique[:15]]


# ============================================================
#  SEO
# ============================================================

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


# ============================================================
#  FINAL MULTI-PLATFORM CAPTION + HASHTAGS GENERATOR
# ============================================================

async def generate_caption_post(
    query: str,
    seed_keywords: list,
    platforms: list
) -> Dict[str, Any]:

    captions: Dict[str, str] = {}
    platform_hashtags: Dict[str, List[str]] = {}

    # Create async Groq instance for hashtag calls
    client = AsyncGroq()

    for platform in platforms:
        p = platform.lower()

        # ---------------------------------------
        # Caption generation
        # ---------------------------------------
        style = PLATFORM_STYLES.get(p, "Write a clean, creative caption.")

        caption_prompt = f"""
Generate a caption for: {p}

Context: {query}

Style: {style}

Rules:
- Do NOT include hashtags.
- Only return ONE clean caption.
"""

        try:
            caption = await groq_generate_text(MODEL, caption_prompt)
        except Exception as e:
            logger.error(f"Caption generation failed for {p}: {e}")
            caption = query

        captions[p] = caption.strip()

        # ---------------------------------------
        # Platform-wise hashtags
        # ---------------------------------------
        try:
            platform_hashtags[p] = await fetch_platform_hashtags(
                client=client,
                seed_keywords=seed_keywords,
                platform=p
            )
        except:
            platform_hashtags[p] = []

    return {
        "captions": captions,
        "platform_hashtags": platform_hashtags
    }


# ============================================================
#  CLASSIFIER
# ============================================================

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
