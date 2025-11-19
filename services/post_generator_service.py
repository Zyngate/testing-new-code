# stelle_backend/services/post_generator_service.py

import asyncio
from typing import List, Dict, Any
from groq import AsyncGroq
from enum import Enum
from config import logger
from services.ai_service import (
    rate_limited_groq_call,
    groq_generate_text
)

# ========================
#  MODEL
# ========================
MODEL = "llama-3.3-70b-versatile"


# ========================
#  SUPPORTED PLATFORMS
# ========================
class Platforms(str, Enum):
    Instagram = "instagram"
    Facebook = "facebook"
    LinkedIn = "linkedin"
    Pinterest = "pinterest"
    Threads = "threads"
    TikTok = "tiktok"
    YouTube = "youtube"
    Twitter = "twitter"
    Reddit = "reddit"


# ========================
#  PLATFORM STYLES
# ========================
PLATFORM_STYLES = {
    "instagram": "Write a sassy, trendy, Gen-Z caption. Short, aesthetic & bold.",
    "linkedin": "Write a polished, professional, corporate-friendly caption.",
    "facebook": "Write a warm, friendly and conversational caption.",
    "pinterest": "Write an aesthetic, dreamy, soft-vibes caption.",
    "threads": "Write a short, spicy, gen-z hot take caption.",
    "tiktok": "Write a chaotic, hook-first caption focused on engagement.",
    "youtube": "Write a SEO-friendly YouTube description with a CTA.",
    "twitter": "Write a short, punchy tweet.",
    "reddit": "Write an informative, discussion-starter style caption."
}



# ========================
# 1. KEYWORD GENERATION
# ========================
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    prompt = (
        f"Generate exactly 3 short marketing keywords based on: {query}. "
        f"Return ONLY 3 keywords, comma-separated."
    )

    try:
        resp = await rate_limited_groq_call(
            client,
            model="llama-3.3-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=30,
        )

        text = resp.choices[0].message.content or ""
        kws = [k.strip() for k in text.replace("\n", ",").split(",") if k.strip()]
        return kws[:3] if kws else ["", "", ""]

    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return ["", "", ""]



# ========================
# 2. PLATFORM-WISE HASHTAGS
# ========================
async def fetch_platform_hashtags(client: AsyncGroq, seed_keywords: List[str], platform: str) -> List[str]:

    # keyword-based tags
    base_tags = []
    for kw in seed_keywords:
        if kw:
            clean = "".join(ch for ch in kw if ch.isalnum())
            if clean:
                base_tags.append(f"#{clean.lower()}")

    # platform personality tags
    platform_extra_tags = {
        "instagram": ["#trending", "#reels", "#instadaily", "#aesthetic"],
        "facebook": ["#community", "#friends", "#share"],
        "linkedin": ["#business", "#growth", "#success", "#innovation"],
        "pinterest": ["#inspo", "#moodboard", "#aesthetic"],
        "threads": ["#genz", "#threadsapp", "#hottake"],
        "tiktok": ["#fyp", "#viral", "#tiktoktrend"],
        "youtube": ["#subscribe", "#creators", "#newvideo"],
        "twitter": ["#update", "#trending", "#tweet"],
        "reddit": ["#discussion", "#insights", "#community"]
    }

    extra = platform_extra_tags.get(platform.lower(), ["#socialmedia"])

    final = list(dict.fromkeys(base_tags + extra))
    return final[:10]



# ========================
# 3. CAPTION GENERATOR
# ========================
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:

    captions = {}
    hashtags_map = {}

    for platform in platforms:

        p = platform.lower().strip()
        style = PLATFORM_STYLES.get(p, "Write a creative social media caption.")

        # --- hashtags
        tags = await fetch_platform_hashtags(None, seed_keywords, p)
        hashtags_map[p] = tags

        # --- marketing expert caption prompt
        prompt = f"""
You are a senior marketing strategist and expert copywriter.

Context: {query}
Keywords: {', '.join(seed_keywords)}

Platform: {p}
Tone: {style}

Write ONE final caption.
Avoid hashtags.
Make it platform-perfect.
"""

        try:
            caption = await groq_generate_text(MODEL, prompt)
            if not caption:
                caption = f"A short {p} caption about {query}"
        except Exception as e:
            logger.error(f"Caption generation error for {p}: {e}")
            caption = f"A short {p} caption about {query}"

        captions[p] = caption.strip()

    return {
        "captions": captions,
        "platform_hashtags": hashtags_map
    }
