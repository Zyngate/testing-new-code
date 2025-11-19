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

# ✅ Updated model
MODEL = "llama-3.3-70b-versatile"


# Supported platforms
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


# -------------------------------------------------------
# 1. Generate seed keywords
# -------------------------------------------------------
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    prompt = (
        f"Generate exactly 3 short marketing keywords based on: {query}. "
        f"Return ONLY 3 keywords, comma-separated."
    )
    try:
        resp = await rate_limited_groq_call(
            client,
            model="llama-3.3-8b-instant",   # ✅ Updated model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=30,
        )
        text = resp.choices[0].message.content or ""
        keywords = [k.strip() for k in text.replace("\n", ",").split(",") if k.strip()]
        return keywords[:3] if keywords else ["", "", ""]

    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return ["", "", ""]


# -------------------------------------------------------
# 2. Platform-based hashtag generator
# -------------------------------------------------------
async def fetch_platform_hashtags(client: AsyncGroq, seed_keywords: List[str], platform: str) -> List[str]:
    base_tags = []
    for kw in seed_keywords:
        if kw:
            clean = "".join(ch for ch in kw if ch.isalnum() or ch == "_")
            if clean:
                base_tags.append(f"#{clean.lower()}")

    platform_style_tags = {
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

    extra_tags = platform_style_tags.get(platform.lower(), ["#socialmedia"])
    final = list(dict.fromkeys(base_tags + extra_tags))
    return final[:10]


# -------------------------------------------------------
# 3. Caption Generator
# -------------------------------------------------------
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:
    captions = {}
    hashtags_map = {}

    for p in platforms:
        p_norm = p.lower().strip()
        style = PLATFORM_STYLES.get(p_norm, "Write a creative social media caption.")

        tags = await fetch_platform_hashtags(None, seed_keywords, p_norm)
        hashtags_map[p_norm] = tags

        prompt = f"""
You are a senior marketing copywriter.
Context: {query}
Keywords: {', '.join(seed_keywords)}

Platform: {p_norm}
Tone: {style}

Write ONE final caption.
DO NOT include hashtags.
Use platform-appropriate tone.
"""

        try:
            caption = await groq_generate_text(MODEL, prompt)

            # ✅ fallback so caption never becomes empty
            if not caption:
                caption = f"A short {p_norm} caption about: {query}"

        except Exception as e:
            logger.error(f"Caption generation error for {p_norm}: {e}")
            caption = f"A short {p_norm} caption about: {query}"

        captions[p_norm] = caption.strip()

    return {"captions": captions, "platform_hashtags": hashtags_map}
