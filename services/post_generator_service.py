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
    "linkedin": "Write a polished professional, business-focused caption.",
    "facebook": "Write a warm, friendly and conversational caption.",
    "pinterest": "Write an aesthetic, dreamy, mood-board style caption.",
    "threads": "Write a spicy, short, gen-z hot take caption.",
    "tiktok": "Write a viral, hook-first caption optimized for engagement.",
    "youtube": "Write a SEO-friendly YouTube description with CTA.",
    "twitter": "Write a short, punchy, bold tweet.",
    "reddit": "Write an informative, discussion-starter caption."
}


# ========================
# 1. FIXED — KEYWORD GENERATION
# ========================
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    prompt = (
        f"Generate exactly 3 short marketing keywords for: {query}. "
        f"Only return comma-separated keywords. No sentences."
    )

    try:
        resp = await rate_limited_groq_call(
            client,
            model="llama-3.3-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=40,
        )

        raw = resp.choices[0].message.content.strip()
        kws = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]

        return kws[:3] if kws else ["brand", "marketing", "content"]

    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return ["brand", "marketing", "content"]


# ========================
# 2. FIXED — PLATFORM-SMART HASHTAGS (Prompt Based)
# ========================
async def fetch_platform_hashtags(client: AsyncGroq, seed_keywords: List[str], platform: str, query: str) -> List[str]:

    prompt = f"""
Generate 12 platform-specific hashtags:

Platform: {platform}
Topic: {query}
Keywords: {", ".join(seed_keywords)}

Rules:
- Only output hashtags separated by spaces
- No explanations
- Make hashtags relevant to both topic AND platform style
"""

    try:
        raw = await groq_generate_text(MODEL, prompt)
        tags = [t for t in raw.split() if t.startswith("#")] or []
    except Exception:
        tags = []

    # Clean + dedupe + limit
    tags = list(dict.fromkeys(tags))
    return tags[:12]


# ========================
# 3. FIXED — FINAL CAPTION GENERATOR
# ========================
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:

    captions = {}
    hashtags_map = {}

    for p in platforms:

        p_norm = p.lower().strip()
        tone = PLATFORM_STYLES.get(p_norm, "Write a clean engaging caption.")

        # --- generate hashtags with FULL context
        tags = await fetch_platform_hashtags(None, seed_keywords, p_norm, query)
        hashtags_map[p_norm] = tags

        # --- marketing copywriting caption prompt
        caption_prompt = f"""
You are a senior marketing strategist + expert social media copywriter.

Create a {p_norm} caption.

Topic: {query}
Keywords: {", ".join(seed_keywords)}
Tone: {tone}

Rules:
- Write ONE final caption
- No hashtags
- No long paragraphs
- Must feel platform-native and audience-perfect
"""

        try:
            caption = await groq_generate_text(MODEL, caption_prompt)
            caption = caption.strip() if caption else f"A {p_norm} caption about {query}"
        except Exception as e:
            logger.error(f"Caption generation failed for {p_norm}: {e}")
            caption = f"A {p_norm} caption about {query}"

        captions[p_norm] = caption

    return {
        "captions": captions,
        "platform_hashtags": hashtags_map
    }
