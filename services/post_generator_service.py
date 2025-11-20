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
#  GROQ WORKING MODELS
# ========================
MODEL_MAIN = "llama-3.1-70b-versatile"
MODEL_KEYWORDS = "llama-3.1-8b-instant"


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
#  PLATFORM TONE PROFILES
# ========================
PLATFORM_STYLES = {
    "instagram": "Write a sassy, trendy, aesthetic Gen-Z style caption.",
    "linkedin": "Write a polished professional corporate caption.",
    "facebook": "Write a friendly warm conversational caption.",
    "pinterest": "Write a dreamy aesthetic mood-board caption.",
    "threads": "Write a spicy Gen-Z short take.",
    "tiktok": "Write a viral hook-first caption.",
    "youtube": "Write a SEO-optimized description with CTA.",
    "twitter": "Write a short punchy tweet.",
    "reddit": "Write an informative discussion-starter caption."
}


# ========================
# 1. FIXED KEYWORD GENERATOR
# ========================
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    prompt = (
        f"Generate exactly 3 short marketing keywords for: {query}.\n"
        f"Return ONLY comma-separated keywords. No sentences."
    )
    try:
        resp = await rate_limited_groq_call(
            client,
            model=MODEL_KEYWORDS,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=20
        )

        raw = resp.choices[0].message.content.strip()
        kws = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
        return kws[:3] if kws else ["marketing", "brand", "content"]

    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return ["marketing", "brand", "content"]


# ========================
# 2. FIXED HASHTAG GENERATOR
# ========================
async def fetch_platform_hashtags(client: AsyncGroq, seed_keywords: List[str], platform: str, query: str) -> List[str]:

    prompt = f"""
Generate 12 high-quality hashtags.
Platform: {platform}
Topic: {query}
Keywords: {", ".join(seed_keywords)}

Rules:
- Only output hashtags separated by spaces
- No sentences
- Make hashtags platform-specific + trending + related to topic
    """

    try:
        raw = await groq_generate_text(MODEL_MAIN, prompt)
        tags = [t for t in raw.split() if t.startswith("#")]
    except Exception as e:
        logger.error(f"Hashtag generation failed: {e}")
        tags = []

    tags = list(dict.fromkeys(tags))  # dedupe
    return tags[:12]


# ========================
# 3. FIXED CAPTION GENERATOR
# ========================
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:

    captions = {}
    hashtags_map = {}

    for p in platforms:
        p_norm = p.lower().strip()
        style = PLATFORM_STYLES.get(p_norm, "Write an engaging caption.")

        # FIXED — pass query argument
        tags = await fetch_platform_hashtags(None, seed_keywords, p_norm, query)
        hashtags_map[p_norm] = tags

        caption_prompt = f"""
You are an expert marketing copywriter.
Write a single {p_norm} caption.

Topic: {query}
Keywords: {', '.join(seed_keywords)}
Tone: {style}

Rules:
- One short caption only
- No hashtags
- Make it platform-native
"""

        try:
            caption = await groq_generate_text(MODEL_MAIN, caption_prompt)
            if not caption:
                caption = f"A {p_norm} caption about {query}"
        except Exception as e:
            logger.error(f"Caption error for {p_norm}: {e}")
            caption = f"A {p_norm} caption about {query}"

        captions[p_norm] = caption.strip()

    return {
        "captions": captions,
        "platform_hashtags": hashtags_map
    }
