# stelle_backend/services/post_generator_service.py

import asyncio
import json
import random
from typing import List, Dict, Any
from groq import AsyncGroq
from enum import Enum
import os

from config import logger, INTERNET_CLIENT_KEY, ASYNC_CLIENT_KEY
from services.ai_service import (
    rate_limited_groq_call,
    query_internet_via_groq,
    groq_generate_text
)
from services.common_utils import get_current_datetime

# -------------------------
# PLATFORM ENUM
# -------------------------
class Platforms(str, Enum):
    Instagram = "Instagram"
    X = "X (Twitter)"
    Reddit = "Reddit"
    LinkedIn = "LinkedIn"
    Facebook = "Facebook"

    platform_list = [Instagram, X, Reddit, LinkedIn, Facebook]

# -------------------------
# NEW CAPTION MODEL CONFIG
# -------------------------
MODEL = "llama-3.3-70b-versatile"

PLATFORM_STYLES = {
    "instagram": "Write a sassy, trendy, Gen-Z styled caption. Use short energetic lines.",
    "linkedin": "Write a polished, professional LinkedIn caption suitable for corporate audience.",
    "facebook": "Write a friendly, conversational caption.",
    "pinterest": "Write an aesthetic, dreamy, soft-vibes caption.",
    "threads": "Write a short, spicy, gen-z hot take caption.",
    "tiktok": "Write a chaotic, hook-first, gen-z styled caption.",
    "youtube": "Write a YouTube video description styled caption with SEO-friendly tone.",
    "twitter": "Write a punchy, short, high-impact tweet."
}

# -------------------------
# LLM FUNCTIONS
# -------------------------

async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    prompt = (
        f"Generate 3 seed keywords based on the following content description: {query}. "
        "Separate the keywords with commas. Output only keywords."
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_completion_tokens=100,
        )
        response = completion.choices[0].message.content
        seed_keywords = [kw.strip() for kw in response.split(",") if kw.strip()][:3]
    except Exception as e:
        logger.error(f"Error generating seed keywords: {e}")
        seed_keywords = []

    if len(seed_keywords) != 3:
        logger.warning(f"Adjusting seed keywords: {seed_keywords}")
        seed_keywords = (seed_keywords + [""] * 3)[:3]

    return seed_keywords


async def fetch_trending_hashtags_post(client: AsyncGroq, seed_keywords: list, platforms: list[Platforms]) -> list:
    hashtags = []
    platform_names = ", ".join([p.value for p in platforms])

    for keyword in seed_keywords:
        prompt = (
            f"Browse {platform_names}, and fetch trending hashtags related to {keyword}. "
            "Ensure they are trending and can boost SEO. Provide up to 10 unique hashtags separated by spaces. ONLY provide hashtags."
        )
        try:
            response_content = await query_internet_via_groq(prompt)
        except Exception as e:
            logger.error(f"Error fetching trending hashtags: {e}")
            response_content = ""

        keyword_hashtags = [
            ht.strip().replace("#", "")
            for ht in response_content.split()
            if ht.strip()
        ]
        hashtags.extend(keyword_hashtags)

    unique_hashtags = list(dict.fromkeys(hashtags))
    return [f"#{ht}" for ht in unique_hashtags[:30]]


async def fetch_seo_keywords_post(client: AsyncGroq, seed_keywords: list) -> list:
    seo_keywords = []

    for keyword in seed_keywords:
        prompt = (
            f"Find the top 5 most searched SEO keywords related to {keyword}. "
            "Only provide keywords separated by commas."
        )
        try:
            response_content = await query_internet_via_groq(prompt)
        except Exception as e:
            logger.error(f"Error fetching SEO keywords: {e}")
            response_content = ""

        keyword_seo = [kw.strip() for kw in response_content.split(",") if kw.strip()]
        seo_keywords.extend(keyword_seo[:5])

    return list(dict.fromkeys(seo_keywords))[:15]


# =============================================================
# 🔥 NEW CAPTION GENERATOR
# =============================================================
# =============================================================
# 🔥 CLEAN + POLISHED CAPTION GENERATOR
# =============================================================

async def generate_caption_post(query: str, seed_keywords: list, hashtags: list, platforms: list) -> Dict[str, Any]:
    final_output = {}

    # reduce hashtags to 5 best ones
    clean_hashtags = []
    for h in hashtags:
        if len(clean_hashtags) >= 5:
            break
        if len(h) > 2 and h not in clean_hashtags:
            clean_hashtags.append(h)

    if not clean_hashtags:
        clean_hashtags = ["#Trending", "#ExploreMore", "#Inspiration"]

    hashtag_block = " ".join(clean_hashtags)

    for platform in platforms:
        platform_normalized = platform.lower().strip()
        style = PLATFORM_STYLES.get(platform_normalized, "Write a simple creative caption.")

        prompt = f"""
Write a clean, final social-media caption based on this context:

➡️ Context / event: "{query}"

Platform: {platform}
Writing Style: {style}

Rules:
- The caption MUST be short, catchy and polished.
- Add ONE clear call-to-action: “Follow us for more updates!”
- Do NOT list keywords.
- Do NOT include explanations.
- Do NOT repeat the full context again.
- Return ONLY the final caption text.
- After the caption, add this hashtag block exactly:
{hashtag_block}
"""

        try:
            caption = await groq_generate_text(MODEL, prompt)
        except Exception as e:
            logger.error(f"Error generating caption for {platform}: {e}")
            caption = query  # fallback

        # final cleaning
        final_caption = caption.strip().replace("\n", " ")

        final_output[platform] = final_caption

    return {
        "captions": final_output,
        "hashtags": clean_hashtags
    }


# =============================================================
# CLASSIFIER
# =============================================================

async def classify_post_type(client: AsyncGroq, prompt: str) -> str:
    classification_prompt = (
        f"Analyze the user's content prompt: '{prompt}'. "
        "Classify the intended post type into: 'Informative', 'Inspirational', 'Promotional', or 'Tutorial'. "
        "Return ONLY the word."
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.3,
            max_completion_tokens=10,
        )
        return completion.choices[0].message.content.strip().capitalize()
    except Exception as e:
        logger.error(f"Error classifying post type: {e}")
        return "Informative"


# =============================================================
# HTML POST GENERATOR
# =============================================================

async def generate_html_code_post(client: AsyncGroq, prompt: str, post_type: str) -> str:
    html_prompt = (
        f"Generate a single-page HTML post about '{prompt}' (Type: {post_type}). "
        "Use modern CSS with a dark theme. Output ONLY HTML starting with <!DOCTYPE html>."
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": html_prompt}],
            temperature=0.8,
            max_completion_tokens=3000,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating HTML: {e}")
        return 
