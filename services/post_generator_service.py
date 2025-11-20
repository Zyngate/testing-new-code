# stelle_backend/services/post_generator_service.py
import asyncio
from typing import List, Dict, Any
from groq import AsyncGroq
from enum import Enum
from config import logger
from services.ai_service import (
    rate_limited_groq_call,
    groq_generate_text,
)

MODEL = "llama-3.3-70b-versatile"

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
    "linkedin": "Write a polished professional, business-focused caption.",
    "facebook": "Write a warm, friendly and conversational caption.",
    "pinterest": "Write an aesthetic, dreamy, mood-board style caption.",
    "threads": "Write a spicy, short, gen-z hot take caption.",
    "tiktok": "Write a viral, hook-first caption optimized for engagement.",
    "youtube": "Write a SEO-friendly YouTube description with CTA.",
    "twitter": "Write a short, punchy, bold tweet.",
    "reddit": "Write an informative, discussion-starter caption."
}

# ---------------------------
# 1) keyword generation
# ---------------------------
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    """
    Uses the provided client (caption client) to generate 1-3 marketing keywords.
    Returns non-empty fallbacks if model fails.
    """
    if not client:
        # defensive: try to call groq_generate_text as fallback
        try:
            fallback = await groq_generate_text(MODEL, f"Generate 3 short marketing keywords for: {query}. Return them comma-separated, no explanation.")
            kws = [k.strip() for k in fallback.replace("\n", ",").split(",") if k.strip()]
            return kws[:3] if kws else ["brand", "marketing", "content"]
        except:
            return ["brand", "marketing", "content"]

    prompt = f"Generate exactly 3 short marketing keywords for: {query}. Only return 3 comma-separated keywords."
    try:
        resp = await rate_limited_groq_call(
            client,
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=40,
        )
        raw = resp.choices[0].message.content or ""
        kws = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
        return kws[:3] if kws else ["brand", "marketing", "content"]
    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return ["brand", "marketing", "content"]

# ---------------------------
# 2) platform hashtag generator (prompt-based + common tags)
# ---------------------------
async def fetch_platform_hashtags(client: AsyncGroq, seed_keywords: List[str], platform: str, query: str) -> List[str]:

    # Common platform hashtags
    COMMON_TAGS = {
        "instagram": [
            "#reels", "#reelitfeelit", "#viral", "#trending", 
            "#explorepage", "#instadaily", "#genzvibes"
        ],
        "tiktok": [
            "#fyp", "#foryou", "#tiktokviral", "#tiktoktrend", 
            "#viralvideo", "#trendingsound"
        ],
        "youtube": [
            "#shorts", "#youtubeshorts", "#viralshorts", "#subscribe", 
            "#creatorlife", "#youtubetrending"
        ],
        "linkedin": [
            "#leadership", "#careerdevelopment", "#professionalnetworking", 
            "#businessstrategy", "#innovation", "#futureofwork"
        ],
        "facebook": [
            "#community", "#friendsandfamily", "#fblife", "#socialvibes"
        ],
        "threads": [
            "#threadsapp", "#genz", "#hotTake", "#trendingNow"
        ],
        "pinterest": [
            "#aesthetic", "#moodboard", "#creativeinspo", "#pinterestideas"
        ],
        "twitter": [
            "#trending", "#viralpost", "#tweetoftheday", "#newpost"
        ],
        "reddit": [
            "#askreddit", "#discussion", "#redditcommunity"
        ]
    }

    platform_context = {
        "instagram": "trendy, aesthetic, gen-z",
        "facebook": "friendly, community vibes",
        "linkedin": "professional, business, corporate",
        "pinterest": "aesthetic, dreamy, creative",
        "threads": "spicy, short, gen-z hot takes",
        "tiktok": "viral, fyp, fast hooks",
        "youtube": "creators, long-form, SEO",
        "twitter": "short, bold, punchy",
        "reddit": "discussion, informative, community"
    }.get(platform.lower(), "general social media")

    # Prompt to AI
    prompt = f"""
Generate 12 platform-specific hashtags.

Platform: {platform}
Platform style: {platform_context}

Topic: {query}
Keywords: {', '.join(seed_keywords)}

Rules:
- Only output hashtags separated by spaces.
- No explanations.
"""

    try:
        text = await groq_generate_text(MODEL, prompt)
        ai_tags = [t for t in text.split() if t.startswith("#")]
    except Exception:
        ai_tags = []

    # Fallback if no hashtags generated
    if not ai_tags:
        ai_tags = [f"#{k.lower()}" for k in seed_keywords if k]

    # Add common platform hashtags
    common = COMMON_TAGS.get(platform.lower(), [])
    final_tags = ai_tags + common

    # Dedupe + limit 15 tags
    final_tags = list(dict.fromkeys(final_tags))[:15]

    return final_tags


# ---------------------------
# 3) caption generator (per-platform)
# ---------------------------
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:
    captions: Dict[str, str] = {}
    platform_hashtags: Dict[str, List[str]] = {}

    # If caller didn't pass a client, generate_keywords_post will attempt to call groq_generate_text and fallback
    # But routes should pass the client from get_groq_client

    # We expect platforms to be list of keys like 'instagram', 'tiktok', etc.
    for p in platforms:
        p_norm = p.lower().strip()
        tone = PLATFORM_STYLES.get(p_norm, "Write a clean, engaging caption.")

        # hashtags (pass query)
        try:
            # We intentionally call groq_generate_text (so it uses caption key) inside fetch_platform_hashtags
            tags = await fetch_platform_hashtags(None, seed_keywords, p_norm, query)
        except Exception as e:
            logger.error(f"Hashtag generation failed for {p_norm}: {e}")
            tags = []

        platform_hashtags[p_norm] = tags

        # caption prompt for marketing expert
        caption_prompt = f"""
You are a senior marketing strategist and expert social media copywriter.
Write a single caption for platform: {p_norm}.

Topic: {query}
Keywords: {', '.join([k for k in seed_keywords if k])}
Tone: {tone}

Rules:
- Write ONE final caption.
- Do NOT include hashtags.
- Keep it platform-appropriate and short.
"""
        try:
            caption_text = await groq_generate_text(MODEL, caption_prompt)
            caption_text = caption_text.strip() if caption_text else f"A {p_norm} caption about {query}"
        except Exception as e:
            logger.error(f"Caption generation error for {p_norm}: {e}")
            caption_text = f"A {p_norm} caption about {query}"

        captions[p_norm] = caption_text

    return {"captions": captions, "platform_hashtags": platform_hashtags}
