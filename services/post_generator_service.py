# stelle_backend/services/post_generator_service.py
import asyncio
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

# Supported platforms (normalized keys)
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

    platform_list = [Instagram, Facebook, LinkedIn, Pinterest, Threads, TikTok, YouTube, Twitter, Reddit]


PLATFORM_STYLES = {
    "instagram": "Write a sassy, trendy, Gen-Z caption. Short, aesthetic & bold.",
    "linkedin": "Write a polished, professional, corporate-friendly caption (suitable for senior leadership announcements).",
    "facebook": "Write a warm, friendly and conversational caption.",
    "pinterest": "Write an aesthetic, dreamy, soft-vibes caption.",
    "threads": "Write a short, spicy, gen-z hot take caption.",
    "tiktok": "Write a chaotic, hook-first caption focused on engagement and a call-to-action.",
    "youtube": "Write a YouTube description style caption with an SEO-friendly first line and a CTA.",
    "twitter": "Write a short, punchy, high-impact tweet.",
    "reddit": "Write an informative, discussion-starter style caption suitable for a subreddit audience."
}

# -------------------------
# Seed keywords (unchanged idea)
# -------------------------
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    prompt = (
        f"Generate exactly 3 short seed keywords for marketing/social media based on: {query}. "
        "Return ONLY 3 keywords, comma-separated."
    )
    try:
        resp = await rate_limited_groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=50,
        )
        text = resp.choices[0].message.content
        keywords = [k.strip() for k in text.replace("\n", ",").split(",") if k.strip()]
        return (keywords + ["", "", ""])[:3]
    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return ["", "", ""]

# -------------------------
# Platform-specific hashtags (UPDATED – NO INTERNET)
# -------------------------
async def fetch_platform_hashtags(client: AsyncGroq, seed_keywords: List[str], platform: str) -> List[str]:
    """
    Generate platform-aware hashtags from seed keywords.
    No internet calls. Tone-specific hashtag styles.
    """

    # Base keyword hashtags (#keyword -> cleaned)
    base_tags = []
    for kw in seed_keywords:
        if kw:
            clean = "".join(ch for ch in kw if ch.isalnum() or ch == "_")
            if clean:
                base_tags.append(f"#{clean.lower()}")

    # Platform-specific extra hashtags
    platform_style_tags = {
        "instagram": ["#trending", "#reels", "#instadaily", "#aesthetic", "#vibes"],
        "facebook": ["#community", "#share", "#friends", "#connect"],
        "linkedin": ["#business", "#leadership", "#success", "#growth", "#innovation"],
        "pinterest": ["#inspo", "#aesthetic", "#moodboard", "#creative"],
        "threads": ["#trendingnow", "#hottake", "#genz", "#threadsapp"],
        "tiktok": ["#fyp", "#viral", "#tiktoktrend", "#creators", "#foryou"],
        "youtube": ["#youtubers", "#creators", "#subscribe", "#newvideo"],
        "twitter": ["#trending", "#update", "#breaking", "#tweet"],
        "reddit": ["#discussion", "#insights", "#redditpost", "#community"]
    }

    extra_tags = platform_style_tags.get(platform.lower(), ["#socialmedia"])

    # Combine & clean duplicates
    final_tags = list(dict.fromkeys(base_tags + extra_tags))

    # Return max 10 hashtags
    return final_tags[:10]

# -------------------------
# SEO keywords (small helper)
# -------------------------
async def fetch_seo_keywords_post(client: AsyncGroq, seed_keywords: List[str]) -> List[str]:
    result = []
    for kw in seed_keywords:
        if not kw:
            continue
        prompt = f"List top SEO keyword phrases (comma separated) related to: {kw}. Provide up to 5 concise phrases."
        try:
            resp = await query_internet_via_groq(prompt)
            parts = [p.strip() for p in resp.replace("\n", ",").split(",") if p.strip()]
            result.extend(parts[:5])
        except Exception:
            continue
    return list(dict.fromkeys(result))[:10]

# -------------------------
# Marketing-expert, platform-aware caption generator
# -------------------------
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:
    """
    Generates one caption per platform in `platforms` list.
    Returns {"captions": {platform: caption}, "platform_hashtags": {platform: [#tags]}}
    The model is instructed to act as a senior marketing copywriter.
    """
    captions = {}
    platform_hashtags_map = {}

    # Prefer using model via groq_generate_text for captions; but hashtags use web query
    for p in platforms:
        p_norm = str(p).lower().strip()
        style = PLATFORM_STYLES.get(p_norm, "Write a clean, creative caption suitable for social media.")
        # generate platform hashtags (async)
        try:
            # pass a real AsyncGroq client if available; query_internet_via_groq uses global internet client so client param is optional in that function.
            tags = await fetch_platform_hashtags(None, seed_keywords, p_norm)
        except Exception as e:
            logger.error(f"Hashtag generation error for {p_norm}: {e}")
            tags = []
        platform_hashtags_map[p_norm] = tags

        # Marketing expert system role + clear rules
        prompt = f"""
You are a senior marketing copywriter with 8+ years of experience writing captions for social media.
User context: {query}
Seed keywords: {', '.join([k for k in seed_keywords if k])}

Platform: {p_norm}
Tone/style instruction: {style}

Write ONE single final caption that:
- Is optimized for engagement on the given platform.
- Includes a short CTA (call to action) if appropriate (follow/visit/book/ticket).
- DOES NOT include hashtags (we will return hashtags separately).
- DOES NOT repeat the seed keywords verbatim.
- Keep it concise (one to three lines for Instagram/Twitter; longer allowed for LinkedIn/YouTube).
Return ONLY the caption text.
"""
        try:
            caption = await groq_generate_text(MODEL, prompt)
        except Exception as e:
            logger.error(f"Caption generation failed for {p_norm}: {e}")
            caption = query  # fallback to the raw query

        captions[p_norm] = caption.strip()

    return {"captions": captions, "platform_hashtags": platform_hashtags_map}
