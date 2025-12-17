# stelle_backend/services/post_generator_service.py
import asyncio
import itertools
import random
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

# ---------------------------
# PLATFORM TONE SETTINGS
# ---------------------------
PLATFORM_STYLES = {
    "instagram": "Write a clean, trendy, aesthetic caption. No Gen-Z slang. No words like lowkey, highkey, obsessed, fr, no cap.",
    "linkedin": "Write a polished professional, business-focused caption.",
    "facebook": "Write a warm, friendly and conversational caption.",
    "pinterest": "Write an aesthetic, dreamy, mood-board style caption.",
    "threads": "Write a bold, expressive caption. No slang like lowkey/highkey.",
    "tiktok": "Write an engaging, hook-first caption without slang (no lowkey/highkey/no cap).",
    "youtube": "Write a SEO-friendly YouTube description with CTA.",
    "twitter": "Write a short, punchy, bold tweet without slang.",
    "reddit": "Write an informative, discussion-starter caption."
}

# ---------------------------
# GLOBAL SLANG BAN WORD LIST
# ---------------------------
BANNED_WORDS = [
    "lowkey", "low key", "low-key",
    "highkey", "high key", "high-key",
    "obsessed", "literally", "fr", "no cap",
    "delulu"
]

TRENDING_POOLS = {

    "instagram": [
        "#fyp", "#foryou", "#foryoupage", "#fypvideo",
        "#reels", "#instareels", "#reelsinstagram", "#reelsoftheday",
        "#explore", "#explorepage", "#viral", "#trending",
        "#instadaily", "#instagood", "#creatorlife",
        "#reelsdaily", "#viralreels", "#discover",
        "#watchthis", "#shortformvideo", "#digitalcreator",
        "#reeltrend", "#socialreels"
    ],

    "tiktok": [
        "#fyp", "#foryou", "#foryoupage", "#fypvideo",
        "#viralvideo", "#tiktoktrend", "#trending",
        "#watchthis", "#discover", "#trendingsound",
        "#creatorcontent", "#videocreator",
        "#dailyvideo", "#shortvideo",
        "#tiktokcommunity", "#fypviral"
    ],

    "youtube": [
        "#shorts", "#youtubeshorts", "#viralshorts",
        "#watchnow", "#mustwatch", "#trendingnow",
        "#contentcreator", "#videocontent",
        "#discover", "#recommended",
        "#subscribe", "#newvideo"
    ],

    "threads": [
        "#threadsapp", "#threadscommunity",
        "#trendingnow", "#dailythoughts",
        "#conversationstarter", "#creatorvoices",
        "#digitalculture", "#modernlife",
        "#discoverthreads", "#threadtalk"
    ],

    "pinterest": [
        "#pinterestinspo", "#pinterestideas",
        "#aestheticinspo", "#creativeideas",
        "#moodboard", "#visualinspo",
        "#designinspiration", "#discoverideas"
    ],

    "facebook": [
        "#watchthis", "#mustsee", "#viralcontent",
        "#socialmedia", "#onlinecontent",
        "#communitypost", "#shareworthy",
        "#discovercontent", "#videooftheday"
    ],

    # ✅ ADD THIS
    "linkedin": [
        "#leadership", "#careerdevelopment",
        "#professionalgrowth", "#industryinsights",
        "#businessstrategy", "#futureofwork",
        "#innovation", "#thoughtleadership",
        "#workplaceculture", "#professionaldevelopment",
        "#careertips", "#businesscontent"
    ]
}

def rotating_hashtag_picker(pool: list, k: int = 4):
    pool = pool[:]  # copy
    random.shuffle(pool)
    cycle = itertools.cycle(pool)
    while True:
        yield [next(cycle) for _ in range(k)]


# ---------------------------
# 1) keyword generation
# ---------------------------
async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    if not client:
        try:
            fallback = await groq_generate_text(MODEL, f"Generate 3 short marketing keywords for: {query}. Return comma-separated.")
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

#hashtag

async def generate_trending_hashtags(platform: str, topic: str) -> List[str]:
    prompt = f"""
Generate 4 currently trending hashtags for {platform}.
They must feel natural, platform-native, and not generic spam.

Context:
{topic}

Rules:
- Output ONLY hashtags
- No explanations
- No generic words like viral, trending unless platform-native
"""
    try:
        text = await groq_generate_text(MODEL, prompt)
        return [t for t in text.split() if t.startswith("#")][:4]
    except:
        return []

TRENDING_GENERATORS = {
    platform: rotating_hashtag_picker(tags, 4)
    for platform, tags in TRENDING_POOLS.items()
}

async def fetch_platform_hashtags(
    client: AsyncGroq,
    seed_keywords: List[str],
    platform: str,
    query: str
) -> List[str]:

    platform = platform.lower()

    # -------------------------------
    # 1) TRENDING / DISCOVERY → 4 (ROTATING)
    # -------------------------------
    gen = TRENDING_GENERATORS.get(platform)
    trending_tags = next(gen) if gen else []

    # -------------------------------
    # 2) RELEVANT (contextual) → 3
    # -------------------------------
    try:
        relevant_prompt = f"""
Generate 3 highly relevant hashtags directly describing this content.

Content:
{query}

Rules:
- Very specific to this content
- No generic discovery tags
- Output ONLY hashtags
"""
        text = await groq_generate_text(MODEL, relevant_prompt)
        relevant_tags = [t for t in text.split() if t.startswith("#")][:3]
    except Exception:
        relevant_tags = [f"#{k.replace(' ', '')}" for k in seed_keywords][:3]

    # -------------------------------
    # 3) BROAD (category-level) → 3
    # -------------------------------
    try:
        broad_prompt = f"""
Generate 3 broad, category-level hashtags.

They should describe the general domain of the content,
not specific details.

Content:
{query}

Rules:
- High-level categories
- No trending or viral tags
- Output ONLY hashtags
"""
        text = await groq_generate_text(MODEL, broad_prompt)
        broad_tags = [t for t in text.split() if t.startswith("#")][:3]
    except Exception:
        broad_tags = []

    # -------------------------------
    # FINAL MERGE (4 + 3 + 3)
    # -------------------------------
    final_tags = trending_tags + relevant_tags + broad_tags

    # Remove duplicates, preserve order
    final_tags = list(dict.fromkeys(final_tags))

    return final_tags


# ---------------------------
# 3) caption generator
# ---------------------------
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:

    captions: Dict[str, str] = {}
    platform_hashtags: Dict[str, List[str]] = {}

    for p in platforms:
        p_norm = p.lower().strip()
        tone = PLATFORM_STYLES.get(p_norm, "Write a clean, engaging caption.")

        # Generate hashtags
        try:
            tags = await fetch_platform_hashtags(None, seed_keywords, p_norm, query)
        except Exception as e:
            logger.error(f"Hashtag generation failed for {p_norm}: {e}")
            tags = []

        platform_hashtags[p_norm] = tags

        # ---------- CAPTION PROMPT ----------
        if p_norm == "instagram":
            caption_prompt = f"""
You are a senior Instagram content strategist.

Write a caption in PARAGRAPH form with:
1) A strong, scroll-stopping hook in the FIRST sentence using contrast or a surprising statement.
2) Followed by an elaborated context that explains the situation emotionally and clearly.
3) The tone should feel relatable, observational, and human.
4) No hashtags.
5) No emojis overload.
6) No explicit CTA words like “comment”, “share”, “follow”.

Context:
{query}

The caption must feel like a thoughtful observation, not a promotion.
"""
        else:
            caption_prompt = f"""
You are a senior marketing strategist and expert social media copywriter.
Write one high-quality caption tailored for the platform: {p_norm}.

ABOUT THE POST:
- Topic: {query}
- Keywords: {', '.join([k for k in seed_keywords if k])}
- Style/Tone: {tone}

STRICT RULES:
- Output ONLY the caption text (no quotes, no bullets, no lists).
- Do NOT use slang (lowkey, highkey, fr, no cap, obsessed, literally).
- Do NOT add hashtags.
- Use natural, human-sounding language.
- Keep it concise, engaging, and platform-appropriate.
- Avoid repeating the same words.
- No personal pronouns (I, we, my, our).
- Never start with generic words like: Introducing, Presenting, Experience.
- Never wrap the caption in quotes.
"""

        # ---------- GENERATE RAW CAPTION ----------
        try:
            caption_text = await groq_generate_text(MODEL, caption_prompt)
            caption_text = caption_text.strip() if caption_text else f"A {p_norm} caption about {query}"
        except Exception as e:
            logger.error(f"Caption generation failed for {p_norm}: {e}")
            caption_text = f"A {p_norm} caption about {query}"

        # ---------- CLEANING STEPS ----------
        caption_text = caption_text.replace('\\"', '').replace('"', '').strip()

        for bad in BANNED_WORDS:
            caption_text = caption_text.replace(bad, "").replace(bad.title(), "")

        caption_text = " ".join(caption_text.split())

        captions[p_norm] = caption_text

    return {"captions": captions, "platform_hashtags": platform_hashtags}
