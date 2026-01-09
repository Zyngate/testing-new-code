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

INSTAGRAM_DISCOVERY_CORE = [
    "#fyp", "#explore", "#viral", "#foryou"
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
    
    if platform == "instagram":
        discovery_tags = random.sample(
            INSTAGRAM_DISCOVERY_CORE,
            k=min(2, len(INSTAGRAM_DISCOVERY_CORE))
        )
        remaining_pool = [
            t for t in TRENDING_POOLS["instagram"]
            if t not in INSTAGRAM_DISCOVERY_CORE
        ]
        secondary_trending = random.sample(
            remaining_pool,
            k=min(2, len(remaining_pool))
        )
        trending_tags = discovery_tags + secondary_trending

    else:
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

def enforce_instagram_constraints(text: str, max_chars: int = 1000) -> str:
    """
    Ensures:
    - Max 1000 characters (including spaces)
    - No sentence cut-off
    - Exactly 3 paragraphs
    """

    # Normalize paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Force exactly 3 paragraphs
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:3]
    elif len(paragraphs) < 3:
        while len(paragraphs) < 3:
            paragraphs.append("")

    text = "\n\n".join(paragraphs)

    # If within limit, return
    if len(text) <= max_chars:
        return text

    # Trim safely at sentence boundary
    trimmed = text[:max_chars]

    last_punct = max(
        trimmed.rfind("."),
        trimmed.rfind("?"),
        trimmed.rfind("!")
    )

    if last_punct != -1:
        trimmed = trimmed[:last_punct + 1]

    return trimmed.strip()


# ---------------------------
# 3) caption generator
# ---------------------------
async def generate_caption_post(query: str, seed_keywords: List[str], platforms: List[str]) -> Dict[str, Any]:

    captions: Dict[str, str] = {}
    platform_hashtags: Dict[str, List[str]] = {}

    for p in platforms:
        p_norm = p.lower().strip()
        tone = PLATFORM_STYLES.get(p_norm, "Write a clean, engaging caption.")

        # ---------------------------
        # Hashtags
        # ---------------------------
        try:
            tags = await fetch_platform_hashtags(None, seed_keywords, p_norm, query)
        except Exception as e:
            logger.error(f"Hashtag generation failed for {p_norm}: {e}")
            tags = []

        platform_hashtags[p_norm] = tags

        # ---------------------------
        # Caption prompt
        # ---------------------------
        if p_norm == "instagram":
            caption_prompt = f"""
You are writing an Instagram caption.

STRICT HARD RULES (NON-NEGOTIABLE):
- Write EXACTLY 3 paragraphs.
- The total length should be close to 1000 characters but MUST NOT cut sentences.
- End sentences properly.
- Write EXACTLY 3 paragraphs.
- Keep the total length within 1000 characters including spaces.
- Do NOT cut sentences. End all sentences properly.
- Each paragraph separated by ONE blank line.
- No hashtags.
- No emojis.
- No explicit CTAs (comment, share, follow).
- No first-person words (I, we, my, our, us).
- Do NOT say "an individual", "a person", or vague identifiers.
- If a known public figure is present, mention their name naturally.

STRUCTURE (MANDATORY):
Paragraph 1 — HOOK:
- Eye-catching and curiosity-driven.
- Immediately pulls attention.

Paragraph 2 — CONTEXT:
- Explains what is happening.
- Clear and factual.

Paragraph 3 — INSIGHT:
- Reflective takeaway.
- Calm and confident.

CONTENT CONTEXT:
{query}

Return ONLY the caption text.
"""
        else:
            caption_prompt = f"""
You are a senior marketing strategist.
Write one caption for {p_norm}.

Topic: {query}
Keywords: {', '.join(seed_keywords)}
Tone: {tone}

Rules:
- No hashtags
- No slang
- No first-person language
- Output ONLY caption text
"""

        # ---------------------------
        # Generate caption
        # ---------------------------
        try:
            caption_text = await groq_generate_text(MODEL, caption_prompt)
            caption_text = caption_text.strip() if caption_text else ""
        except Exception as e:
            logger.error(f"Caption generation failed for {p_norm}: {e}")
            caption_text = ""

        # ---------------------------
        # Cleaning (SAFE)
        # ---------------------------
        caption_text = caption_text.replace('\\"', '').replace('"', '').strip()

        for bad in BANNED_WORDS:
            caption_text = caption_text.replace(bad, "").replace(bad.title(), "")

        # Preserve paragraphs
        caption_text = "\n\n".join(
            [" ".join(p.split()) for p in caption_text.split("\n\n") if p.strip()]
        )

        # ---------------------------
        # Instagram validation
        # ---------------------------
        if p_norm == "instagram":
            caption_text = enforce_instagram_constraints(caption_text, 1000)

            paragraphs = [p for p in caption_text.split("\n\n") if p.strip()]
            if len(paragraphs) != 3 or len(caption_text) > 1000:
                logger.warning(
                    f"Instagram caption still failed constraints "
                    f"(paragraphs={len(paragraphs)}, chars={len(caption_text)})"
            )

                # Optional: regenerate once (not mandatory now)

        captions[p_norm] = caption_text

    return {
        "captions": captions,
        "platform_hashtags": platform_hashtags
    }