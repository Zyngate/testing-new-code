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


# ---------------------------
# 2) platform hashtag generator
# ---------------------------
async def fetch_platform_hashtags(client: AsyncGroq, seed_keywords: List[str], platform: str, query: str) -> List[str]:

    COMMON_TAGS = {
        "instagram": ["#reels", "#reelitfeelit", "#viral", "#trending", "#explorepage", "#instadaily",
                      "#ExplorePage", "#TrendingNow", "#InstaVibes", "#ReelsDaily", "#ViralReels", "#InstaGood"],
        
        "tiktok": ["#fyp", "#foryou", "#tiktokviral", "#tiktoktrend", "#viralvideo",
                   "#foryoupage", "#tiktokmademedoit", "#watchtiltheend", "#creatorspotlight", "#viralmoments", "#trendingsounds"],
        
        "youtube": ["#shorts", "#youtubeshorts", "#viralshorts", "#subscribe", "#creatorlife",
                    "#ContentCreator", "#SubscribeNow"],
        
        "linkedin": ["#leadership", "#careerdevelopment", "#professionalnetworking", "#businessstrategy",
                     "#Leadership", "#CareerGrowth", "#BusinessInsights", "#ProfessionalDevelopment",
                     "#FutureOfWork", "#IndustryTrends"],
        
        "facebook": ["#community", "#friendsandfamily", "#socialvibes", "#SocialVibes", "#CommunityLove",
                     "#FBFamily", "#GoodVibesOnly", "#StayConnected", "#FriendsAndFamily"],
        
        "threads": ["#threadsapp", "#trendingNow", "#ThreadsApp", "#HotTake", "#TrendingNow",
                    "#DailyThoughts", "#CreatorsOnThreads", "#TechTalks"],
        
        "pinterest": ["#aesthetic", "#moodboard", "#creativeinspo", "#AestheticInspo", "#DreamyVibes",
                      "#CreativeIdeas", "#PinterestFinds", "#InspoDaily", "#MoodBoardMagic"],
        
        "twitter": ["#trending", "#viralpost", "#newpost"],
        
        "reddit": ["#askreddit", "#discussion", "#redditcommunity"]
    }

    platform_context = {
        "instagram": "clean, aesthetic content",
        "facebook": "friendly, community vibes",
        "linkedin": "professional, business language",
        "pinterest": "aesthetic, creative moodboard",
        "threads": "expressive, bold",
        "tiktok": "engaging, fast-paced",
        "youtube": "SEO, creator-focused",
        "twitter": "short and bold",
        "reddit": "discussion-based"
    }.get(platform.lower(), "general")

    prompt = f"""
Generate 12 platform-specific hashtags.

Platform: {platform}
Platform style: {platform_context}

Topic: {query}
Keywords: {', '.join(seed_keywords)}

Rules:
- Only output hashtags separated by spaces.
- No explanations.
- No slang (lowkey, highkey, no cap, fr, obsessed).
"""

    try:
        text = await groq_generate_text(MODEL, prompt)
        ai_tags = [t for t in text.split() if t.startswith("#")]
    except Exception:
        ai_tags = []

    if not ai_tags:
        ai_tags = [f"#{k.lower()}" for k in seed_keywords if k]

    # -------------------------------
    # ✅ LIMITS & MERGING LOGIC HERE
    # -------------------------------
    ai_tags = ai_tags[:6]                 # limit AI-generated hashtags
    common = COMMON_TAGS.get(platform.lower(), [])[:10]  # limit common hashtags

    # Merge: common first, then AI
    merged = common + [tag for tag in ai_tags if tag not in common]

    # Remove duplicates while preserving order
    final_tags = list(dict.fromkeys(merged))

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
