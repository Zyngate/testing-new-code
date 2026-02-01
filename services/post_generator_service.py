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

async def safe_generate_caption(prompt: str, platform: str, retries: int = 2) -> str | None:
    for attempt in range(retries):
        try:
            text = await groq_generate_text(MODEL, prompt)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            logger.warning(
                f"Caption generation failed for {platform} (attempt {attempt + 1}): {e}"
            )
    return None


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
async def generate_keywords_post(client: AsyncGroq, effective_query: str) -> List[str]:
    if not client:
        try:
            fallback = await groq_generate_text(MODEL, f"Generate 3 short marketing keywords for: {effective_query}. Return comma-separated.")
            kws = [k.strip() for k in fallback.replace("\n", ",").split(",") if k.strip()]
            return kws[:3] if kws else ["brand", "marketing", "content"]
        except:
            return ["brand", "marketing", "content"]

    prompt = f"Generate exactly 3 short marketing keywords for: {effective_query}. Only return 3 comma-separated keywords."

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
    effective_query: str
) -> List[str]:

    platform = platform.lower()

    # -------------------------------
    # 1) TRENDING / DISCOVERY → 4
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
        if platform == "instagram":
            relevant_prompt = f"""
Generate EXACTLY 3 Instagram hashtags that are:

CRITICAL REQUIREMENTS:
- DIRECTLY relevant to the video topic
- COMMONLY FOLLOWED by users (not just used)
- Estimated usage between ~300K and ~5M posts
- High-discovery but NOT ultra-broad
- Creator-native hashtags

AVOID:
- Ultra-broad hashtags (>50M posts)
- Tiny niche hashtags (<50K posts)
- Keyword-stuffed or invented hashtags

VIDEO CONTEXT:
{effective_query}

Output ONLY hashtags.
"""
        else:
            relevant_prompt = f"""
Generate 3 relevant hashtags commonly used on {platform}.

Context:
{effective_query}

Output ONLY hashtags.
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
{effective_query}

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
    # FINAL MERGE
    # -------------------------------
    final_tags = trending_tags + relevant_tags + broad_tags

    # Remove duplicates, preserve order
    return list(dict.fromkeys(final_tags))

def enforce_instagram_constraints(text: str, target_chars: int = 1000) -> str:
    """
    Enforces:
    - EXACTLY target_chars characters (including spaces)
    - EXACTLY 3 paragraphs
    - No sentence cut-off
    """

    # 1️⃣ Normalize paragraphs FIRST (before counting)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Force exactly 3 paragraphs
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:3]
    while len(paragraphs) < 3:
        paragraphs.append("")

    text = "\n\n".join(paragraphs)

    # 2️⃣ If too long → trim safely at sentence boundary
    if len(text) > target_chars:
        trimmed = text[:target_chars]

        last_punct = max(
            trimmed.rfind("."),
            trimmed.rfind("?"),
            trimmed.rfind("!")
        )

        if last_punct != -1:
            trimmed = trimmed[: last_punct + 1]

        text = trimmed.strip()

    # 3️⃣ If too short → PAD safely (controlled filler)
    filler = " Moments like this invite closer attention to how public images are shaped and interpreted."
    while len(text) < target_chars:
        # Add filler to LAST paragraph only
        parts = text.split("\n\n")
        parts[-1] += filler
        text = "\n\n".join(parts)

    # 4️⃣ Final hard trim (guaranteed safe now)
    return text[:target_chars]

# ---------------------------
# 3) caption generator
# ---------------------------
async def generate_caption_post(
    effective_query: str,
    seed_keywords: List[str],
    platforms: List[str],
) -> Dict[str, Any]:

    captions: Dict[str, str] = {}
    platform_hashtags: Dict[str, List[str]] = {}

    for p in platforms:
        p_norm = p.lower().strip()
        tone = PLATFORM_STYLES.get(p_norm, "Write a clean, engaging caption.")

        # ---------------------------
        # Hashtags
        # ---------------------------
        if p_norm == "instagram":
            tags = await fetch_platform_hashtags(
        client=None,
        seed_keywords=seed_keywords,
        platform="instagram",
        effective_query=effective_query
    )

        else:
            try:
                tags = await fetch_platform_hashtags(
                    None,
                    seed_keywords,
                    p_norm,
                    effective_query
                )
            except Exception as e:
                logger.error(f"Hashtag generation failed for {p_norm}: {e}")
                tags = []

        platform_hashtags[p_norm] = tags

        

        # ---------------------------
        # Caption prompt
        # ---------------------------
        if p_norm == "instagram":
            caption_prompt = f"""
Write a long-form Instagram Reels caption in EXACTLY 3 paragraphs.

STRUCTURE (MANDATORY):

PARAGRAPH 1 — HOOK  
- 2–3 short lines  
- Strong curiosity, tension, or emotional pull  
- Must clearly connect to the video  
- Designed to stop scrolling and trigger “more”

PARAGRAPH 2 — CONTEXT & INSIGHT  
- Explain what’s happening in the video  
- Add reasoning, meaning, or perspective  
- Human, conversational tone  
- Grounded in the actual video content  
- This should be the longest paragraph

PARAGRAPH 3 — REFLECTION / CTA  
- Invite the viewer to think, react, or comment  
- Natural and thoughtful, not salesy  
- End with a question or reflective line

STYLE:
- Human and engaging  
- Confident, not corporate  
- Clear, not abstract  

RULES:
- You MAY reference visuals or moments in the video  
- STRICTLY NO first-person language (no I, me, my, we)  
- No emojis  
- No hashtags inside the caption text  
- Avoid generic motivational filler  
SELF-CHECK:
- If first-person appears, rewrite internally before responding  


LENGTH:
- Long-form: 800–1,100 characters total  
- EXACTLY 3 paragraphs separated by a blank line  

VIDEO CONTEXT:
{effective_query}

Return ONLY the caption text.
"""

        elif p_norm == "threads":
            caption_prompt = f"""
Write a Threads post as a HUMAN reaction.

CRITICAL:
- DO NOT describe the image or video
- DO NOT summarize what is happening
- React as if you just watched this and had an immediate thought
- Do NOT explain the full context or backstory
- You MAY imply a consequence, concern, or realization

STRUCTURE (MANDATORY):
- MUST be written in 2 or 3 lines
- Each line must be a complete thought
- A single-line response is INVALID

ENGAGEMENT RULE:
- Engagement = facts + tension
- Do not state facts alone
- Every factual idea must introduce doubt, contrast, risk, or a consequence
- At least one line must introduce a specific implication or consequence
- If a sentence sounds neutral, rewrite it internally to add tension

STYLE:
- Conversational
- Reflective or curious
- No hashtags
- No emojis

LENGTH:
- 120–280 characters total

Context (for understanding only):
{effective_query}

VALIDATION:
- If the caption is only one line, rewrite it into multiple lines before returning.

Return ONLY the caption text with line breaks preserved.
"""

        elif p_norm == "linkedin":
            caption_prompt = f"""
Write a LinkedIn post reacting to this content.

RULES:
- Do NOT describe the video/image
- Focus on insight, takeaway, or professional relevance
- Thoughtful, human, confident tone
- No hashtags inside the text
ENGAGEMENT RULE:
- Engagement = facts + tension
- Do not state facts alone
- Every factual idea must introduce doubt, contrast, risk, or a consequence
- If a sentence sounds neutral or agreeable, rewrite it internally to add tension

LENGTH:
- 200–800 characters
- 3–5 short lines max

Context:
{effective_query}

Return ONLY the caption text.
"""

        elif p_norm == "facebook":
            caption_prompt = f"""
Write a Facebook caption as a casual human reaction.

CRITICAL:
- Do NOT describe the video or image
- Do NOT summarize events

STRUCTURE (MANDATORY):
- MUST contain at least 3 lines
- First line: engaging or opinionated opening
- Remaining lines: expand with thoughts, implications, or takeaways
- A single-line caption is INVALID

ENGAGEMENT RULE:
- Engagement = relatability + implication
- At least one line must add a concern, insight, or reflection
- If the caption feels light, expand it internally before returning

STYLE:
- Conversational
- Relatable
- Human
- Emojis optional (max 1–2 if natural)
- No hashtags

LENGTH:
- 180–450 characters

Context:
{effective_query}

VALIDATION:
- If fewer than 3 lines are produced, rewrite the caption to meet the structure.

Return ONLY the caption text with line breaks preserved.
"""
        elif p_norm == "pinterest":
            caption_prompt = f"""
Write a LONG-FORM Pinterest description (800–1000 characters).

STRUCTURE:
- Inspiring or curiosity-driven hook
- Aesthetic or thoughtful insight
- Clear CTA (save, reflect, explore)

RULES:
- Do NOT describe the image literally
- Evoke mood and meaning
- No hashtags in text

CONTEXT:
{effective_query}

Return ONLY the caption text.
"""
        elif p_norm == "youtube":
            caption_prompt = f"""
Write a LONG-FORM YouTube description (800–1000 characters).

STRUCTURE:
- Hook that builds curiosity
- Explain why this video matters
- Invite engagement (like, comment, reflect)

RULES:
- Do NOT list scenes
- Human, engaging tone
- CTA is mandatory

CONTEXT:
{effective_query}

Return ONLY the description text.
"""
        elif p_norm == "tiktok":
            caption_prompt = f"""
Write a LONG-FORM TikTok caption (800–1000 characters).

STRUCTURE (MANDATORY):

PARAGRAPH 1 — HOOK
- First 2–3 lines must stop scrolling
- Create curiosity, tension, or emotion
- Make the viewer NEED to watch

PARAGRAPH 2 — CORE IDEA
- Focus on ONE strong reaction, thought, or insight
- Explain why this moment matters
- Human, creator-style tone
- This should be the longest paragraph

PARAGRAPH 3 — CTA
- Invite engagement (comment, share, reflect)
- End with a direct or thoughtful question

RULES:
- Do NOT describe scenes or actions
- Do NOT summarize the video
- Sound like a real creator, not a brand
- No emojis
- No hashtags inside the text

CONTEXT (for understanding only):
{effective_query}

Return ONLY the caption text.
"""

        
        else:
            caption_prompt = f"""
Write a natural social media caption.

RULES:
- Do NOT describe the media
- React like a human
- Be platform-appropriate
- No hashtags in text

Context:
{effective_query}

Return ONLY the caption text.
"""

        # ---------------------------
        # Generate caption
        # ---------------------------
        caption_text = await safe_generate_caption(
            caption_prompt,
            platform=p_norm
)
        if not caption_text:
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

        logger.info(
    f"Caption generated for {p_norm}: {len(caption_text)} characters"
)

        
        if not caption_text:
            caption_text = f"Caption could not be generated for {p_norm}. Please retry."
        captions[p_norm] = caption_text


    return {
        "captions": captions,
        "platform_hashtags": platform_hashtags
    }   