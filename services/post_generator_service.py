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

def enforce_vertical_bullets(text: str) -> str:
    lines = text.split("\n")
    fixed_lines = []

    for line in lines:
        if "•" in line and not line.strip().startswith("•"):
            # Split inline bullets into separate lines
            parts = [p.strip() for p in line.split("•") if p.strip()]
            for part in parts:
                fixed_lines.append(f"• {part}")
        else:
            fixed_lines.append(line)

    return "\n\n".join(fixed_lines)


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

def _build_caption_prompt(p_norm: str, effective_query: str) -> str:
    """Build the caption prompt for a given platform."""
    if p_norm == "instagram":
        return f"""
Write a long-form Instagram Reels caption in EXACTLY 3 paragraphs.

STRUCTURE (MANDATORY):

PARAGRAPH 1 — HOOK  
- 2–3 short lines  
- Strong curiosity, tension, or emotional pull  
- Must clearly connect to the video  
- Designed to stop scrolling and trigger "more"

PARAGRAPH 2 — CONTEXT & INSIGHT  
- Explain what's happening in the video  
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
        return f"""
Write a Threads caption that STOPS the scroll.

CRITICAL:
- Do NOT describe the video or image
- Do NOT summarize what happened
- Do NOT use first-person language (no "I", "we", "my", "our")
- Neutral or polite reactions are NOT allowed

OPENING RULE (MANDATORY):
- The FIRST line must be eye-catching
- Use a bold claim, sharp contrast, uncomfortable truth, or provocative question
- The first line must feel risky or surprising

ENGAGEMENT RULE:
- Engagement = tension + specificity
- Introduce doubt, risk, contradiction, or consequence
- If the caption feels safe or agreeable, rewrite it internally

STYLE:
- Conversational but impersonal
- Direct
- Punchy
- No hashtags
- No emojis

FORMAT:
- 2–5 short lines
- Line breaks required
- Do NOT write a single-line caption

LENGTH:
- 100–250 characters

Context (for understanding only):
{effective_query}

Return ONLY the caption text.
"""

    elif p_norm == "linkedin":
        return f"""
Write a LinkedIn post intended to be published from a PERSONAL PROFILE (not a company page).

GOAL:
- Write a structured, professional mini-blog post
- Share thoughtful insight relevant to work, leadership, or decision-making

TONE:
- Professional
- Calm
- Reflective
- Descriptive (not promotional, not reactive)

STRICT RULES:
- STRICTLY NO first-person language (no I, me, my, we, our)
- Do NOT sound like TikTok, Instagram, or Threads
- Do NOT use emojis or hashtags
- Do NOT write marketing copy or slogans
- Do NOT describe the video or visuals

STRUCTURE (MANDATORY — MUST FOLLOW EXACTLY):

PARAGRAPH 1:
- Professional framing and context (2–3 lines)

PARAGRAPH 2:
- Why this topic matters from a professional or decision-making perspective (1–2 lines)

BULLET SECTION (MANDATORY):
- Write 4–6 bullet points
- EACH bullet MUST be on its OWN LINE
- EACH bullet MUST start with "• " at the beginning of the line
- AFTER EVERY bullet, insert a newline
- NEVER place multiple bullets on the same line
- NEVER write bullets inline with text
- ONE idea per bullet only

CLOSING PARAGRAPH:
- A reflective conclusion connecting the topic to judgment, responsibility, or leadership

FORMAT RULES (CRITICAL):
- Use blank lines between paragraphs
- Bullets must appear as a vertical list
- If bullets appear on the same line, rewrite internally until they are vertical

LENGTH:
- 500–1,000 characters
- Multi-paragraph output required

Context (for understanding only):
{effective_query}

Return ONLY the caption text.
"""

    elif p_norm == "facebook":
        return f"""
Write a Facebook caption as a casual human reaction.

CRITICAL:
- STRICTLY NO first-person language (no I, me, my, we, our)
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
        return f"""
Write a SHORT Pinterest caption (100–150 characters max).

STYLE:
- Aesthetic, dreamy, inspiring
- One punchy line + optional short CTA
- Think mood-board vibes

RULES:
- STRICTLY NO first-person language (no I, me, my, we, our)
- Short and sweet - max 2 sentences
- Evoke emotion, not description
- No hashtags in text

CONTEXT:
{effective_query}

Return ONLY the caption text (under 150 characters).
"""

    elif p_norm == "youtube":
        return f"""
Write a LONG-FORM YouTube description (800–1000 characters).

STRUCTURE:
- Hook that builds curiosity
- Explain why this video matters
- Invite engagement (like, comment, reflect)

RULES:
- STRICTLY NO first-person language (no I, me, my, we, our)
- Do NOT list scenes
- Human, engaging tone
- CTA is mandatory

CONTEXT:
{effective_query}

Return ONLY the description text.
"""

    elif p_norm == "tiktok":
        return f"""
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
- STRICTLY NO first-person language (no I, me, my, we, our)
- Do NOT describe scenes or actions
- Do NOT summarize the video
- Sound like a real creator, not a brand
- No emojis
- No hashtags inside the text

CONTEXT (for understanding only):
{effective_query}

Return ONLY the caption text.
"""

    elif p_norm == "twitter":
        return f"""
Write a short, punchy tweet.

RULES:
- STRICTLY NO first-person language (no I, me, my, we, our)
- Bold and direct
- No slang
- No hashtags in text
- No emojis

Context:
{effective_query}

Return ONLY the caption text.
"""

    elif p_norm == "reddit":
        return f"""
Write an informative Reddit caption.

RULES:
- STRICTLY NO first-person language (no I, me, my, we, our)
- Discussion-starter style
- Do NOT summarize the video
- Sound like a real creator, not a brand
- No emojis
- No hashtags inside the text

CONTEXT (for understanding only):
{effective_query}

Return ONLY the caption text.
"""

    else:
        return f"""
Write a natural social media caption.

RULES:
- STRICTLY NO first-person language (no I, me, my, we, our)
- Do NOT describe the media
- React like a human
- Be platform-appropriate
- No hashtags in text

Context:
{effective_query}

Return ONLY the caption text.
"""


async def _generate_hashtags_for_platform(
    p_norm: str,
    seed_keywords: List[str],
    effective_query: str
) -> tuple[str, List[str]]:
    """Generate hashtags for a single platform. Returns (platform, hashtags)."""
    try:
        tags = await fetch_platform_hashtags(
            client=None,
            seed_keywords=seed_keywords,
            platform=p_norm,
            effective_query=effective_query
        )
    except Exception as e:
        logger.error(f"Hashtag generation failed for {p_norm}: {e}")
        tags = []
    return (p_norm, tags)


async def _generate_caption_for_platform(
    p_norm: str,
    effective_query: str
) -> tuple[str, str]:
    """Generate caption for a single platform. Returns (platform, caption)."""
    caption_prompt = _build_caption_prompt(p_norm, effective_query)
    caption_text = await safe_generate_caption(caption_prompt, platform=p_norm)
    
    if not caption_text:
        caption_text = ""

    # 🔒 PLATFORM-SPECIFIC FORMAT GUARD
    if p_norm == "linkedin":
        caption_text = enforce_vertical_bullets(caption_text)

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

    logger.info(f"Caption generated for {p_norm}: {len(caption_text)} characters")

    if not caption_text:
        caption_text = f"Caption could not be generated for {p_norm}. Please retry."
    
    return (p_norm, caption_text)


async def generate_caption_post(
    effective_query: str,
    seed_keywords: List[str],
    platforms: List[str],
) -> Dict[str, Any]:

    captions: Dict[str, str] = {}
    platform_hashtags: Dict[str, List[str]] = {}

    # Normalize platforms
    normalized_platforms = [p.lower().strip() for p in platforms]

    # Create all tasks for parallel execution
    hashtag_tasks = [
        _generate_hashtags_for_platform(p_norm, seed_keywords, effective_query)
        for p_norm in normalized_platforms
    ]
    caption_tasks = [
        _generate_caption_for_platform(p_norm, effective_query)
        for p_norm in normalized_platforms
    ]

    # Run all hashtag and caption tasks in parallel
    all_results = await asyncio.gather(*hashtag_tasks, *caption_tasks)

    # Split results: first half are hashtags, second half are captions
    num_platforms = len(normalized_platforms)
    hashtag_results = all_results[:num_platforms]
    caption_results = all_results[num_platforms:]

    # Populate dictionaries from results
    for p_norm, tags in hashtag_results:
        platform_hashtags[p_norm] = tags

    for p_norm, caption_text in caption_results:
        captions[p_norm] = caption_text

    return {
        "captions": captions,
        "platform_hashtags": platform_hashtags
    }
