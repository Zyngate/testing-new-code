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

HASHTAG_LIMITS = {
    "instagram": 10,
    "linkedin": 10,
    "facebook": 10,
    "pinterest": 10,
    "tiktok": 10,
    "youtube": 10,
    "threads": 10,
    "twitter": 10,
    "reddit": 10,
}

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

def is_marketing_campaign(effective_query: str) -> bool:
    keywords = [
        "marketing", "automation", "campaign", "saas",
        "tool", "platform", "ai", "product", "growth", "brand", "business", "entrepreneur", "startup", "content strategy", "social media strategy"
    ]
    q = effective_query.lower()
    return any(k in q for k in keywords)
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

# High-reach discovery tags for algorithm boost
INSTAGRAM_DISCOVERY_CORE = [
    "#fyp", "#explore", "#reels", "#foryou", "#explorepage", "#trending"
]


# Algorithm-optimized hashtag pools for maximum reach
TRENDING_POOLS = {

    "instagram": [
        "#fyp", "#foryou", "#foryoupage", "#reels",
        "#instareels", "#explorepage", "#explore",
        "#trending", "#reelsoftheday", "#instagood",
        "#reelsviral", "#trendingreels", "#instadaily",
        "#reelstrending", "#instaviral", "#reelsofinstagram",
        "#discoverypage", "#viralreels", "#instagramreels",
        "#reelitfeelit", "#reelslovers"
    ],

    "tiktok": [
        "#fyp", "#foryou", "#foryoupage", "#xyzbca",
        "#viral", "#tiktokviral", "#trending",
        "#tiktoktrending", "#fypシ", "#fypviral",
        "#trendingsound", "#viralvideo", "#tiktokcommunity",
        "#tiktokfamous", "#duet", "#stitch",
        "#parati", "#xuhuong"
    ],

    "youtube": [
        "#shorts", "#youtubeshorts", "#shortsvideo",
        "#shortsyoutube", "#shortsfeed", "#trending",
        "#viralshorts", "#shortsviral", "#ytshorts",
        "#subscribe", "#youtube", "#shortsforyou",
        "#shortsclip", "#shortsdaily"
    ],

    "threads": [
        "#threads", "#threadsapp", "#threadscommunity",
        "#trending", "#threadsviral", "#threadspost",
        "#meta", "#threadsdaily", "#threadstalk",
        "#threadsupdate"
    ],

    "pinterest": [
        "#pinterest", "#pinterestinspired", "#pinterestideas",
        "#aesthetic", "#inspo", "#moodboard",
        "#pinterestfinds", "#pinterestworthy",
        "#aestheticpinterest", "#pinit"
    ],

    "facebook": [
        "#facebookreels", "#fbreels", "#reels",
        "#facebookviral", "#facebookwatch", "#fbvideo",
        "#facebookvideo", "#trending", "#viral",
        "#facebookcommunity"
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
    final_tags: List[str] = []

    # --------------------------------------------------
    # 1️⃣ EXTREME TRENDING (platform-native, high reach)
    # --------------------------------------------------
    trending_tag = None

    try:
        if platform == "instagram":
            trending_tag = random.choice(INSTAGRAM_DISCOVERY_CORE)
        else:
            pool = TRENDING_POOLS.get(platform, [])
            if pool:
                trending_tag = random.choice(pool)
            else:
                prompt = f"""
Generate ONE extremely trending hashtag for {platform} with very high reach.
Return ONLY one hashtag.

Context:
{effective_query}
"""
                text = await groq_generate_text(MODEL, prompt)
                trending_tag = next(
                    (t for t in text.replace("\n", " ").split() if t.startswith("#")),
                    None
                )
    except Exception:
        trending_tag = None

    if trending_tag:
        final_tags.append(trending_tag)

    # --------------------------------------------------
    # 2️⃣ RELEVANT (most contextually accurate)
    # --------------------------------------------------
    relevant_tag = None

    try:
        prompt = f"""
Generate ONE highly relevant hashtag for {platform} that directly matches the content topic.
Return ONLY one hashtag.

Context:
{effective_query}
"""
        text = await groq_generate_text(MODEL, prompt)
        relevant_tag = next(
            (t for t in text.replace("\n", " ").split() if t.startswith("#")),
            None
        )
    except Exception:
        relevant_tag = None

    if not relevant_tag and seed_keywords:
        relevant_tag = f"#{seed_keywords[0].replace(' ', '')}"

    if relevant_tag and relevant_tag not in final_tags:
        final_tags.append(relevant_tag)

    # --------------------------------------------------
    # 3️⃣ BROAD (category / domain-level)
    # --------------------------------------------------
    broad_tag = None

    try:
        prompt = f"""
Generate ONE broad, category-level hashtag for {platform} describing the general domain.
Return ONLY one hashtag.

Context:
{effective_query}
"""
        text = await groq_generate_text(MODEL, prompt)
        broad_tag = next(
            (t for t in text.replace("\n", " ").split() if t.startswith("#")),
            None
        )
    except Exception:
        broad_tag = None

    if not broad_tag and len(seed_keywords) > 1:
        broad_tag = f"#{seed_keywords[1].replace(' ', '')}"
    elif not broad_tag:
        broad_tag = "#content"

    if broad_tag and broad_tag not in final_tags:
        final_tags.append(broad_tag)

    # --------------------------------------------------
    # 4️⃣ BUILD 4 : 3 : 3 HASHTAG BUCKETS
    # --------------------------------------------------

    trending_tags = []
    relevant_tags = []
    broad_tags = []

    # ---- 4 Trending ----
    if platform == "instagram":
        core_pool = INSTAGRAM_DISCOVERY_CORE[:]
        random.shuffle(core_pool)
        for tag in core_pool:
            if len(trending_tags) >= 2:
                break
            if tag not in trending_tags:
                trending_tags.append(tag)

    # 2 from Instagram TRENDING_POOLS
        insta_pool = TRENDING_POOLS.get("instagram", [])[:]
        random.shuffle(insta_pool)
        for tag in insta_pool:
            if len(trending_tags) >= 4:
                break
            if tag not in trending_tags:
                trending_tags.append(tag)

    else:
        pool = TRENDING_POOLS.get(platform, [])[:]
        random.shuffle(pool)
        for tag in pool:
            if len(trending_tags) >= 4:
                break
            if tag not in trending_tags:
                trending_tags.append(tag)

    # ---- 3 Relevant ----
    for kw in seed_keywords:
        tag = f"#{kw.replace(' ', '')}"
        if tag not in relevant_tags:
            relevant_tags.append(tag)
        if len(relevant_tags) >= 3:
            break

    # ---- 3 Broad ----
    while len(broad_tags) < 3:
        broad_tags.append("#content")

    # ---- MERGE + DEDUPE (Order: relevant → broad → trending for 10M+ reach) ----
    final_tags = []
    for group in (relevant_tags, broad_tags, trending_tags):
        for tag in group:
            if tag not in final_tags:
                final_tags.append(tag)

    # ---- PLATFORM LIMIT ----
    limit = HASHTAG_LIMITS.get(platform, 10)
    return final_tags[:limit]

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
    is_campaign = is_marketing_campaign(effective_query)
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
Write a Threads post that's genuinely funny and sarcastic.

VOICE:
- Sarcastic, witty, dark humor
- Like a comedian roasting the situation
- The kind of post people screenshot and share
- Deadpan delivery, no trying too hard

RULES:
- 1-2 lines MAX
- Make it actually funny, not just edgy
- Observations > opinions
- NO first-person (no I, me, my, we)
- NO emojis
- NO hashtags
- NO explanations

TOPIC:
{effective_query}

Return ONLY the caption.
"""

    elif p_norm == "linkedin":
        return f"""
Write a LinkedIn post intended to be published from a PERSONAL PROFILE (not a company page).

GOAL:
- Write a structured, professional mini-blog post
- Share thoughtful insight relevant to work, industry, or professional growth (not always about leadership)

TONE:
- Professional
- Insightful
- Calm
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
- Why this topic matters from a professional or industry perspective (1–2 lines)

BULLET SECTION (MANDATORY):
- Write 4–6 bullet points
- EACH bullet MUST be on its OWN LINE
- EACH bullet MUST start with "• " at the beginning of the line
- AFTER EVERY bullet, insert a newline
- NEVER place multiple bullets on the same line
- NEVER write bullets inline with text
- ONE idea per bullet only

CLOSING PARAGRAPH:
- A reflective conclusion connecting the topic to professional growth, industry trends, or practical takeaways (not always leadership)

FORMAT RULES (CRITICAL):
- Use blank lines between paragraphs
- Bullets must appear as a vertical list
- If bullets appear on the same line, rewrite internally until they are vertical

If this is a marketing campaign:
- Frame this as an industry shift, not a product
- Emphasize outcomes, efficiency, and decision-making
- Avoid hype; sound credible and strategic

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
You are writing a Pinterest caption meant to be saved, not debated.

TONE:
- Calm
- Aesthetic
- Quietly clever
- Slightly ironic or poetic
- Never aggressive

RULES:
- Short caption
- 1–2 sentences max
- No questions
- No calls to action
- No controversy
- No emojis
- Feels like something worth saving at 2 a.m.

Context (for understanding only):
{effective_query}

Return ONLY the caption text.
"""
    elif p_norm == "youtube":
        return f"""
You are writing a YouTube video DESCRIPTION optimized for watch time and comments.

VOICE:
- Slightly dangerous
- Accusatory, but smart
- Implies something is being hidden or misunderstood
- Never educational
- Never safe

RULES:
- Build tension, do NOT resolve it
- Add context without giving answers
- Make the viewer feel late or misled
- End with a question that provokes disagreement
- Short paragraphs, readable out loud

STRICT:
- NO emojis
- NO hashtags
- NO motivational tone
- NO politeness

Context (for understanding only):
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
    # Always start with a fresh titles dict for each call
    titles: Dict[str, str] = {}

    # Normalize platforms
    normalized_platforms = [p.lower().strip() for p in platforms]

    # Create all tasks for parallel execution
    hashtag_tasks = [
        _generate_hashtags_for_platform(p_norm, seed_keywords, effective_query)
        for p_norm in normalized_platforms
    ]

    # Title generation tasks (YouTube and Pinterest only, platform-specific prompts)
    async def generate_title(platform: str, query: str, caption: str = "") -> tuple[str, str]:
        if platform == "youtube":
            prompt = f"""
Generate a UNIQUE YouTube video title that is curiosity-driven, highly clickable, and trend-aware. Do NOT repeat or paraphrase the video description or caption. Use a different angle, hook, or question. Make it stand out in search and recommendations.

Rules:
- 40–60 characters
- Clickable but not clickbait
- Use curiosity, action, or emotional hooks
- Avoid repeating or paraphrasing the main description/caption
- Make it feel native to YouTube trends

Context:
{query}

Caption (for reference, do NOT copy):
{caption}

Return ONLY the title text.
"""
        elif platform == "pinterest":
            prompt = f"""
Generate a UNIQUE Pinterest pin title that is inspirational, visual, and mood-board friendly. Do NOT repeat or paraphrase the pin description or caption. Use a different visual or emotional angle. Make it stand out for discovery and inspiration.

Rules:
- Short (3–6 words)
- Inspirational, visual, or aspirational
- Avoid repeating or paraphrasing the main description/caption
- Make it feel native to Pinterest trends

Context:
{query}

Caption (for reference, do NOT copy):
{caption}

Return ONLY the title text.
"""
        else:
            return (platform, "")
        try:
            text = await groq_generate_text(MODEL, prompt)
            title = text.split("\n")[0].replace('"', '').strip()
            return (platform, title)
        except Exception:
            return (platform, "")

    # Only generate title tasks for YouTube and Pinterest, passing the generated caption for uniqueness
    # Wait for captions to be generated first
    caption_results = await asyncio.gather(*[
        _generate_caption_for_platform(p_norm, effective_query)
        for p_norm in normalized_platforms
    ])
    captions = {p_norm: caption_text for p_norm, caption_text in caption_results}

    title_tasks = [
        generate_title(p_norm, effective_query, captions.get(p_norm, ""))
        for p_norm in normalized_platforms if p_norm in ("youtube", "pinterest")
    ]

    # Run hashtag and title tasks in parallel (captions already generated above)
    hashtag_results, title_results = await asyncio.gather(
        asyncio.gather(*hashtag_tasks),
        asyncio.gather(*title_tasks)
    )

    for p_norm, tags in hashtag_results:
        platform_hashtags[p_norm] = tags

    for p_norm, title in title_results:
        # Always include YouTube and Pinterest in the titles dict, even if title is empty
        if p_norm in ("youtube", "pinterest"):
            titles[p_norm] = title

    return {
        "captions": captions,
        "platform_hashtags": platform_hashtags,
        "titles": titles
    }