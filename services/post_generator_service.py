# stelle_backend/services/post_generator_service.py
import asyncio
import itertools
import random
import re
import httpx
from typing import List, Dict, Any
from groq import AsyncGroq
from enum import Enum
from config import logger, HASHTAG_API_KEYS
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
            text = await groq_generate_text(
                MODEL,
                prompt,
                max_completion_tokens=600,
                temperature=0.7
)
            if text and text.strip():
                return text.strip()
        except Exception as e:
            logger.warning(
                f"Caption generation failed for {platform} (attempt {attempt + 1}): {e}"
            )
    return None

def is_marketing_campaign(effective_query: str) -> bool:
    """
    Detect if content is marketing/business-related.
    Requires 2+ keyword matches for stricter classification.
    
    Marketing = professional funny tone
    Non-marketing = brutally sarcastic tone
    """
    keywords = [
        "marketing", "automation", "campaign", "saas",
        "tool", "platform", "ai", "product", "growth", 
        "brand", "business", "entrepreneur", "startup", 
        "content strategy", "social media strategy", "software",
        "app", "service", "solution"
    ]
    q = effective_query.lower()
    matches = sum(1 for k in keywords if k in q)
    return matches >= 2  # Require 2+ keywords for marketing classification
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
    "obsessed", "literally", "no cap",
    "delulu"
]

# High-reach discovery tags for algorithm boost
INSTAGRAM_DISCOVERY_CORE = [
    "#fyp", "#explore", "#reels", "#foryou", "#explorepage"
]

# Broad-tag blacklist: discovery/algorithmic boost tags that should NOT be used
# as the 'broad/category' tag. We prefer category-level tags for the broad slot.
BROAD_BLACKLIST = {
    "#explore", "#explorepage", "#discoverypage",
    "#fyp", "#foryou", "#foryoupage", "#reels",
    "#trending", "#viral",
}

SUGGESTION_CACHE = {}

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

# Hashtag key rotation index
_hashtag_key_index = 0

def _get_next_hashtag_key() -> str:
    """Get next API key from hashtag pool with rotation."""
    global _hashtag_key_index
    key = HASHTAG_API_KEYS[_hashtag_key_index % len(HASHTAG_API_KEYS)]
    _hashtag_key_index += 1
    return key

async def _groq_hashtag_call(prompt: str) -> str | None:
    """Fast hashtag-specific Groq call with dedicated key pool."""
    api_key = _get_next_hashtag_key()
    client = AsyncGroq(api_key=api_key)
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=50,  # Short response for hashtags
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning(f"Hashtag API call failed: {e}")
        return None

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
        text = await _groq_hashtag_call(prompt)
        if text:
            return [t for t in text.split() if t.startswith("#")][:4]
        return []
    except:
        return []

TRENDING_GENERATORS = {
    platform: rotating_hashtag_picker(tags, 4)
    for platform, tags in TRENDING_POOLS.items()
}


async def _fetch_trending_tag(platform: str, effective_query: str) -> str | None:
    """Fetch trending tag - run in parallel."""
    if platform == "instagram":
        return random.choice(INSTAGRAM_DISCOVERY_CORE)
    
    pool = TRENDING_POOLS.get(platform, [])
    if pool:
        return random.choice(pool)
    
    # Only call AI if no pool exists
    prompt = f"""Generate ONE extremely trending hashtag for {platform} with very high reach.
Return ONLY one hashtag.
Context: {effective_query}"""
    text = await _groq_hashtag_call(prompt)
    if text:
        return next((t for t in text.replace("\n", " ").split() if t.startswith("#")), None)
    return None

async def fetch_search_suggestions(query: str, platform: str) -> List[str]:
    base_url = "https://suggestqueries.google.com/complete/search"

    cache_key = f"{platform}:{query.lower()}"
    if cache_key in SUGGESTION_CACHE:
        return SUGGESTION_CACHE[cache_key]

    if platform == "youtube":
        params = {"client": "firefox", "ds": "yt", "q": query}
    elif platform == "instagram":
        params = {"client": "firefox", "q": f"{query} instagram hashtag"}
    elif platform == "tiktok":
        params = {"client": "firefox", "q": f"{query} tiktok hashtag"}
    elif platform == "linkedin":
        params = {"client": "firefox", "q": f"{query} linkedin professional"}
    else:
        params = {"client": "firefox", "q": query}

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=2.0)
        ) as client:
            response = await client.get(base_url, params=params)
            suggestions = response.json()[1]
            SUGGESTION_CACHE[cache_key] = suggestions
            return suggestions
    except Exception:
        return []

def build_relevant_from_suggestions(
    suggestions: List[str],
    seed_keywords: List[str],
    platform: str
) -> List[str]:

    max_tags = 3
    final = []
    seen = set()

    for suggestion in suggestions:
        words = re.findall(r"[A-Za-z]+", suggestion)

        # Skip very short suggestions
        if len(words) < 2:
            continue

        # 1️⃣ Primary entity (first 2 words)
        primary = "#" + "".join(w.capitalize() for w in words[:2])

        # 2️⃣ Single strongest keyword
        single = "#" + words[0].capitalize()

        # 3️⃣ Two-word compressed variant
        if len(words) >= 3:
            compressed = "#" + words[0].capitalize() + words[2].capitalize()
        else:
            compressed = None

        candidates = [primary, single, compressed]

        for tag in candidates:
            if not tag:
                continue

            # Avoid overly long hashtags
            if len(tag) > 25:
                continue

            if tag.lower() not in seen:
                seen.add(tag.lower())
                final.append(tag)

        if len(final) >= max_tags:
            break

    # Fallback if suggestions weak
    if len(final) < max_tags:
        for kw in seed_keywords:
            tag = "#" + kw.replace(" ", "").capitalize()
            if tag.lower() not in seen:
                final.append(tag)
            if len(final) >= max_tags:
                break

    return final[:max_tags]

async def _fetch_broad_tag(platform: str, effective_query: str, seed_keywords: List[str]) -> str | None:
    """Fetch broad category tag - run in parallel."""
    prompt = f"""Generate ONE broad, category-level hashtag for {platform}.
CRITICAL: Must be a real category tag with 30K+ posts (e.g. #photography, #marketing, #fitness).
BANNED: #fyp, #explore, #viral, #trending, #reels, #foryou
Content: {effective_query}
Return EXACTLY ONE hashtag."""
    
    text = await _groq_hashtag_call(prompt)
    if text:
        candidate = next((t for t in text.replace("\n", " ").split() if t.startswith("#")), None)
        if candidate and candidate.lower() not in BROAD_BLACKLIST:
            return candidate
    
    # Fallback to seed keywords
    if len(seed_keywords) > 1:
        return f"#{seed_keywords[1].replace(' ', '').lower()}"
    elif seed_keywords:
        return f"#{seed_keywords[0].replace(' ', '').lower()}"
    return None


async def fetch_platform_hashtags(
    client: AsyncGroq,
    seed_keywords: List[str],
    platform: str,
    effective_query: str,
    autoposting: bool = False
) -> List[str]:

    platform = platform.lower()

    # 1️⃣ Suggestion-based relevant tags
    # Extract only meaningful sentence (skip instruction text)
    lines = [l.strip() for l in effective_query.split("\n") if l.strip()]
    clean_query = ""

    for line in lines:
        if line.lower().startswith(("this video", "you are", "focus on", "context")):
            continue
        if len(line.split()) > 3:  # Must be meaningful phrase
            clean_query = line
            break

    # Fallback
    if not clean_query:
        clean_query = effective_query[:80]
    clean_query = clean_query[:80]

    suggestions = await fetch_search_suggestions(clean_query, platform)
    relevant_tags = build_relevant_from_suggestions(
        suggestions,
        seed_keywords,
        platform
    )

    # 2️⃣ Run trending + broad in parallel
    trending_task = _fetch_trending_tag(platform, effective_query)
    broad_task = _fetch_broad_tag(platform, effective_query, seed_keywords)

    results = await asyncio.gather(
        trending_task,
        broad_task,
        return_exceptions=True
    )

    trending_tag = results[0] if not isinstance(results[0], Exception) else None
    broad_tag = results[1] if not isinstance(results[1], Exception) else None

    # 🎯 STRICT 3:3:4 STRUCTURE
    relevant_limit = 3
    broad_limit = 3
    trending_limit = 4

    ordered_tags = []

    # 1️⃣ RELEVANT (max 3)
    for tag in relevant_tags:
        if len(ordered_tags) >= relevant_limit:
            break
        if tag not in ordered_tags:
            ordered_tags.append(tag)

    # 2️⃣ BROAD (max 3 unique)
    broad_tags = []

    # Add AI broad first
    if broad_tag and broad_tag not in ordered_tags:
        broad_tags.append(broad_tag)

    # Add from seed keywords (unique only)
    for kw in set(seed_keywords):
        candidate = f"#{kw.replace(' ', '').lower()}"
        if candidate not in ordered_tags and candidate not in broad_tags:
            broad_tags.append(candidate)
        if len(broad_tags) >= broad_limit:
            break

    for tag in broad_tags[:broad_limit]:
        ordered_tags.append(tag)

    # 3️⃣ TRENDING (max 4 unique)
    pool = TRENDING_POOLS.get(platform, [])
    random.shuffle(pool)

    trending_added = 0
    for tag in pool:
        if trending_added >= trending_limit:
            break
        if tag not in ordered_tags:
            ordered_tags.append(tag)
            trending_added += 1

    return ordered_tags[:10]

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
    # Detect if this is a marketing campaign (affects Threads tone)
    is_campaign = is_marketing_campaign(effective_query)
    
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

CONTEXT ANCHOR (MANDATORY):

Base the caption on what specifically happens in the video.
Describe actions, visuals, or sequences shown.
Do not write abstract commentary about the topic.
The caption should reflect the flow of the video, not just the theme.

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

VIDEO DETAILS (USE THESE SPECIFICS CAREFULLY):

TOPIC:
{effective_query}

MANDATORY SPECIFICITY RULE:
- You MUST reference at least ONE concrete detail from the topic.
- Do NOT write abstract commentary.
- The caption should make it obvious what the video is about.
- If someone reads the caption without watching the video, 
  they must clearly understand the subject.

SELF-CHECK:
If the topic is not clearly identifiable from the caption,
rewrite internally before returning.

Return ONLY the caption text.
"""

    elif p_norm == "threads":
        return f"""
You are writing a Threads caption. The tone DEPENDS on the content type.

STEP 1: READ THE TOPIC AND DETERMINE CONTENT TYPE
Topic: {effective_query}

STEP 2: CHECK FOR MARKETING/BUSINESS KEYWORDS
Does the topic contain 2+ of these keywords?
- marketing, automation, campaign, saas, tool, platform, product, business, startup, brand, software, app, service, solution, entrepreneur, content strategy, social media strategy

If YES (2+ keywords) → USE TONE A (promotional)
If NO (0-1 keywords) → USE TONE B (sarcastic)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE A: PROMOTIONAL (for marketing/business content)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Make automation/product look SMART and DESIRABLE

MESSAGE:
✅ Automation = POWERFUL and EFFICIENT
✅ Using it = YOU ARE SMART
✅ Manual work = OLD/SLOW (mock the process, not the person)

RULES:
- Maximum 20 words
- Make the product sound DESIRABLE
- Make customers sound SMART for using it
- NEVER use: "still needs", "apparently", "trying to", "outsourcing", "can't manage"

EXAMPLES:
✅ "Manual post scheduling: Because who needs sleep or sanity."
✅ "Automation does in 10 seconds what used to take 4 hours."
✅ "Smart brands automate. Everyone else copies and pastes until midnight."

FORBIDDEN:
❌ "Apparently still needs automation." (sounds incomplete)
❌ "Still can't manage social media." (mocks customer)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TONE B: SARCASTIC (for non-marketing content)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Goal: Mock the absurd SITUATION, not the subject itself

VOICE:
- Deadpan and exhausted
- "Of course this happened" energy
- Mock humanity's inability to solve/explain things

RULES:
- MAXIMUM 15 words
- Mock the mystery/debate, NOT the thing itself
- Sound tired and unimpressed

EXAMPLES:
✅ "Built centuries ago. Still arguing about it in 2026. Very productive."
✅ "Ancient structure. Zero clue who made it. Humans peaked then forgot everything."
✅ "Science: We figured it out. Also science: Actually no we didn't."

FORBIDDEN:
❌ "Great, another mystery." (generic sarcasm)
❌ "Someone's hobby." (dismissive of subject)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FINAL CHECK BEFORE RESPONDING:
1. Did you count the keywords correctly?
2. Are you using the right tone for the content type?
3. If promotional: Does the product sound powerful (not broken)?
4. If sarcastic: Are you mocking the situation (not the subject)?

Return ONLY the caption.
"""

    elif p_norm == "linkedin":
        return f"""
Write a direct, blunt LinkedIn post.

This is NOT an article.
This is NOT educational.
This is NOT motivational.

VOICE:
- Clear
- Assertive
- Matter-of-fact
- Professional, but blunt
- Zero fluff

HARD RULES (NON-NEGOTIABLE):
- NO definitions (e.g. “X is important because…”)
- NO generic statements (“Effective communication is key…”)
- NO motivational language
- NO reflective summaries
- NO hashtags
- NO emojis
- NO first-person language (no I, we, my, our)
- NO questions

STRUCTURE (MUST FOLLOW EXACTLY):

OPENING:
- 1–2 short lines stating a concrete, opinionated claim
- It must sound like a conclusion, not an introduction

BULLETS:
- 4–6 bullets
- Each bullet must state a specific consequence, pattern, or mistake
- Each bullet must be actionable or observable
- No repeating ideas
- No filler language
- Each bullet starts with "• " on its own line

CLOSING:
- ONE short line stating a professional implication
- No reflection, no inspiration, no CTA

SELF-CHECK BEFORE FINAL ANSWER:
- If the opening could appear in a textbook → rewrite
- If any line sounds motivational → rewrite
- If it feels safe → rewrite

Context:
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

CONTEXT ANCHOR (MANDATORY):

Base the caption on what specifically happens in the video.
Describe actions, visuals, or sequences shown.
Do not write abstract commentary about the topic.
The caption should reflect the flow of the video, not just the theme.

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
You are a creator-growth AI writing a Pinterest caption.
Goal: Maximum saves and clicks. NOT engagement bait.

CORE VOICE:
- Sharp, self-aware but NOT aggressive here
- Aesthetic, curious, or quietly clever
- Slightly poetic or ironic
- "This feels true" energy, not "argue with me" energy

PINTEREST RULES (PRIORITY: SAVES + CLICKS):
- No hot takes
- Feel like something you would save at 2 a.m.
- Works with memes, visuals, or short-form video
- Short
- Never aggressive
- Never ask questions directly

CONTEXT ANCHOR (MANDATORY):

Base the caption on what specifically happens in the video.
Describe actions, visuals, or sequences shown.
Do not write abstract commentary about the topic.
The caption should reflect the flow of the video, not just the theme.

AVOID:
- First-person (no I, me, my, we)
- Emojis
- Hashtags
- Questions
- Calls to action
- Controversy

If something can be said in 5 words, do not use 10.

TOPIC:
{effective_query}

Return ONLY the caption (1-2 sentences max).
"""
    elif p_norm == "youtube":
        return f"""
You are writing a YouTube video description.

GOAL:
- Clearly explain the topic
- Improve search visibility
- Keep viewers interested
- Sound human, not academic

OPENING (First 2–3 lines):
- State the core topic immediately
- Mention relevant keywords naturally
- Do NOT reference the video itself
- Do NOT write like a blog article

AVOID THESE OPENING PHRASES:
- "This video is about"
- "In this video"
- "Today we"
- "This video explains"
- "This video covers"

BODY:
- Expand using 2–4 specific insights
- Anchor explanation to the example in the topic
- Avoid generic marketing advice
- Keep paragraphs short and readable
- Avoid textbook tone
- No artificial mystery

ENDING:
- Add a natural engagement prompt
- Keep it simple and authentic

SELF-CHECK:
- Fix spelling or grammar mistakes
- Ensure no dropped words (e.g., "om" instead of "from")

CONTEXT ANCHOR (MANDATORY):

Base the caption on what specifically happens in the video.
Describe actions, visuals, or sequences shown.
Do not write abstract commentary about the topic.
The caption should reflect the flow of the video, not just the theme.

VOICE RULE:
Write as if explaining the strategy behind the example,
not reviewing someone else’s advertisement.

Do NOT refer to the example as "an advertisement".
Do NOT write like an external observer.

RULES:
- No emojis
- No hashtags
- No exaggerated tension
- No vague filler language

TOPIC:
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

CONTEXT ANCHOR (MANDATORY):

Base the caption on what specifically happens in the video.
Describe actions, visuals, or sequences shown.
Do not write abstract commentary about the topic.
The caption should reflect the flow of the video, not just the theme.

CONTEXT (for understanding only):
{effective_query}

Return ONLY the caption text.
"""

    elif p_norm == "twitter":
        return f"""
Write a short, punchy tweet.

CONTEXT ANCHOR (MANDATORY):

Base the caption on what specifically happens in the video.
Describe actions, visuals, or sequences shown.
Do not write abstract commentary about the topic.
The caption should reflect the flow of the video, not just the theme.

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

CONTEXT ANCHOR (MANDATORY):

Base the caption on what specifically happens in the video.
Describe actions, visuals, or sequences shown.
Do not write abstract commentary about the topic.
The caption should reflect the flow of the video, not just the theme.

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

def has_prefix_corruption(text: str) -> bool:
    # Detect words like "eeing", "oming"
    bad_ing = re.findall(r"\b[a-z]{1,2}ing\b", text)

    # Detect suspicious short fragments surrounded by spaces
    suspicious_fragments = re.findall(r"\s[a-z]{1,2}\s", text)

    allowed = {" to ", " in ", " on ", " of ", " at ", " by ", " is ", " it ", " as ", " be ", " an ", " or ", " if ", " go ", " do "}
    suspicious_fragments = [w for w in suspicious_fragments if w not in allowed]

    return bool(bad_ing or suspicious_fragments)

def fix_dropped_first_char(text: str) -> str:
    """
    Fix common dropped-first-letter corruption:
    freedom -> eedom
    from -> om
    free -> ee
    freeing -> eeing
    """

    text = re.sub(r"\beedom\b", "freedom", text)
    text = re.sub(r"\bom\b", "from", text)
    text = re.sub(r"\beeing\b", "freeing", text)
    text = re.sub(r"\bee\b", "free", text)

    return text

async def _generate_caption_for_platform(
    p_norm: str,
    effective_query: str
) -> tuple[str, str]:
    """Generate caption for a single platform. Returns (platform, caption)."""

    caption_prompt = _build_caption_prompt(p_norm, effective_query)

    # 1️⃣ Generate caption (single pass only)
    caption_text = await safe_generate_caption(
        caption_prompt,
        platform=p_norm
    )

    if not caption_text:
        caption_text = ""

    # -------- HARD CLEANUP (Fix dropped prefixes safely) --------
    caption_text = re.sub(r"[ \t]+", " ", caption_text)
    caption_text = fix_dropped_first_char(caption_text)

    # 🔒 PLATFORM-SPECIFIC FORMAT GUARD
    if p_norm == "linkedin":
        caption_text = enforce_vertical_bullets(caption_text)

    # 🔒 THREADS SARCASTIC ENFORCEMENT
    if p_norm == "threads":
        is_campaign = is_marketing_campaign(effective_query)
        if not is_campaign:
            words = caption_text.split()

            # Max 15 words
            if len(words) > 15:
                caption_text = " ".join(words[:15])

            # Max 120 characters
            if len(caption_text) > 120:
                caption_text = caption_text[:120].rsplit(" ", 1)[0]

            if not caption_text.strip():
                caption_text = "Well, this happened."

    # -------- SAFE CLEANING --------
    caption_text = caption_text.replace('\\"', "").replace('"', "").strip()

    # Fix common dropped-preposition typos
    caption_text = re.sub(r"\bom\b", "from", caption_text)
    caption_text = re.sub(r"\bor\b", "for", caption_text)
    caption_text = re.sub(r"\bf\b", "of", caption_text)

    for bad in BANNED_WORDS:
        pattern = r'\b' + re.escape(bad) + r'\b'
        caption_text = re.sub(pattern, '', caption_text, flags=re.IGNORECASE)

    # Preserve paragraph breaks properly
    caption_text = "\n\n".join(
        [" ".join(p.split()) for p in caption_text.split("\n\n") if p.strip()]
    )

    if not caption_text:
        caption_text = f"Caption could not be generated for {p_norm}. Please retry."

    logger.info(f"Caption generated for {p_norm}: {len(caption_text)} characters")

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
Generate ONE highly clickable YouTube title.

PRIMARY GOAL:
Make the viewer feel compelled to click.

The title MUST create a strong curiosity gap.
It should hint at something hidden, overlooked, changing, or misunderstood.

PSYCHOLOGICAL TRIGGERS (Use at least ONE):
- A hidden reason
- A mistake most people make
- A shift happening quietly
- Something becoming obsolete
- An unexpected outcome
- A contrast between belief vs reality
- A system behind the scenes

REQUIREMENTS:
- 45–65 characters
- Must clearly reference the core topic
- Must include at least one relevant keyword
- Must feel slightly bold or disruptive
- No emojis
- No hashtags

STRICTLY AVOID:
- "What they're not telling you"
- "The truth about"
- "Complete guide"
- "Explained"
- Generic vague titles
- Corporate tone
- Marketing hype words (revolutionary, ultimate, game-changing)

CRITICAL:
If the title feels safe, neutral, or predictable,
rewrite it internally until it creates tension.

The viewer should feel:
"I need to know what this means."

Context:
{query}

Return ONLY the title.
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