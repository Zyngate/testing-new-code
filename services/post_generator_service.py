# stelle_backend/services/post_generator_service.py
import asyncio
import itertools
import random
import re
import httpx
from typing import List, Dict, Any, Set
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
    "tiktok": 4,  # TikTok: exactly 4 hashtags
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
# PLATFORM CTA SETTINGS
# ---------------------------

PLATFORM_CTAS = {
    "instagram": [
        "What are your thoughts?",
        "Save this for later.",
        "Follow for more insights.",
        "Would love your perspective.",
        "Comment your take below.",
        "Share this with someone who needs it.",
        "Food for thought.",
        "Let that sink in."
    ],
    "threads": [
        "Thoughts?",
        "This says a lot.",
        "Make it make sense.",
        "Well then.",
        "That’s something."
    ],
    "tiktok": [
        "Follow for more",
        "Like & share",
        "Comment your thoughts"
    ],
    "youtube": [
        "Let me know your take in the comments.",
        "More breakdowns coming soon.",
        "Subscribe for more insights.",
        "Share your thoughts below.",
        "Stay tuned for the next one.",
        "Drop your perspective below.",
        "Curious to hear your view."
    ],
    "pinterest": [
        "Save for later.",
        "Pin this for inspiration.",
        "Add to your board.",
        "Worth saving.",
        "Bookmark this idea."
    ],
    "facebook": [
        "What do you think?",
        "Tag someone who needs this.",
        "Share your thoughts below.",
        "Drop a comment.",
        "React if you agree."
    ],
    "linkedin": [
        "Thoughts?",
        "What's your take on this?",
        "Agree or disagree?",
        "Share your experience.",
        "Worth discussing."
    ],
    "twitter": [
        "Thoughts?",
        "RT if you agree.",
        "Quote tweet your take.",
        "What's your stance?",
        "Discuss."
    ]
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

INSTAGRAM_FORBIDDEN_TERMS = (
    "on-screen text",
    "ocr",
    "transcript",
    "the phrase",
    "this phrase",
    "that phrase",
    "the line",
    "this line",
)

INSTAGRAM_FORBIDDEN_OPENERS = (
    "this video shows",
    "this video delves",
    "the video shows",
    "the video delves",
    "in this video",
)

INSTAGRAM_GENERIC_ABSTRACT_TERMS = (
    "the conversation",
    "this conversation",
    "the discussion",
    "this discussion",
    "raises important questions",
    "nature of equality",
    "complexities and nuances",
)

INSTAGRAM_ZERO_TOLERANCE_TERMS = (
    "conversation",
    "discussion",
)

INSTAGRAM_PASSIVE_OPENERS = (
    "a distinction is made",
    "it is discussed",
    "it is explained",
    "the discussion raises",
)

INSTAGRAM_DESCRIPTION_STYLE_TERMS = (
    "as the scene unfolds",
    "the scene unfolds",
    "in front of",
    "in the background",
    "the image of",
    "is displayed prominently",
    "the tone is serious",
    "it becomes clear",
    "steps up to",
    "speaking in front of",
)

INSTAGRAM_OVERLAP_STOPWORDS = {
    "the", "and", "that", "with", "from", "into", "this", "there", "their",
    "about", "because", "while", "where", "which", "what", "when", "have",
    "will", "would", "could", "should", "were", "been", "being", "they",
    "then", "than", "just", "very", "more", "most", "also", "only", "over",
    "under", "after", "before", "again", "still", "your", "you", "them",
}

TIKTOK_MIN_CHARS = 600
TIKTOK_MAX_CHARS = 700

# High-reach discovery tags for algorithm boost
INSTAGRAM_DISCOVERY_CORE = [
    "#fyp", "#explore", "#foryou", "#explorepage"
]

_instagram_discovery_deck = INSTAGRAM_DISCOVERY_CORE[:]
random.shuffle(_instagram_discovery_deck)
_instagram_discovery_index = 0


def _next_instagram_discovery_tag() -> str:
    """Return a rotating discovery tag so consecutive videos don't repeat one tag."""
    global _instagram_discovery_index, _instagram_discovery_deck

    if not _instagram_discovery_deck:
        return "#explore"

    if _instagram_discovery_index >= len(_instagram_discovery_deck):
        _instagram_discovery_deck = INSTAGRAM_DISCOVERY_CORE[:]
        random.shuffle(_instagram_discovery_deck)
        _instagram_discovery_index = 0

    tag = _instagram_discovery_deck[_instagram_discovery_index]
    _instagram_discovery_index += 1
    return tag

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


KEYWORD_PREFIX_PATTERNS = [
    r"^\s*here\s+are\s+(?:exactly\s+)?(?:three|3)\s+(?:short\s+)?(?:marketing\s+)?key\s*words?\s*[:\-]?",
    r"^\s*generate\s+(?:exactly\s+)?(?:three|3)\s+(?:short\s+)?(?:marketing\s+)?key\s*words?\s*[:\-]?",
    r"^\s*keywords?\s*[:\-]?",
]

INSTRUCTIONAL_QUERY_PATTERNS = [
    r"\bhere\s+are\s+(?:exactly\s+)?(?:three|3)\s+(?:short\s+)?(?:marketing\s+)?key\s*words?\b[:\-]?",
    r"\bgenerate\s+(?:exactly\s+)?(?:three|3)\s+(?:short\s+)?(?:marketing\s+)?key\s*words?\b[:\-]?",
    r"\bbased\s+on\s+the\s+conversation\b[:\-]?",
    r"\bfocus\s*:\s*[\w\s]*",
]

INSTRUCTIONAL_HINTS = (
    "here are",
    "marketing keywords",
    "keyword",
    "generate",
    "return only",
    "based on the conversation",
)


def _extract_keywords_from_text(raw: str) -> List[str]:
    cleaned = raw.strip()
    for pattern in KEYWORD_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[\n;|]+", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    keywords: List[str] = []
    for chunk in cleaned.split(","):
        kw = re.sub(r"[^A-Za-z0-9\s&+\-]", "", chunk).strip().lower()
        if not kw:
            continue
        if len(kw.split()) > 3:
            continue
        if kw in keywords:
            continue
        keywords.append(kw)

    return keywords[:3]


def _sanitize_query_for_suggestions(effective_query: str, seed_keywords: List[str]) -> str:
    clean_query = effective_query
    for pattern in INSTRUCTIONAL_QUERY_PATTERNS:
        clean_query = re.sub(pattern, " ", clean_query, flags=re.IGNORECASE)

    clean_query = re.sub(r"[^A-Za-z0-9\s]", " ", clean_query)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()
    clean_query = " ".join(clean_query.split()[:12]).strip()

    if len(clean_query.split()) < 2:
        clean_query = " ".join(seed_keywords).strip()

    return clean_query


def _is_instructional_suggestion(suggestion: str) -> bool:
    normalized = re.sub(r"[^A-Za-z\s]", " ", suggestion.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return True
    return any(hint in normalized for hint in INSTRUCTIONAL_HINTS)

# ---------------------------
# 1) keyword generation
# ---------------------------
async def generate_keywords_post(client: AsyncGroq, effective_query: str) -> List[str]:
    if not client:
        try:
            fallback = await groq_generate_text(MODEL, f"Generate 3 short marketing keywords for: {effective_query}. Return comma-separated.")
            kws = _extract_keywords_from_text(fallback or "")
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
        kws = _extract_keywords_from_text(raw)
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
        return _next_instagram_discovery_tag()
    
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
    final: List[str] = []
    seen: Set[str] = set()

    for suggestion in suggestions:
        if _is_instructional_suggestion(suggestion):
            continue

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

            lowered = tag.lower()
            if lowered not in seen:
                seen.add(lowered)
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

def is_safe_hashtag(tag: str) -> bool:
    banned_words = [
        "pedophilia",
        "abuse",
        "rape",
        "molestation",
        "sexual",
        "minor",
        "exploit"
    ]

    tag_lower = tag.lower()
    return not any(word in tag_lower for word in banned_words)

async def fetch_platform_hashtags(
    client: AsyncGroq,
    seed_keywords: List[str],
    platform: str,
    effective_query: str,
    autoposting: bool = False
) -> List[str]:

    platform = platform.lower()

    # 🎯 TikTok Hashtag Strategy
    # Structure: 1×Tier1 (broad) + 2×Tier2 (mid-size) + 1×Branded = 4 tags
    if platform == "tiktok":
        tags = []

        # Ensure at least 3 keywords for Tier 1 & 2
        kws = seed_keywords + ["content", "growth"]
        kws = kws[:3]

        # 1️⃣ Tier 1 — Broad category (1 tag)
        # High volume, general discovery tag
        # 🎯 High-volume TikTok discovery tags (rotate for variety)
        high_volume_tags = ["#fyp", "#foryou", "#viral", "#foryoupage"]

        # Pick one randomly (prevents repetition)
        tags.append(random.choice(high_volume_tags))

        # 2️⃣ Tier 2 — Mid-size, topic-specific (2 tags)
        tags.append(f"#{kws[1].replace(' ', '').lower()}")
        tags.append(f"#{kws[2].replace(' ', '').lower()}")

        # 3️⃣ Branded/Community tag (1 tag)
        tags.append("#StelleWorld")

        # Return exactly 4 hashtags
        return tags[:4]

    # 1️⃣ Suggestion-based relevant tags
    clean_query = _sanitize_query_for_suggestions(effective_query, seed_keywords)

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

    # 🎯 STRUCTURE BASED ON MODE
    if autoposting:
        relevant_limit = 1
        broad_limit = 1
        trending_limit = 1
    else:
        relevant_limit = 3
        broad_limit = 3
        trending_limit = 4

    ordered_tags: List[str] = []
    ordered_lower: Set[str] = set()

    # 1️⃣ RELEVANT (max 3)
    for tag in relevant_tags:
        if len(ordered_tags) >= relevant_limit:
            break
        if tag and tag.lower() not in ordered_lower:
            ordered_tags.append(tag)
            ordered_lower.add(tag.lower())

    # 2️⃣ BROAD (max 3 unique)
    broad_tags: List[str] = []
    broad_lower: Set[str] = set()

    # Add AI broad first
    if broad_tag:
        lowered = broad_tag.lower()
        if lowered not in ordered_lower and lowered not in broad_lower:
            broad_tags.append(broad_tag)
            broad_lower.add(lowered)

    # Add from seed keywords (unique only)
    for kw in set(seed_keywords):
        candidate = f"#{kw.replace(' ', '').lower()}"
        lowered = candidate.lower()
        if lowered not in ordered_lower and lowered not in broad_lower:
            broad_tags.append(candidate)
            broad_lower.add(lowered)
        if len(broad_tags) >= broad_limit:
            break

    for tag in broad_tags[:broad_limit]:
        lowered = tag.lower()
        if lowered not in ordered_lower:
            ordered_tags.append(tag)
            ordered_lower.add(lowered)

    # 3️⃣ TRENDING (max 4 unique)
    # 3️⃣ TRENDING
    if platform == "instagram":
        if autoposting:
            pool = INSTAGRAM_DISCOVERY_CORE
        else:
            pool = TRENDING_POOLS.get(platform, [])
    else:
        pool = TRENDING_POOLS.get(platform, [])

    pool = pool[:]  # avoid mutating global pool
    random.shuffle(pool)

    trending_added = 0

    # Use pre-fetched trending tag first so every request gets fresh rotation.
    if trending_limit > 0 and trending_tag and trending_tag.lower() not in ordered_lower:
        ordered_tags.append(trending_tag)
        ordered_lower.add(trending_tag.lower())
        trending_added += 1

    for tag in pool:
        if trending_added >= trending_limit:
            break
        if tag and tag.lower() not in ordered_lower:
            ordered_tags.append(tag)
            ordered_lower.add(tag.lower())
            trending_added += 1

    return ordered_tags[:10]

def enforce_instagram_constraints(text: str, target_chars: int = 1500) -> str:
    """
    Enforces:
    - <= target_chars characters (including spaces)
    - Exactly 3 paragraphs
    - No repetitive filler loops
    - No sentence cut-off at the end
    """

    min_chars = min(1000, target_chars)

    # 1) Normalize paragraphs first.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:3]
    while len(paragraphs) < 3:
        paragraphs.append("")

    text = "\n\n".join(paragraphs).strip()

    # 2) If too long, trim to sentence boundary.
    if len(text) > target_chars:
        text = _trim_to_sentence_boundary(text, target_chars)

    # 3) If too short, expand paragraph 3 with varied non-repetitive lines.
    expansion_lines = [
        "The pressure now is not abstract; the next response carries real strategic weight.",
        "What happens next will define whether this remains rhetoric or turns into policy.",
        "That is why every public statement here functions as a signal, not just a soundbite.",
        "Each line now raises the stakes for allies, opponents, and undecided observers alike.",
        "The outcome depends on what action follows these words, not on the words alone.",
    ]

    line_index = 0
    while len(text) < min_chars and line_index < len(expansion_lines) * 2:
        parts = text.split("\n\n")
        if len(parts) < 3:
            while len(parts) < 3:
                parts.append("")

        candidate = expansion_lines[line_index % len(expansion_lines)]
        line_index += 1

        # Avoid adding a sentence that already exists in the caption.
        if candidate.lower() in text.lower():
            continue

        joiner = " " if parts[2].strip() else ""
        parts[2] = (parts[2] + joiner + candidate).strip()
        text = "\n\n".join(parts).strip()

        if len(text) > target_chars:
            text = _trim_to_sentence_boundary(text, target_chars)
            break

    # 4) Final safety trim and clean sentence ending.
    if len(text) > target_chars:
        text = _trim_to_sentence_boundary(text, target_chars)

    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."

    # Keep exactly 3 paragraph blocks after all edits.
    final_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(final_parts) > 3:
        final_parts = final_parts[:3]
    while len(final_parts) < 3:
        final_parts.append("")

    return "\n\n".join(final_parts).strip()



# ---------------------------
# 3) caption generator
# ---------------------------

def _build_caption_prompt(
    p_norm: str,
    effective_query: str,
    detected_person: str | None = None,
    ocr_text: str | None = None,
    transcript: str | None = None
) -> str:
    person_instruction = ""
    raw_material = ""

    # Always inject OCR + transcript
    if ocr_text:
        raw_material += f"\nOCR TEXT DETECTED:\n{ocr_text[:400]}\n"

    if transcript:
        raw_material += f"\nTRANSCRIPT EXCERPT:\n{transcript[:400]}\n"

    # Person instruction only if person exists
    if detected_person:
        person_instruction = f"""
IMPORTANT:
- If relevant, you may mention {detected_person} once.
- Do NOT force the name repeatedly.
- Do NOT describe physical behavior, posture, or appearance.
"""
    """Build the caption prompt for a given platform."""
    # Detect if this is a marketing campaign (affects Threads tone)
    is_campaign = is_marketing_campaign(effective_query)
    
    identity_hook_rule = ""
    if detected_person:
        identity_hook_rule = f"""
- The first sentence does NOT need to start with a person's name.
- Keep focus on the claim, implication, or accountability angle.
- Do NOT use the word "speaker".
- Mention {detected_person} only if it adds context.
"""
    else:
        identity_hook_rule = """
STRICT RULE:
- No person was detected in the video.
- Do NOT invent or describe any person.

FORBIDDEN PHRASES:
- "a person"
- "a man"
- "a woman"
- "someone"
- "the speaker"
- "a presenter"
- "a guy"
- "a lady"

START the caption using:
- a quote
- an action
- an object
- a key moment from the video

Example openings:
"One sentence changes the entire discussion."
"The phrase 'build systems, not tasks' shifts the tone immediately."
"A whiteboard fills with three ideas that reshape the argument."
"""
    
    if p_norm == "instagram":
        return f"""
{person_instruction}

Write a long-form Instagram Reels caption in EXACTLY 3 paragraphs.

STRUCTURE (MANDATORY):

PARAGRAPH 1 — HOOK (NON-NEGOTIABLE)

{identity_hook_rule}

- 3-4 short lines.
- Must reference one concrete detail from OCR or transcript naturally.
- Do NOT label it as "OCR", "on-screen text", "transcript", "the phrase", "this phrase", "that phrase", "the line", or "this line".
- Write the hook as a sharp public-facing claim, not scene narration.

PARAGRAPH 2 — CONTEXT & INSIGHT  
- This paragraph MUST add NEW information and MUST NOT repeat the hook wording or meaning.
- Do NOT rephrase the hook sentence. Continue the story forward.
- Explain what happens next and why it matters.
- Human, conversational tone.
- Grounded in the topic meaning, not scene narration.
- This should be the longest paragraph.
- Never mention what anyone wore in the video.
- FORBIDDEN OPENERS: "This video shows", "This video delves", "The video shows", "The video delves", "In this video".

PARAGRAPH 3 — REFLECTION / CTA  
- 3-4 short lines.
- Invite the viewer to think, react, or comment  
- Natural and thoughtful, not salesy  
- End with a question or reflective line

CONTEXT ANCHOR (MANDATORY):

Base the caption on the main point/topic.
Do NOT narrate the scene like a visual description.
Do NOT describe posture, clothing, camera framing, background, or facial expressions.
Write as a social caption reaction/opinion, not as a shot-by-shot summary.

STYLE:
- Human and engaging  
- Confident, not corporate
- Critical-editorial voice (serious, direct, accountable)
- Clear, not abstract
- Sound like a real caption, not a report
- Use varied sentence shapes; avoid repetitive structures.
MANDATORY SPECIFICITY RULE:
- You MUST reference at least ONE concrete detail from the topic.
RULES:
- You MAY mention the topic, claim, or implication, but do NOT describe the scene frame-by-frame  
- Do NOT describe body language, camera angle, background objects, or posture  
- STRICTLY NO first-person language (no I, me, my, we)  
- No emojis  
- No hashtags inside the caption text  
- Avoid generic motivational filler  

SELF-CHECK:
- If first-person appears, rewrite internally before responding  
- If paragraph 2 repeats hook meaning, rewrite paragraph 2.
- If any forbidden opener appears, rewrite before returning.
- If "on-screen text" appears, rewrite before returning.
- If the caption reads like video description (scene/background/body language), rewrite before returning.
- If it sounds like a neutral news report instead of an opinionated caption, rewrite before returning.


LENGTH:
- Long-form: 1,000–1,500 characters total (including spaces)  
- EXACTLY 3 paragraphs separated by a blank line  

VIDEO DETAILS (USE THESE SPECIFICS CAREFULLY):

RAW VIDEO MATERIAL (PRIORITY FOR HOOK):
{raw_material}

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
- NEVER default to "Because..."
- If the sentence starts with "Because", rewrite internally

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
- NEVER default to "Because..."
- If the sentence starts with "Because", rewrite internally


EXAMPLES:
"Built centuries ago. Still arguing about it in 2026."
"Ancient structure. Zero clue who made it."
"Science: We figured it out. Also science: Actually no we didn't."
"Centuries passed. Still no agreement."
"Ancient structure. Zero clue who made it. Humans peaked then forgot everything."
"Science: We figured it out. Also science: Actually no we didn't."



FORBIDDEN:
❌ "Great, another mystery." (generic sarcasm)
❌ "Someone's hobby." (dismissive of subject)

ANTI-REPETITION RULE:
- DO NOT use overused words like "hidden", "secret", or "truth"
- Avoid repeating the same Example style across outputs
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

PARAPHRASE RULE:
- Reference at most two concrete elements from the source.
- Never copy full sentences or long phrases (6+ consecutive words) from the transcript, OCR, or topic text.
- Summarize the visual moment in fresh language so it reads like an interpretation, not a transcript.

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
Write a TikTok caption (TARGET {TIKTOK_MIN_CHARS}-{TIKTOK_MAX_CHARS} characters).

━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━

Line 1 — HOOK (MOST IMPORTANT)
- The strongest sentence
- Must grab attention immediately (before "more")
- Can be:
  • a bold claim
  • a sharp question
  • a relatable pain point

DO NOT start with:
"The phrase", "This phrase", "That phrase", "The line"

━━━━━━━━━━━━━━━━━━━━━━━
Line 2–4 — CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━

- Explain what the video delivers in simple, natural language
- MUST include 1–2 strong keywords from the content
- Keep it clear, not abstract
- No storytelling, no fluff


━━━━━━━━━━━━━━━━━━━━━━━
CONTENT RULES
━━━━━━━━━━━━━━━━━━━━━━━

- Keep it punchy but not too short
- Focus on ONE clear idea
- No surreal, poetic, or vague writing
- No exaggeration or fake drama

STYLE BOOST (MANDATORY):

- Sound like a real human, not a report
- Avoid formal phrases like "a discussion around", "highlights", "explores"
- Use simple, natural language
- First line must feel like a real thought someone would say out loud

━━━━━━━━━━━━━━━━━━━━━━━
PLATFORM SAFETY (STRICT ENFORCEMENT)
━━━━━━━━━━━━━━━━━━━━━━━

The caption MUST strictly follow TikTok Community Guidelines.

If ANY violation occurs, the output is INVALID and MUST be rewritten internally BEFORE returning.

STRICTLY FORBIDDEN:
- Misleading or exaggerated claims (e.g., "guaranteed", "100%", "you won’t believe")
- Engagement bait (e.g., "wait till the end", "like for part 2")
- Aggressive, insulting, or rude tone
- Harmful, sensitive, or controversial topics

ZERO TOLERANCE RULE:
If even ONE violation appears → rewrite the caption silently.

FINAL OUTPUT MUST BE:
✔ Safe
✔ Neutral
✔ Non-manipulative
✔ Platform-compliant

━━━━━━━━━━━━━━━━━━━━━━━
GLOBAL RULES
━━━━━━━━━━━━━━━━━━━━━━━

- STRICTLY NO first-person (no I, me, my, we)
- No emojis
- No hashtags inside caption

━━━━━━━━━━━━━━━━━━━━━━━
FINAL CHECK
━━━━━━━━━━━━━━━━━━━━━━━

- Must be between {TIKTOK_MIN_CHARS} and {TIKTOK_MAX_CHARS} characters
- Hook must be in first line
- Keywords must appear naturally
- CTA must be present

━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━

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
    effective_query: str,
    autoposting: bool
) -> tuple[str, List[str]]:
    """Generate hashtags for a single platform. Returns (platform, hashtags)."""
    try:
        tags = await fetch_platform_hashtags(
            client=None,
            seed_keywords=seed_keywords,
            platform=p_norm,
            effective_query=effective_query,
            autoposting=autoposting,
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


def _instagram_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _token_set_for_overlap(text: str) -> Set[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return {
        t for t in tokens
        if len(t) > 3 and t not in INSTAGRAM_OVERLAP_STOPWORDS
    }


def _has_instagram_forbidden_language(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in INSTAGRAM_FORBIDDEN_TERMS)


def _paragraph_two_has_forbidden_opener(paragraph_two: str) -> bool:
    p2 = paragraph_two.strip().lower()
    return any(p2.startswith(prefix) for prefix in INSTAGRAM_FORBIDDEN_OPENERS)


def _paragraph_has_passive_ai_opener(paragraph: str) -> bool:
    lowered = paragraph.strip().lower()
    return any(lowered.startswith(prefix) for prefix in INSTAGRAM_PASSIVE_OPENERS)


def _has_instagram_generic_abstract_language(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in INSTAGRAM_GENERIC_ABSTRACT_TERMS):
        return True

    # Zero-tolerance mode: reject any usage of these words.
    if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in INSTAGRAM_ZERO_TOLERANCE_TERMS):
        return True

    # Reject captions that overuse abstract framing instead of concrete moments.
    abstract_hits = 0
    for token in ("debate",):
        abstract_hits += len(re.findall(rf"\b{token}\b", lowered))
    return abstract_hits >= 2


def _has_instagram_description_style(text: str) -> bool:
    lowered = text.lower()
    term_hits = sum(1 for term in INSTAGRAM_DESCRIPTION_STYLE_TERMS if term in lowered)

    # Caption should not sound like scene narration.
    if term_hits >= 1:
        return True

    if re.search(r"\bas\s+the\s+scene\s+unfolds\b", lowered):
        return True

    if re.search(r"\bin\s+front\s+of\s+(a|the)\b", lowered):
        return True

    return False


def _is_hook_context_repetitive(hook: str, context: str) -> bool:
    hook_tokens = _token_set_for_overlap(hook)
    context_tokens = _token_set_for_overlap(context)
    if not hook_tokens or not context_tokens:
        return False

    overlap = len(hook_tokens & context_tokens)
    min_len = min(len(hook_tokens), len(context_tokens))
    return overlap >= max(4, int(0.45 * min_len))


def _needs_instagram_strict_rewrite(caption: str) -> bool:
    paragraphs = _instagram_paragraphs(caption)
    if len(paragraphs) < 3:
        return True
    if _has_instagram_forbidden_language(caption):
        return True
    if _has_instagram_generic_abstract_language(caption):
        return True
    if _has_instagram_description_style(caption):
        return True
    if _paragraph_two_has_forbidden_opener(paragraphs[1]):
        return True
    if _paragraph_has_passive_ai_opener(paragraphs[1]):
        return True
    if _paragraph_has_passive_ai_opener(paragraphs[2]):
        return True
    if _is_hook_context_repetitive(paragraphs[0], paragraphs[1]):
        return True
    if _is_hook_context_repetitive(paragraphs[1], paragraphs[2]):
        return True
    return False


async def _enforce_instagram_caption_strict(
    caption_text: str,
    effective_query: str,
    detected_person: str | None,
    ocr_text: str | None,
    transcript: str | None,
) -> str:
    if not caption_text:
        return caption_text

    if not _needs_instagram_strict_rewrite(caption_text):
        return caption_text

    repair_prompt = f"""
Rewrite this Instagram caption in EXACTLY 3 paragraphs.

STRICT RULES:
- Paragraph 1 = hook with one concrete detail.
- Paragraph 2 = NEW progression only. Do NOT repeat hook wording or meaning.
- Paragraph 3 = reflection only. Do NOT restate paragraph 2 in different words.
- Paragraph 2 must NOT start with: This video shows / This video delves / The video shows / The video delves / In this video.
- Paragraph 2 and 3 must NOT start with passive openers like: A distinction is made / The discussion raises.
- Do NOT use these words/phrases anywhere: on-screen text, OCR, transcript, the phrase, this phrase, that phrase, the line, this line.
- Do NOT write scene narration (no "in front of", "in the background", "as the scene unfolds", posture/body-language description, or camera framing details).
- Avoid abstract filler words: conversation, discussion, debate, complexities and nuances.
- Voice must be critical-editorial and opinionated, like a serious Instagram commentary caption.
- Do NOT write like a neutral news report.
- Human, natural, conversational tone.
- No first-person language.
- No emojis. No hashtags.

TOPIC:
{effective_query}

PERSON (if any):
{detected_person or "none"}

OCR:
{(ocr_text or "")[:300]}

TRANSCRIPT:
{(transcript or "")[:300]}

ORIGINAL CAPTION:
{caption_text}

Return ONLY the rewritten caption.
"""

    for pass_index in range(2):
        pass_prompt = repair_prompt
        if pass_index == 1:
            pass_prompt += "\nSECOND PASS (MANDATORY): Remove every occurrence of the words 'conversation' and 'discussion'. Keep exactly 3 paragraphs.\n"

        repaired = await safe_generate_caption(pass_prompt, platform="instagram", retries=1)
        if repaired and not _needs_instagram_strict_rewrite(repaired):
            return repaired.strip()

    # Last-resort deterministic cleanup to avoid forbidden AI-like phrasing.
    paragraphs = _instagram_paragraphs(caption_text)
    while len(paragraphs) < 3:
        paragraphs.append("")

    p1, p2, p3 = paragraphs[:3]
    lowered_p2 = p2.lower().strip()
    for prefix in INSTAGRAM_FORBIDDEN_OPENERS:
        if lowered_p2.startswith(prefix):
            p2 = re.sub(r"^\s*[^.?!]*[.?!]?\s*", "Then the moment shifts: ", p2, count=1)
            break

    combined = "\n\n".join([p1, p2, p3]).strip()
    for term in INSTAGRAM_FORBIDDEN_TERMS:
        combined = re.sub(re.escape(term), "", combined, flags=re.IGNORECASE)

    # Remove repetitive abstract framing that sounds AI-generated.
    combined = re.sub(r"\b(the\s+)?conversation\b", "the moment", combined, flags=re.IGNORECASE)
    combined = re.sub(r"\b(the\s+)?discussion\b", "the point", combined, flags=re.IGNORECASE)
    combined = re.sub(r"\bdebate\b", "issue", combined, flags=re.IGNORECASE)

    combined = re.sub(r"\s+", " ", combined)
    combined = combined.replace(" .", ".").replace(" ,", ",").strip()
    rebuilt = "\n\n".join([p.strip() for p in _instagram_paragraphs(combined)[:3] if p.strip()])
    return rebuilt if rebuilt else caption_text


def _trim_to_sentence_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    cut = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
    if cut != -1 and cut > int(max_chars * 0.6):
        return trimmed[: cut + 1].strip()
    return trimmed.rsplit(" ", 1)[0].strip()


def _is_valid_tiktok_caption_length(text: str) -> bool:
    # Count characters as displayed in the caption text (including spaces).
    length = len(text)
    return TIKTOK_MIN_CHARS <= length <= TIKTOK_MAX_CHARS


def _extract_tiktok_core_topic(effective_query: str) -> str:
    raw = re.sub(r"\s+", " ", (effective_query or "")).strip()
    if not raw:
        return ""

    # Remove boilerplate that can leak into fallback text.
    cleaned = re.sub(
        r"(?i)this video contains a real spoken conversation\.?\s*base understanding primarily on what is being said\.?",
        "",
        raw,
    )
    cleaned = re.sub(
        r"(?i)\b(conversation|on-screen text|visual context \(secondary\)|scene|visual details)\s*:\s*",
        " ",
        cleaned,
    )
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    return " ".join(words[:12]).strip()


async def _enforce_tiktok_caption_length(caption_text: str, effective_query: str) -> str:
    if not caption_text:
        return caption_text

    if _is_valid_tiktok_caption_length(caption_text):
        return caption_text

    repair_prompt = f"""
Rewrite this TikTok caption.

STRICT RULES:
- Keep it natural and human.
- No first-person.
- No emojis.
- No hashtags.
- Keep CTA in final line.
- Use 3-4 short lines.
- Final length MUST be between {TIKTOK_MIN_CHARS} and {TIKTOK_MAX_CHARS} characters.

TOPIC:
{effective_query}

ORIGINAL:
{caption_text}

Return ONLY the rewritten caption text.
"""

    repaired = await safe_generate_caption(repair_prompt, platform="tiktok", retries=1)
    if repaired and _is_valid_tiktok_caption_length(repaired):
        return repaired.strip()

    fallback = caption_text.strip()
    core_topic = _extract_tiktok_core_topic(effective_query)
    if len(fallback) < TIKTOK_MIN_CHARS:
        expansion_lines = [
            f"The core issue here is {core_topic}." if core_topic else "The core issue here is practical and immediate.",
            "This matters because people form opinions based on what they hear and repeat.",
            "The bigger question is how these ideas shape fairness, status, and day-to-day treatment.",
            "If we normalize hierarchy too early, it can influence behavior long before anyone questions it.",
            "Context matters because repeated beliefs can become social rules people stop challenging.",
            "The long-term impact is not only personal identity, but also how communities define who belongs.",
        ]

        idx = 0
        while len(fallback) < TIKTOK_MIN_CHARS:
            line = expansion_lines[idx % len(expansion_lines)]
            idx += 1
            if line:
                fallback += f"\n\n{line}"

    fallback = _trim_to_sentence_boundary(fallback, TIKTOK_MAX_CHARS)
    if len(fallback) < TIKTOK_MIN_CHARS:
        strict_pad = " Additional context is included to keep this explanation clear and complete."
        while len(fallback) < TIKTOK_MIN_CHARS:
            needed = TIKTOK_MIN_CHARS - len(fallback)
            fallback += strict_pad[:needed]

    return fallback

# ---------------------------
# CTA INJECTION ENGINE
# ---------------------------

def inject_platform_cta(platform: str, caption: str) -> tuple[str, str | None]:
    import random

    platform = platform.lower()

    if platform not in PLATFORM_CTAS:
        return caption, None

    cta = random.choice(PLATFORM_CTAS[platform])

    # Do NOT inject into caption
    # Only return separately

    return caption, cta

def contains_sensitive_topic(text: str) -> bool:
    risky_topics = [
        "pedophilia",
        "child abuse",
        "sexual abuse",
        "rape",
        "minor exploitation",
        "abuse allegations",
        "molestation"
    ]

    text = text.lower()
    return any(topic in text for topic in risky_topics)

async def _generate_caption_for_platform(
    p_norm: str,
    effective_query: str,
    detected_person: str | None = None,
    ocr_text: str | None = None,
    transcript: str | None = None
) -> tuple[str, str]:
    """Generate caption for a single platform. Returns (platform, caption)."""

    caption_prompt = _build_caption_prompt(
        p_norm,
        effective_query,
        detected_person,
        ocr_text,
        transcript,
    )
    # 🚨 HARD BLOCK: Sensitive topics (TikTok only)
    if p_norm == "tiktok" and contains_sensitive_topic(effective_query):
        logger.warning("Blocked unsafe TikTok topic")

        return (
            p_norm,
        "A serious discussion around safety and accountability in the industry\n\nKey concerns and perspectives are highlighted clearly\n\nComment your thoughts",
        "Comment your thoughts"
    )
    # 1️⃣ Generate caption (single pass only)
    caption_text = await safe_generate_caption(
        caption_prompt,
        platform=p_norm
    )
    # 🚨 POST SAFETY CHECK
    if p_norm == "tiktok" and caption_text:
        if contains_sensitive_topic(caption_text):
            logger.warning("Unsafe caption detected. Rewriting...")
            caption_text = "A serious discussion around safety and accountability in the industry\n\nKey concerns are explained with clarity and context\n\nComment your thoughts"
    
    if not caption_text:
        caption_text = ""
    
    # 🚫 Prevent hallucinated people when no detection exists
    if not detected_person and caption_text:
        caption_text = re.sub(
            r"\b(a person|a man|a woman|someone|the speaker|a presenter|a guy|a lady)\b",
        "",
        caption_text,
        flags=re.IGNORECASE
    )

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

    if p_norm == "instagram" and caption_text:
        caption_text = await _enforce_instagram_caption_strict(
            caption_text,
            effective_query,
            detected_person,
            ocr_text,
            transcript,
        )
        caption_text = enforce_instagram_constraints(caption_text, target_chars=1500)

    if p_norm == "tiktok" and caption_text:
        caption_text = await _enforce_tiktok_caption_length(
            caption_text,
            effective_query,
        )

    if not caption_text:
        caption_text = f"Caption could not be generated for {p_norm}. Please retry."

    logger.info(f"Caption generated for {p_norm}: {len(caption_text)} characters")

    caption_text, selected_cta = inject_platform_cta(p_norm, caption_text)
    return (p_norm, caption_text, selected_cta)

async def generate_topic(effective_query: str, seed_keywords: List[str]) -> str:
    prompt = f"""
Extract ONE clear topic (1–2 words max) from this content.

Rules:
- Must be a category (not a sentence)
- Human-readable (no #)
- Examples: Politics, AI, Marketing, Fitness

Content:
{effective_query}

Return ONLY the topic.
"""
    try:
        text = await groq_generate_text(MODEL, prompt)
        topic = text.strip().split("\n")[0]
        return topic
    except:
        return seed_keywords[0].capitalize()

def clean_youtube_title(title: str) -> str:
    import re

    banned_patterns = [
        r"\bhidden\b",
        r"\bthe hidden\b",
        r"\bhidden reason\b",
        r"\bhidden truth\b"
    ]

    for pattern in banned_patterns:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)

    # Clean spacing issues
    title = re.sub(r"\s+", " ", title).strip()

    return title

async def generate_caption_post(
    effective_query: str,
    seed_keywords: List[str],
    platforms: List[str],
    autoposting: bool = False,
    detected_person: str | None = None,
    ocr_text: str | None = None,
    transcript: str | None = None
) -> Dict[str, Any]:

    captions: Dict[str, str] = {}
    platform_hashtags: Dict[str, List[str]] = {}
    # Always start with a fresh titles dict for each call
    titles: Dict[str, str] = {}

    # Normalize platforms
    normalized_platforms = [p.lower().strip() for p in platforms]

    topic = await generate_topic(effective_query, seed_keywords)
    # Create all tasks for parallel execution
    hashtag_tasks = [
    _generate_hashtags_for_platform(
        p_norm,
        seed_keywords,
        effective_query,
        autoposting
    )
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

ANTI-REPETITION RULE:
- DO NOT use overused words like "hidden", "secret", or "truth"
- Avoid repeating the same trigger style across outputs

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
            title = clean_youtube_title(title)
            return (platform, title)
        except Exception:
            return (platform, "")

    # Only generate title tasks for YouTube and Pinterest, passing the generated caption for uniqueness
        # Wait for captions to be generated first
    caption_results = await asyncio.gather(*[
        _generate_caption_for_platform(
            p_norm,
            effective_query,
            detected_person,
            ocr_text,
            transcript
        )
        for p_norm in normalized_platforms
    ])


    captions = {}
    platform_ctas = {}

    # Collect captions + selected CTA per platform
    for result in caption_results:
        if len(result) == 3:
            p_norm, caption_text, selected_cta = result
            captions[p_norm] = caption_text

            if selected_cta:
                # Keep one selected CTA per platform.
                platform_ctas[p_norm] = selected_cta
        else:
            p_norm, caption_text = result
            captions[p_norm] = caption_text

    # Generate titles (after captions are ready)
    title_tasks = [
        generate_title(p_norm, effective_query, captions.get(p_norm, ""))
        for p_norm in normalized_platforms
        if p_norm in ("youtube", "pinterest")
    ]

        # Run hashtag + title tasks in parallel
    hashtag_results, title_results = await asyncio.gather(
        asyncio.gather(*hashtag_tasks),
        asyncio.gather(*title_tasks)
    )

    for p_norm, tags in hashtag_results:
        # 🚀 Skip topic tag modification for TikTok (exact 4 tags)
        if p_norm == "tiktok":
            platform_hashtags[p_norm] = tags
            continue

        topic_tag = f"#{topic.replace(' ', '')}"

        # Ensure topic is FIRST hashtag
        if topic_tag not in tags:
            tags.insert(0, topic_tag)
        else:
            tags.remove(topic_tag)
            tags.insert(0, topic_tag)

        # OPTIONAL: limit Threads hashtags
        if p_norm == "threads":
            tags = tags[:10]

        platform_hashtags[p_norm] = tags
        # Ensure exactly 10 hashtags for non-autoposting
        if not autoposting:
            tags = tags[:10] if len(tags) >= 10 else tags
        platform_hashtags[p_norm] = tags

    for p_norm, title in title_results:
        if p_norm in ("youtube", "pinterest"):
            titles[p_norm] = title

    # For NON-AUTOPOSTING: Return only caption + optional title per platform.
    # `caption` contains final composed text in strict order: caption -> hashtags -> selected CTA.
    if not autoposting:
        platforms_combined = {}
        for p_norm in normalized_platforms:
            caption_text = captions.get(p_norm, "")
            # TikTok: exactly 4 hashtags, others: up to 10
            hashtag_limit = 4 if p_norm == "tiktok" else 10
            hashtags_list = platform_hashtags.get(p_norm, [])[:hashtag_limit]
            selected_cta = platform_ctas.get(p_norm, "")
            title_text = titles.get(p_norm, "") if p_norm in ("youtube", "pinterest") else ""

            # Build fully composed caption in strict order: caption -> hashtags -> selected CTA.
            composed_parts = []
            if caption_text:
                composed_parts.append(caption_text.strip())

            # Add hashtag dump after caption
            if hashtags_list:
                hashtags_str = " ".join(hashtags_list)
                composed_parts.append(hashtags_str)

            # Add selected CTA at the end
            if selected_cta:
                composed_parts.append(selected_cta.strip())
            
            # Join with double newlines for clean separation
            composed_caption = "\n\n".join(composed_parts)
            
            platform_data = {
                "title": title_text if title_text else None,
                "caption": composed_caption,
            }

            # Only add topic for Threads
            if p_norm == "threads":
                platform_data["topic"] = topic
            platforms_combined[p_norm] = platform_data
        return {
            "status": "success",
            "keywords": seed_keywords,
            "platforms": platforms_combined
        }

    # For AUTOPOSTING: Keep original separate format (bulk service handles combining)
    return {
        "status": "success",
        "keywords": seed_keywords,
        "captions": captions,
        "topic": topic,
        "platform_hashtags": platform_hashtags,
        "ctas": platform_ctas,
        "titles": titles
    }