# stelle_backend/services/engagement_service.py
"""
Engagement Service — Phase 4 Core of Autonomous Engagement.

Responsible for:
- Building the tone-matched LLM prompt
- Generating replies via Groq
- Posting replies via platform APIs
- Logging every reply to comment_replies_log
- Spam detection
- Model routing (simple vs complex comments)
"""

import asyncio
import json
import random
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from database import db
from config import logger
from services.ai_service import groq_generate_text
from services.tone_calibration_service import get_active_tone_profile
from services.post_content_analyzer import get_post_context
from services.social_engagement_api import post_reply


# ── Collections ──────────────────────────────────────────────
replies_log_col = db["comment_replies_log"]
engagement_settings_col = db["engagement_settings"]


# ── Constants ────────────────────────────────────────────────
SIMPLE_MODEL = "llama-3.1-8b-instant"
COMPLEX_MODEL = "llama-3.3-70b-versatile"

# Human-like pacing: random delay between replies (seconds)
MIN_REPLY_DELAY = 15
MAX_REPLY_DELAY = 45

# Spam detection keywords (basic — can be extended per user via blacklisted_words)
DEFAULT_SPAM_KEYWORDS = [
    "buy followers", "get rich", "dm me for", "check my bio",
    "free money", "click this link", "onlyfans", "🔥🔥🔥🔥🔥",
    "follow me back", "f4f", "l4l", "sub4sub",
]


# ── System Prompt Template ───────────────────────────────────
REPLY_SYSTEM_PROMPT = """You are acting as {user_name} replying to comments on their social media post.
Your ONLY job is to write a reply that sounds EXACTLY like this person would write it.

TONE PROFILE:
- Style: {tone_label}
- Formality: {formality} (0=very casual, 1=very formal)
- Emoji usage: {emoji_frequency} (0=never, 1=always)
- Average reply length: ~{avg_reply_length} words
- Humor level: {humor_level}
- When someone says something negative: {confrontation_style}
- Signature phrases they use: {signature_phrases}
- Words/topics to AVOID: {avoid_topics}

POST CONTEXT (what this post is about):
- Content: {content_summary}
- Topics: {key_topics}
- Post sentiment: {post_sentiment}
- Likely comment themes: {likely_comment_themes}
- Platform: {platform}

COMMENT CONTEXT:
- Comment type: {comment_type}
- Inferred commenter tone: {comment_tone}

STRICT RULES:
1. Match the tone profile EXACTLY — this must sound like the real person
2. Keep reply around {avg_reply_length} words (can be ±5 words)
3. Use emojis at approximately {emoji_frequency} frequency (0=never, 1=every reply)
4. Be relevant to the post content
5. NEVER be rude, offensive, political, or controversial
6. NEVER reveal you are an AI or automated system
7. Do NOT use generic stock phrases like "Great content!" or "Thanks for sharing!"
8. NEVER disagree, argue, or contradict the commenter. Stay supportive, appreciative, or neutral.
9. If comment_type is emoji_only, your reply MUST be emoji-only (no words, no hashtags, no punctuation), with 1-4 emojis that fit post context and user tone.
10. Reply ONLY with the reply text — nothing else, no quotes, no prefix
"""


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

async def generate_reply_for_comment(
    user_id: str,
    post_id: str,
    platform: str,
    comment: Dict[str, Any],
    access_token: str = "",
    user_name: str = "the user",
    **reply_kwargs,
) -> Dict[str, Any]:
    """
    Core function: Generate a tone-matched reply for a single comment.
    
    Steps:
        1. Load user's Tone DNA
        2. Load post content context
        3. Select appropriate LLM model
        4. Build prompt and generate reply
        5. Post reply (or queue for review)
        6. Log to comment_replies_log

    Returns:
        Dict with reply details including status.
    """
    comment_id = comment.get("comment_id", "")
    comment_text = (comment.get("text", "") or "").strip()
    comment_author = comment.get("author", "unknown")

    try:
        # ── 1. Check if already replied ──
        already = await replies_log_col.find_one({
            "comment_id": comment_id,
            "user_id": user_id,
            "status": {"$in": ["posted", "approved_posted", "pending_review", "draft"]}
        })
        if already:
            return {"status": "skipped", "reason": "already_replied"}

        # ── 2. Load engagement settings ──
        settings = await engagement_settings_col.find_one({"user_id": user_id})
        if not settings:
            return {"status": "skipped", "reason": "no_engagement_settings"}

        reply_mode = settings.get("reply_mode", "automatic")

        # ── 3. Blank comment check (always treated as spam) ──
        if _is_blank_comment(comment_text):
            logger.info(f"🚫 Blank comment detected, skipping comment {comment_id}")
            await _log_reply(
                user_id,
                post_id,
                comment_id,
                comment_author,
                comment_text,
                "[SPAM: BLANK_COMMENT]",
                platform,
                0,
                "spam_skipped",
            )
            return {"status": "spam_skipped", "reason": "blank_comment"}

        # ── 4. Spam check ──
        if await _is_spam(comment_text, settings):
            logger.info(f"🚫 Spam detected, skipping comment {comment_id}")
            await _log_reply(user_id, post_id, comment_id, comment_author,
                             comment_text, "[SPAM DETECTED]", platform, 0, "spam_skipped")
            return {"status": "spam_skipped"}

        # ── 5. Load Tone DNA ──
        tone = await get_active_tone_profile(user_id)
        if not tone:
            return {"status": "skipped", "reason": "no_tone_profile"}

        # ── 6. Load post context ──
        post_ctx = await get_post_context(post_id, user_id)

        comment_type = "emoji_only" if _is_emoji_only_comment(comment_text) else "text"
        comment_tone = _infer_comment_tone(comment_text)

        # ── 7. Select model ──
        model = select_model_for_comment(comment_text)

        # ── 8. Build prompt & generate ──
        reply_text = await _generate_reply(
            tone_profile=tone,
            post_context=post_ctx,
            comment_text=comment_text,
            comment_author=comment_author,
            platform=platform,
            model=model,
            user_name=user_name,
            comment_type=comment_type,
            comment_tone=comment_tone,
        )

        if reply_text and comment_type == "emoji_only" and not _is_emoji_only_comment(reply_text):
            # Hard guardrail: emoji-only comments must receive emoji-only replies.
            reply_text = _build_contextual_emoji_reply(post_ctx, tone)

        if reply_text and _is_disagreeing_reply(reply_text):
            reply_text = await _rewrite_non_disagreeing_reply(
                model=model,
                tone_profile=tone,
                post_context=post_ctx,
                comment_text=comment_text,
                comment_author=comment_author,
                platform=platform,
                user_name=user_name,
                draft_reply=reply_text,
                comment_type=comment_type,
                comment_tone=comment_tone,
            )

        if not reply_text:
            await _log_reply(user_id, post_id, comment_id, comment_author,
                             comment_text, "", platform,
                             tone.get("version", 1), "failed")
            return {"status": "failed", "reason": "llm_generation_failed"}

        # ── 9. Handle based on reply_mode ──
        tone_version = tone.get("version", 1)

        if reply_mode == "automatic":
            # Post immediately via platform API
            success = await post_reply(
                platform=platform,
                comment_id=comment_id,
                reply_text=reply_text,
                access_token=access_token,
                **reply_kwargs,
            )
            status = "posted" if success else "failed"

            # Human-like delay before next reply
            delay = random.uniform(MIN_REPLY_DELAY, MAX_REPLY_DELAY)
            await asyncio.sleep(delay)

        elif reply_mode == "review":
            # Queue for user review
            status = "pending_review"

        elif reply_mode == "manual":
            # Save as draft suggestion
            status = "draft"

        else:
            status = "pending_review"  # Default to safe mode

        # ── 10. Log ──
        log_doc = await _log_reply(
            user_id, post_id, comment_id, comment_author,
            comment_text, reply_text, platform,
            tone_version, status
        )

        return {
            "status": status,
            "reply_text": reply_text,
            "comment_id": comment_id,
            "model_used": model,
            "log_id": log_doc.get("_id", ""),
        }

    except Exception as e:
        logger.error(f"❌ Error generating reply for comment {comment_id}: {e}", exc_info=True)
        return {"status": "failed", "reason": str(e)}


async def approve_reply(log_id: str, user_id: str, edited_text: str = None) -> Dict[str, Any]:
    """
    Approve a pending_review reply (and optionally edit it).
    Posts to platform and updates status.
    """
    from bson import ObjectId
    try:
        doc = await replies_log_col.find_one({
            "_id": ObjectId(log_id),
            "user_id": user_id,
            "status": {"$in": ["pending_review", "draft"]}
        })
        if not doc:
            return {"success": False, "reason": "Reply not found or already processed"}

        reply_text = edited_text if edited_text else doc.get("generated_reply", "")
        platform = doc.get("platform", "")
        comment_id = doc.get("comment_id", "")

        # Get auth token — for Instagram, prefer instafb (Business Graph)
        from services.social_engagement_api import get_user_auth
        auth = None
        if platform == "instagram":
            auth = await get_user_auth(user_id, "instafb")
        if not auth:
            auth = await get_user_auth(user_id, platform)
        if not auth:
            return {"success": False, "reason": "No auth token for platform"}

        access_token = auth.get("accessToken", "")

        # Post the reply
        success = await post_reply(
            platform=platform,
            comment_id=comment_id,
            reply_text=reply_text,
            access_token=access_token,
        )

        new_status = "approved_posted" if success else "failed"

        await replies_log_col.update_one(
            {"_id": ObjectId(log_id)},
            {"$set": {
                "status": new_status,
                "generated_reply": reply_text,  # In case it was edited
                "approved_at": datetime.now(timezone.utc),
            }}
        )

        return {"success": success, "status": new_status}

    except Exception as e:
        logger.error(f"Error approving reply {log_id}: {e}")
        return {"success": False, "reason": str(e)}


async def reject_reply(log_id: str, user_id: str) -> Dict[str, Any]:
    """Reject a pending_review reply."""
    from bson import ObjectId
    try:
        result = await replies_log_col.update_one(
            {"_id": ObjectId(log_id), "user_id": user_id},
            {"$set": {"status": "rejected", "rejected_at": datetime.now(timezone.utc)}}
        )
        return {"success": result.modified_count > 0}
    except Exception as e:
        logger.error(f"Error rejecting reply {log_id}: {e}")
        return {"success": False, "reason": str(e)}


async def get_reply_queue(user_id: str, status: str = "pending_review", limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get the reply approval queue for a user.
    Can filter by status: pending_review, draft, posted, etc.
    Maps DB field names to frontend QueuedReply interface.
    """
    try:
        cursor = replies_log_col.find(
            {"user_id": user_id, "status": status}
        ).sort("created_at", -1).limit(limit)

        items = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            # Map DB field 'generated_reply' to frontend field 'reply_text'
            doc["reply_text"] = doc.pop("generated_reply", "")
            # Ensure 'model_used' is present (frontend expects it)
            doc.setdefault("model_used", "")
            # Convert datetime to ISO string
            for dt_key in ("created_at", "replied_at", "approved_at", "rejected_at"):
                val = doc.get(dt_key)
                if val and hasattr(val, "isoformat"):
                    doc[dt_key] = val.isoformat()
            # Map 'replied_at' to 'posted_at' for frontend
            if "replied_at" in doc:
                doc["posted_at"] = doc.get("replied_at")
            items.append(doc)
        return items
    except Exception as e:
        logger.error(f"Error fetching reply queue for {user_id}: {e}")
        return []


async def get_reply_stats(user_id: str) -> Dict[str, Any]:
    """
    Get engagement reply statistics for a user.
    Returns fields matching the frontend EngagementStats interface:
      total_replies, automatically_posted, pending_review, approved,
      rejected, draft, replies_today, by_platform
    """
    try:
        # Status breakdown
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1}
            }}
        ]
        stats = {}
        async for doc in replies_log_col.aggregate(pipeline):
            stats[doc["_id"]] = doc["count"]

        total = sum(stats.values())

        # Today's count
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        replies_today = await replies_log_col.count_documents({
            "user_id": user_id,
            "status": {"$in": ["posted", "approved_posted"]},
            "created_at": {"$gte": today_start},
        })

        # Platform breakdown
        platform_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$platform",
                "count": {"$sum": 1}
            }}
        ]
        by_platform = {}
        async for doc in replies_log_col.aggregate(platform_pipeline):
            if doc["_id"]:  # Skip null platforms
                by_platform[doc["_id"]] = doc["count"]

        return {
            "total_replies": total,
            "automatically_posted": stats.get("posted", 0) + stats.get("approved_posted", 0),
            "pending_review": stats.get("pending_review", 0),
            "approved": stats.get("approved_posted", 0),
            "rejected": stats.get("rejected", 0),
            "draft": stats.get("draft", 0),
            "replies_today": replies_today,
            "by_platform": by_platform,
        }
    except Exception as e:
        logger.error(f"Error fetching reply stats for {user_id}: {e}")
        return {
            "total_replies": 0,
            "automatically_posted": 0,
            "pending_review": 0,
            "approved": 0,
            "rejected": 0,
            "draft": 0,
            "replies_today": 0,
            "by_platform": {},
        }


# ══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════

def select_model_for_comment(comment_text: str) -> str:
    """Route to appropriate model based on comment complexity."""
    words = comment_text.split()
    has_question = "?" in comment_text
    is_short = len(words) <= 6

    # Simple: short positive comments, emojis only, etc.
    if is_short and not has_question:
        return SIMPLE_MODEL
    # Complex: questions, long text, or potentially sensitive
    return COMPLEX_MODEL


async def _generate_reply(
    tone_profile: Dict[str, Any],
    post_context: Optional[Dict[str, Any]],
    comment_text: str,
    comment_author: str,
    platform: str,
    model: str,
    user_name: str = "the user",
    comment_type: str = "text",
    comment_tone: str = "neutral",
) -> str:
    """Build the LLM prompt and generate the reply."""

    traits = tone_profile.get("extracted_traits", {})

    # Build system prompt
    system_msg = REPLY_SYSTEM_PROMPT.format(
        user_name=user_name,
        tone_label=tone_profile.get("tone_label", "Friendly"),
        formality=traits.get("formality", 0.3),
        emoji_frequency=traits.get("emoji_frequency", 0.5),
        avg_reply_length=traits.get("avg_reply_length_words", 18),
        humor_level=traits.get("humor_level", "subtle"),
        confrontation_style=traits.get("confrontation_style", "polite_correction"),
        signature_phrases=", ".join(traits.get("signature_phrases", [])) or "none",
        avoid_topics=", ".join(traits.get("avoid_topics", [])) or "none",
        content_summary=post_context.get("content_summary", "Social media post") if post_context else "Social media post",
        key_topics=", ".join(post_context.get("key_topics", [])) if post_context else "general",
        post_sentiment=post_context.get("sentiment", "neutral") if post_context else "neutral",
        likely_comment_themes=", ".join(post_context.get("likely_comment_themes", [])) if post_context else "general audience reactions",
        platform=platform,
        comment_type=comment_type,
        comment_tone=comment_tone,
    )

    # Build user prompt
    user_prompt = f'Reply to this comment from @{comment_author}: "{comment_text}"'

    reply = await groq_generate_text(
        model=model,
        prompt=user_prompt,
        system_msg=system_msg,
        temperature=0.8,
        max_completion_tokens=150,
    )

    # Clean up: remove quotes, prefixes like "Reply:" etc.
    if reply:
        reply = reply.strip().strip('"').strip("'")
        # Remove common LLM artifacts
        for prefix in ["Reply:", "Response:", "Here's my reply:", "My reply:"]:
            if reply.lower().startswith(prefix.lower()):
                reply = reply[len(prefix):].strip()

    return reply


async def _rewrite_non_disagreeing_reply(
    model: str,
    tone_profile: Dict[str, Any],
    post_context: Optional[Dict[str, Any]],
    comment_text: str,
    comment_author: str,
    platform: str,
    user_name: str,
    draft_reply: str,
    comment_type: str,
    comment_tone: str,
) -> str:
    """One-pass rewrite when a draft sounds contradictory or argumentative."""
    rewrite_system = REPLY_SYSTEM_PROMPT + "\n11. Never include correctional language like 'you are wrong', 'actually', or 'I disagree'."

    traits = tone_profile.get("extracted_traits", {})
    rewrite_system = rewrite_system.format(
        user_name=user_name,
        tone_label=tone_profile.get("tone_label", "Friendly"),
        formality=traits.get("formality", 0.3),
        emoji_frequency=traits.get("emoji_frequency", 0.5),
        avg_reply_length=traits.get("avg_reply_length_words", 18),
        humor_level=traits.get("humor_level", "subtle"),
        confrontation_style=traits.get("confrontation_style", "polite_correction"),
        signature_phrases=", ".join(traits.get("signature_phrases", [])) or "none",
        avoid_topics=", ".join(traits.get("avoid_topics", [])) or "none",
        content_summary=post_context.get("content_summary", "Social media post") if post_context else "Social media post",
        key_topics=", ".join(post_context.get("key_topics", [])) if post_context else "general",
        post_sentiment=post_context.get("sentiment", "neutral") if post_context else "neutral",
        likely_comment_themes=", ".join(post_context.get("likely_comment_themes", [])) if post_context else "general audience reactions",
        platform=platform,
        comment_type=comment_type,
        comment_tone=comment_tone,
    )

    rewrite_prompt = (
        f'Original comment from @{comment_author}: "{comment_text}"\n'
        f'Draft reply: "{draft_reply}"\n\n'
        "Rewrite this so it does not disagree with the commenter while staying relevant to the post context and user tone. "
        "Return reply text only."
    )

    rewritten = await groq_generate_text(
        model=model,
        prompt=rewrite_prompt,
        system_msg=rewrite_system,
        temperature=0.6,
        max_completion_tokens=120,
    )

    if rewritten:
        cleaned = rewritten.strip().strip('"').strip("'")
        if not _is_disagreeing_reply(cleaned):
            return cleaned
    return draft_reply


async def _is_spam(comment_text: str, settings: Dict[str, Any]) -> bool:
    """
    Check if a comment is spam using keyword matching.
    Combines default keywords + user's blacklisted words.
    """
    if _is_blank_comment(comment_text):
        return True

    if not settings.get("spam_filter_enabled", True):
        return False

    text_lower = comment_text.lower()

    # Combined blacklist
    all_keywords = DEFAULT_SPAM_KEYWORDS + settings.get("blacklisted_words", [])

    for keyword in all_keywords:
        if keyword.lower() in text_lower:
            return True

    # Emoji-only spam: allow normal emoji reactions, block excessive bursts.
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]'
    )
    text_no_emoji = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+',
        '', comment_text
    ).strip()
    emoji_count = len(emoji_pattern.findall(comment_text))
    if not text_no_emoji and emoji_count >= 8:
        return True

    return False


def _is_blank_comment(comment_text: str) -> bool:
    return not (comment_text or "").strip()


def _is_emoji_only_comment(text: str) -> bool:
    """True when text contains emoji/symbol reactions only (no words or digits)."""
    if _is_blank_comment(text):
        return False

    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+'
    )
    has_emoji = bool(emoji_pattern.search(text))
    stripped = emoji_pattern.sub("", text)
    stripped = re.sub(r"[\s\.,!?~#@&*+\-_/\\|:;'\"()\[\]{}<>]+", "", stripped)

    return has_emoji and stripped == ""


def _infer_comment_tone(comment_text: str) -> str:
    text = (comment_text or "").lower()
    if not text:
        return "neutral"
    if "?" in text:
        return "curious"
    if any(w in text for w in ["love", "awesome", "great", "amazing", "🔥", "😍", "👏"]):
        return "positive"
    if any(w in text for w in ["bad", "hate", "worst", "boring", "terrible", "not good"]):
        return "critical"
    return "neutral"


def _is_disagreeing_reply(reply_text: str) -> bool:
    text = (reply_text or "").lower()
    patterns = [
        r"\bi disagree\b",
        r"\byou(?:\s+are|'re)\s+wrong\b",
        r"\bthat(?:'s| is)\s+wrong\b",
        r"\bnot true\b",
        r"\bactually,?\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _build_contextual_emoji_reply(
    post_context: Optional[Dict[str, Any]],
    tone_profile: Optional[Dict[str, Any]],
) -> str:
    """Build a context-aware emoji-only fallback reply."""
    context_text = ""
    if post_context:
        context_text = (
            f"{post_context.get('content_summary', '')} "
            f"{' '.join(post_context.get('key_topics', []))} "
            f"{post_context.get('sentiment', '')}"
        ).lower()

    topic_map = [
        (["workout", "fitness", "gym", "training"], ["💪", "🔥", "🏋️"]),
        (["food", "recipe", "cooking", "meal"], ["😋", "🍽️", "🔥"]),
        (["travel", "trip", "vacation", "adventure"], ["✈️", "🌍", "✨"]),
        (["music", "song", "beat", "dance"], ["🎵", "🎶", "💃"]),
        (["business", "startup", "marketing", "sales"], ["📈", "🚀", "🙌"]),
        (["tech", "ai", "code", "software"], ["🤖", "💡", "🚀"]),
        (["fashion", "style", "outfit", "beauty"], ["✨", "💅", "😍"]),
        (["pet", "dog", "cat", "puppy"], ["🐾", "❤️", "🥹"]),
    ]

    emoji_pool = ["🔥", "👏", "🙌", "✨", "❤️"]
    for keywords, candidates in topic_map:
        if any(k in context_text for k in keywords):
            emoji_pool = candidates
            break

    emoji_frequency = (
        (tone_profile or {}).get("extracted_traits", {}).get("emoji_frequency", 0.5)
    )
    if emoji_frequency >= 0.8:
        count = 4
    elif emoji_frequency >= 0.4:
        count = 3
    else:
        count = 2

    chosen = [random.choice(emoji_pool) for _ in range(count)]
    return "".join(chosen)


async def _log_reply(
    user_id: str,
    post_id: str,
    comment_id: str,
    comment_author: str,
    comment_text: str,
    generated_reply: str,
    platform: str,
    tone_version: int,
    status: str,
) -> Dict[str, Any]:
    """Log a reply attempt to comment_replies_log."""
    doc = {
        "user_id": user_id,
        "post_id": post_id,
        "comment_id": comment_id,
        "comment_author": comment_author,
        "comment_text": comment_text,
        "generated_reply": generated_reply,
        "platform": platform,
        "tone_version": tone_version,
        "status": status,
        "created_at": datetime.now(timezone.utc),
        "replied_at": datetime.now(timezone.utc) if status in ("posted", "approved_posted") else None,
    }
    result = await replies_log_col.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return doc


async def get_daily_reply_count(user_id: str) -> int:
    """Count how many replies have been posted today for a user."""
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    count = await replies_log_col.count_documents({
        "user_id": user_id,
        "status": {"$in": ["posted", "approved_posted"]},
        "replied_at": {"$gte": today_start},
    })
    return count


async def get_post_reply_count(user_id: str, post_id: str) -> int:
    """Count replies for a specific post."""
    count = await replies_log_col.count_documents({
        "user_id": user_id,
        "post_id": post_id,
        "status": {"$in": ["posted", "approved_posted"]},
    })
    return count
