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

CHANGES v2:
- [SECURITY]  access_token always fetched server-side from DB — never trusted from caller
- [SECURITY]  Daily reply cap enforced inside generate_reply_for_comment (not just exposed as util)
- [SECURITY]  approve_reply validates edited_text length + strips script-like content
- [SECURITY]  ObjectId format validated before DB hits in approve_reply / reject_reply
- [SECURITY]  Spam filter normalizes Unicode before matching (prevents lookalike bypass)
- [SECURITY]  comment_text length capped before any LLM call
- [FINE-TUNE] Extracted _build_system_prompt() helper — eliminates 70-line prompt duplication
- [FINE-TUNE] select_model_for_comment: questions now always route to COMPLEX_MODEL (bug fix)
- [FINE-TUNE] _generate_reply: temperature is dynamic based on comment_tone
- [FINE-TUNE] _build_contextual_emoji_reply: uses random.sample to avoid repeated emojis
- [FINE-TUNE] _infer_comment_tone: checks positive/negative before '?' to avoid false 'curious'
"""

import asyncio
import json
import random
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from database import db
from config import logger
from services.ai_service import groq_generate_text
from services.tone_calibration_service import get_active_tone_profile
from services.post_content_analyzer import get_post_context
from services.video_cache_repo import get_cached_video_analysis
from services.social_engagement_api import post_reply, get_user_auth


# ── Collections ──────────────────────────────────────────────
replies_log_col = db["comment_replies_log"]
engagement_settings_col = db["engagement_settings"]
scheduled_posts_col = db["scheduledposts"]


# ── Constants ────────────────────────────────────────────────
SIMPLE_MODEL = "llama-3.1-8b-instant"
COMPLEX_MODEL = "llama-3.3-70b-versatile"
FACT_CHECK_MODEL = "llama-3.3-70b-versatile"

# Human-like pacing: random delay between replies (seconds)
MIN_REPLY_DELAY = 15
MAX_REPLY_DELAY = 45

# [SECURITY] Cap raw comment length fed into LLM (prevents prompt-stuffing)
MAX_COMMENT_INPUT_LEN = 1000

# [SECURITY] Cap edited reply length in approve_reply
MAX_EDITED_REPLY_LEN = 500

# Default daily reply cap — overridden by settings.max_replies_per_day if set
DEFAULT_MAX_DAILY_REPLIES = 200

# Spam detection keywords (basic — extended per user via blacklisted_words)
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
- Signature phrases/expressions (use ONLY when natural & context-appropriate): {signature_phrases}
- Exact words/phrases to AVOID using in final reply: {avoid_words_exact}
- Topics to avoid mentioning: {avoid_topics}

POST CONTEXT (what this post is about):
- Content: {content_summary}
- Topics: {key_topics}
- Post sentiment: {post_sentiment}
- Likely comment themes: {likely_comment_themes}
- Platform: {platform}

COMMENT CONTEXT:
- Comment type: {comment_type}
- Inferred commenter tone: {comment_tone}

FACT CONTEXT:
- Video summary: {video_summary}
- Transcript highlights: {transcript_highlights}
- Verified anchors: {fact_anchors}
- Lexical sync words: {lexical_sync_words}

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
11. IMPORTANT: Signature phrases should ONLY appear if they fit naturally into the conversation. Do NOT force them in. Omit them entirely if context doesn't support their use.
12. If FACT CONTEXT is available, keep the reply fact-grounded and never invent details.
13. Use 2-5 lexical sync words naturally (when available) so wording aligns with the commenter and post/video context.
14. Never use any exact word/phrase listed in the avoid list, even if it appears in the incoming comment.
"""


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

async def generate_reply_for_comment(
    user_id: str,
    post_id: str,
    platform: str,
    comment: Dict[str, Any],
    access_token: str = "",   # kept for API compatibility but ignored — fetched from DB
    user_name: str = "the user",
    **reply_kwargs,
) -> Dict[str, Any]:
    """
    Core function: Generate a tone-matched reply for a single comment.

    Steps:
        1. Load engagement settings + enforce daily cap
        2. Spam / blank checks
        3. Load Tone DNA + post context
        4. Select LLM model
        5. Build prompt and generate reply
        6. Post reply (or queue for review) using server-side auth token
        7. Log to comment_replies_log
    """
    comment_id = comment.get("comment_id", "")
    # [SECURITY] Cap comment text length before any processing or LLM call
    comment_text = (comment.get("text", "") or "").strip()[:MAX_COMMENT_INPUT_LEN]
    comment_author = comment.get("author", "unknown")

    try:
        # ── 1. Check if already replied ──
        already = await replies_log_col.find_one({
            "comment_id": comment_id,
            "user_id": user_id,
            "status": {"$in": ["posted", "approved_posted", "pending_review", "draft"]},
        })
        if already:
            return {"status": "skipped", "reason": "already_replied"}

        # ── 2. Load engagement settings ──
        settings = await engagement_settings_col.find_one({"user_id": user_id})
        if not settings:
            return {"status": "skipped", "reason": "no_engagement_settings"}

        reply_mode = settings.get("reply_mode", "automatic")

        # [SECURITY] Enforce daily reply cap inside the core function
        max_daily = settings.get("max_replies_per_day") or DEFAULT_MAX_DAILY_REPLIES
        if max_daily > 0:
            daily_count = await get_daily_reply_count(user_id)
            if daily_count >= max_daily:
                logger.info(f"🚫 Daily reply cap ({max_daily}) reached for user {user_id}")
                return {"status": "skipped", "reason": "daily_limit_reached"}

        # ── 3. Blank comment check ──
        if _is_blank_comment(comment_text):
            logger.info(f"🚫 Blank comment detected, skipping comment {comment_id}")
            await _log_reply(
                user_id, post_id, comment_id, comment_author, comment_text,
                "[SPAM: BLANK_COMMENT]", platform, 0, "spam_skipped",
            )
            return {"status": "spam_skipped", "reason": "blank_comment"}

        # ── 4. Spam check ──
        if await _is_spam(comment_text, settings):
            logger.info(f"🚫 Spam detected, skipping comment {comment_id}")
            await _log_reply(
                user_id, post_id, comment_id, comment_author,
                comment_text, "[SPAM DETECTED]", platform, 0, "spam_skipped",
            )
            return {"status": "spam_skipped"}

        # ── 5. Load Tone DNA ──
        tone = await get_active_tone_profile(user_id)
        if not tone:
            return {"status": "skipped", "reason": "no_tone_profile"}

        # ── 6. Load post context ──
        post_ctx = await get_post_context(post_id, user_id)
        video_cache_ctx = await _load_video_cache_context(user_id, post_id, post_ctx)
        fact_bundle = _build_fact_bundle(post_ctx, video_cache_ctx, comment_text)

        comment_type = "emoji_only" if _is_emoji_only_comment(comment_text) else "text"
        comment_tone = _infer_comment_tone(comment_text)

        # ── 7. Select model ──
        model = select_model_for_comment(comment_text)

        # ── 8. Build prompt & generate ──
        reply_text = await _generate_reply(
            tone_profile=tone,
            post_context=post_ctx,
            fact_bundle=fact_bundle,
            comment_text=comment_text,
            comment_author=comment_author,
            platform=platform,
            model=model,
            user_name=user_name,
            comment_type=comment_type,
            comment_tone=comment_tone,
        )

        if reply_text and comment_type == "emoji_only" and not _is_emoji_only_comment(reply_text):
            reply_text = _build_contextual_emoji_reply(post_ctx, tone)

        if reply_text and _is_disagreeing_reply(reply_text):
            reply_text = await _rewrite_non_disagreeing_reply(
                model=model,
                tone_profile=tone,
                post_context=post_ctx,
                fact_bundle=fact_bundle,
                comment_text=comment_text,
                comment_author=comment_author,
                platform=platform,
                user_name=user_name,
                draft_reply=reply_text,
                comment_type=comment_type,
                comment_tone=comment_tone,
            )

        if reply_text and fact_bundle.get("has_facts"):
            reply_text = await _fact_align_reply(
                model=FACT_CHECK_MODEL,
                comment_text=comment_text,
                comment_author=comment_author,
                draft_reply=reply_text,
                fact_bundle=fact_bundle,
                comment_type=comment_type,
            )

        # Final deterministic forbidden-term pass
        forbidden_terms = _get_exact_avoid_words(tone, tone.get("extracted_traits", {}))
        if reply_text and forbidden_terms and _contains_forbidden_terms(reply_text, forbidden_terms):
            reply_text = _hard_remove_forbidden_terms(reply_text, forbidden_terms)

        if not reply_text:
            await _log_reply(
                user_id, post_id, comment_id, comment_author,
                comment_text, "", platform, tone.get("version", 1), "failed",
            )
            return {"status": "failed", "reason": "llm_generation_failed"}

        # ── 9. Duplicate guard post-generation ──
        already_after_generation = await replies_log_col.find_one({
            "comment_id": comment_id,
            "user_id": user_id,
            "status": {"$in": ["posted", "approved_posted", "pending_review", "draft"]},
        })
        if already_after_generation:
            return {"status": "skipped", "reason": "already_replied"}

        tone_version = tone.get("version", 1)

        # ── 10. Handle based on reply_mode ──
        if reply_mode == "automatic":
            # [SECURITY] Always fetch token server-side — never trust caller's access_token
            auth = await _get_platform_auth(user_id, platform)
            if not auth:
                logger.warning(f"⚠️ No auth token for user {user_id} on {platform}, queuing for review")
                status = "pending_review"
            else:
                server_token = auth.get("accessToken", "")
                success = await post_reply(
                    platform=platform,
                    comment_id=comment_id,
                    reply_text=reply_text,
                    access_token=server_token,
                    **reply_kwargs,
                )
                status = "posted" if success else "failed"
                delay = random.uniform(MIN_REPLY_DELAY, MAX_REPLY_DELAY)
                await asyncio.sleep(delay)

        elif reply_mode == "review":
            status = "pending_review"

        elif reply_mode == "manual":
            status = "draft"

        else:
            status = "pending_review"  # Safe default

        # ── 11. Log ──
        log_doc = await _log_reply(
            user_id, post_id, comment_id, comment_author,
            comment_text, reply_text, platform, tone_version, status,
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

    [SECURITY] Validates log_id format, caps edited_text length, strips dangerous content.
    [SECURITY] Auth token always fetched server-side from DB.
    """
    # [SECURITY] Validate ObjectId format before touching the DB
    if not _is_valid_object_id(log_id):
        return {"success": False, "reason": "invalid_log_id"}

    # [SECURITY] Validate and sanitize edited_text
    if edited_text is not None:
        if not isinstance(edited_text, str):
            return {"success": False, "reason": "edited_text must be a string"}
        if len(edited_text) > MAX_EDITED_REPLY_LEN:
            return {"success": False, "reason": f"edited_text exceeds {MAX_EDITED_REPLY_LEN} character limit"}
        edited_text = _strip_script_content(edited_text.strip())
        if not edited_text:
            return {"success": False, "reason": "edited_text is empty after sanitization"}

    from bson import ObjectId
    try:
        doc = await replies_log_col.find_one({
            "_id": ObjectId(log_id),
            "user_id": user_id,
            "status": {"$in": ["pending_review", "draft"]},
        })
        if not doc:
            return {"success": False, "reason": "Reply not found or already processed"}

        reply_text = edited_text if edited_text else doc.get("generated_reply", "")
        platform = doc.get("platform", "")
        comment_id = doc.get("comment_id", "")

        # [SECURITY] Always fetch token server-side
        auth = await _get_platform_auth(user_id, platform)
        if not auth:
            return {"success": False, "reason": "No auth token for platform"}

        access_token = auth.get("accessToken", "")
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
                "generated_reply": reply_text,
                "approved_at": datetime.now(timezone.utc),
            }},
        )

        return {"success": success, "status": new_status}

    except Exception as e:
        logger.error(f"Error approving reply {log_id}: {e}")
        return {"success": False, "reason": str(e)}


async def reject_reply(log_id: str, user_id: str) -> Dict[str, Any]:
    """Reject a pending_review reply."""
    # [SECURITY] Validate ObjectId format before touching the DB
    if not _is_valid_object_id(log_id):
        return {"success": False, "reason": "invalid_log_id"}

    from bson import ObjectId
    try:
        result = await replies_log_col.update_one(
            {"_id": ObjectId(log_id), "user_id": user_id},
            {"$set": {"status": "rejected", "rejected_at": datetime.now(timezone.utc)}},
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
        fetch_limit = max(limit * 5, limit)
        cursor = replies_log_col.find(
            {"user_id": user_id, "status": status}
        ).sort("created_at", -1).limit(fetch_limit)

        items = []
        seen_comment_keys: set = set()
        async for doc in cursor:
            comment_key = str(doc.get("comment_id", "")).strip() or str(doc.get("_id", ""))
            if comment_key in seen_comment_keys:
                continue
            seen_comment_keys.add(comment_key)

            doc["_id"] = str(doc["_id"])
            doc["reply_text"] = doc.pop("generated_reply", "")
            doc.setdefault("model_used", "")
            for dt_key in ("created_at", "replied_at", "approved_at", "rejected_at"):
                val = doc.get(dt_key)
                if val and hasattr(val, "isoformat"):
                    doc[dt_key] = val.isoformat()
            if "replied_at" in doc:
                doc["posted_at"] = doc.get("replied_at")
            items.append(doc)
            if len(items) >= limit:
                break
        return items
    except Exception as e:
        logger.error(f"Error fetching reply queue for {user_id}: {e}")
        return []


async def get_reply_stats(user_id: str) -> Dict[str, Any]:
    """Get engagement reply statistics for a user."""
    try:
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        stats: Dict[str, int] = {}
        async for doc in replies_log_col.aggregate(pipeline):
            stats[doc["_id"]] = doc["count"]

        total = sum(stats.values())

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        replies_today = await replies_log_col.count_documents({
            "user_id": user_id,
            "status": {"$in": ["posted", "approved_posted"]},
            "created_at": {"$gte": today_start},
        })

        platform_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$platform", "count": {"$sum": 1}}},
        ]
        by_platform: Dict[str, int] = {}
        async for doc in replies_log_col.aggregate(platform_pipeline):
            if doc["_id"]:
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
            "total_replies": 0, "automatically_posted": 0, "pending_review": 0,
            "approved": 0, "rejected": 0, "draft": 0, "replies_today": 0, "by_platform": {},
        }


# ══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════

def _is_valid_object_id(value: str) -> bool:
    """[SECURITY] Validate MongoDB ObjectId format (24 hex chars)."""
    return bool(re.fullmatch(r"[0-9a-fA-F]{24}", value or ""))


def _strip_script_content(text: str) -> str:
    """[SECURITY] Remove HTML tags and javascript: references from user-edited text."""
    cleaned = re.sub(r"<[^>]{0,200}>", "", text)
    cleaned = re.sub(r"javascript\s*:", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


async def _get_platform_auth(user_id: str, platform: str) -> Optional[Dict[str, Any]]:
    """
    [SECURITY] Fetch access token for user+platform from DB.
    For Instagram, prefers instafb (Business Graph API).
    Never accepts tokens from callers.
    """
    auth = None
    if platform == "instagram":
        auth = await get_user_auth(user_id, "instafb")
    if not auth:
        auth = await get_user_auth(user_id, platform)
    return auth


def select_model_for_comment(comment_text: str) -> str:
    """
    Route to appropriate model based on comment complexity.

    [FINE-TUNE] Fixed bug: questions now always route to COMPLEX_MODEL.
    Previously, a question with ≤5 words would hit the first 'if' and return
    SIMPLE_MODEL before the 'has_question' check was ever reached.
    Now: complexity signals are evaluated first, short-circuit last.
    """
    words = comment_text.split()
    word_count = len(words)

    has_question = "?" in comment_text
    has_negation = any(neg in comment_text.lower() for neg in ["not", "don't", "can't", "won't", "shouldn't"])
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    exclamation_count = comment_text.count("!")
    has_multiple_sentences = comment_text.count(".") + comment_text.count("!") > 1

    # [FINE-TUNE] Evaluate COMPLEX triggers first — they always win regardless of length
    if (
        has_question
        or has_negation
        or word_count > 15
        or all_caps_words > 1
        or exclamation_count > 2
        or has_multiple_sentences
        or word_count >= 8
    ):
        return COMPLEX_MODEL

    # Simple: short, no complexity signals, no questions, no negation
    return SIMPLE_MODEL


def _build_system_prompt(
    tone_profile: Dict[str, Any],
    post_context: Optional[Dict[str, Any]],
    fact_bundle: Dict[str, Any],
    platform: str,
    comment_type: str,
    comment_tone: str,
    user_name: str,
    extra_rules: str = "",
) -> str:
    """
    [FINE-TUNE] Extracted shared helper — eliminates the 70-line duplicated
    .format() block that existed in both _generate_reply and _rewrite_non_disagreeing_reply.
    Both functions now call this single source of truth.
    """
    traits = tone_profile.get("extracted_traits", {})
    avoid_words_exact = _get_exact_avoid_words(tone_profile, traits)
    avoid_topics = _normalize_phrase_list(traits.get("avoid_topics", []))
    if not avoid_topics and avoid_words_exact:
        avoid_topics = list(avoid_words_exact)

    prompt_template = REPLY_SYSTEM_PROMPT
    if extra_rules:
        prompt_template = prompt_template + f"\n{extra_rules}"

    return prompt_template.format(
        user_name=user_name,
        tone_label=tone_profile.get("tone_label", "Friendly"),
        formality=traits.get("formality", 0.3),
        emoji_frequency=traits.get("emoji_frequency", 0.5),
        avg_reply_length=traits.get("avg_reply_length_words", 18),
        humor_level=traits.get("humor_level", "subtle"),
        confrontation_style=traits.get("confrontation_style", "polite_correction"),
        signature_phrases=", ".join(traits.get("signature_phrases", [])) or "none",
        avoid_words_exact=", ".join(avoid_words_exact) or "none",
        avoid_topics=", ".join(avoid_topics) or "none",
        content_summary=post_context.get("content_summary", "Social media post") if post_context else "Social media post",
        key_topics=", ".join(post_context.get("key_topics", [])) if post_context else "general",
        post_sentiment=post_context.get("sentiment", "neutral") if post_context else "neutral",
        likely_comment_themes=", ".join(post_context.get("likely_comment_themes", [])) if post_context else "general audience reactions",
        platform=platform,
        comment_type=comment_type,
        comment_tone=comment_tone,
        video_summary=fact_bundle.get("video_summary", "none"),
        transcript_highlights=fact_bundle.get("transcript_highlights", "none"),
        fact_anchors=fact_bundle.get("fact_anchors", "none"),
        lexical_sync_words=fact_bundle.get("lexical_sync_words", "none"),
    )


async def _generate_reply(
    tone_profile: Dict[str, Any],
    post_context: Optional[Dict[str, Any]],
    fact_bundle: Optional[Dict[str, Any]],
    comment_text: str,
    comment_author: str,
    platform: str,
    model: str,
    user_name: str = "the user",
    comment_type: str = "text",
    comment_tone: str = "neutral",
) -> str:
    """
    Build the LLM prompt and generate the reply.

    [FINE-TUNE] Temperature is now dynamic based on comment_tone:
    - critical comments → 0.5 (more controlled, careful output)
    - positive comments → 0.8 (slightly more expressive/warm)
    - neutral/curious   → 0.7 (balanced default)
    """
    fact_bundle = fact_bundle or {}
    system_msg = _build_system_prompt(
        tone_profile, post_context, fact_bundle,
        platform, comment_type, comment_tone, user_name,
    )

    user_prompt = f'Reply to this comment from @{comment_author}: "{comment_text}"'

    # [FINE-TUNE] Dynamic temperature by comment tone
    temp_map = {"critical": 0.5, "positive": 0.8, "curious": 0.65}
    temperature = temp_map.get(comment_tone, 0.7)

    reply = await groq_generate_text(
        model=model,
        prompt=user_prompt,
        system_msg=system_msg,
        temperature=temperature,
        max_completion_tokens=160,
    )

    if reply:
        reply = reply.strip().strip('"').strip("'")
        for prefix in ["Reply:", "Response:", "Here's my reply:", "My reply:"]:
            if reply.lower().startswith(prefix.lower()):
                reply = reply[len(prefix):].strip()

        avoid_words_exact = _get_exact_avoid_words(
            tone_profile, tone_profile.get("extracted_traits", {})
        )
        if avoid_words_exact and _contains_forbidden_terms(reply, avoid_words_exact):
            reply = _hard_remove_forbidden_terms(reply, avoid_words_exact)

    return reply


async def _rewrite_non_disagreeing_reply(
    model: str,
    tone_profile: Dict[str, Any],
    post_context: Optional[Dict[str, Any]],
    fact_bundle: Optional[Dict[str, Any]],
    comment_text: str,
    comment_author: str,
    platform: str,
    user_name: str,
    draft_reply: str,
    comment_type: str,
    comment_tone: str,
) -> str:
    """
    One-pass rewrite when a draft sounds contradictory or argumentative.

    [FINE-TUNE] Now uses _build_system_prompt() instead of duplicating the format block.
    """
    fact_bundle = fact_bundle or {}
    rewrite_system = _build_system_prompt(
        tone_profile, post_context, fact_bundle,
        platform, comment_type, comment_tone, user_name,
        extra_rules="15. Never include correctional language like 'you are wrong', 'actually', or 'I disagree'.",
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
        temperature=0.5,
        max_completion_tokens=130,
    )

    if rewritten:
        cleaned = rewritten.strip().strip('"').strip("'")
        if not _is_disagreeing_reply(cleaned):
            return cleaned
    return draft_reply


async def _fact_align_reply(
    model: str,
    comment_text: str,
    comment_author: str,
    draft_reply: str,
    fact_bundle: Dict[str, Any],
    comment_type: str,
) -> str:
    """Final fact-consistency pass using post/video context."""
    if not fact_bundle.get("has_facts"):
        return draft_reply

    prompt = (
        f'Comment from @{comment_author}: "{comment_text}"\n'
        f'Draft reply: "{draft_reply}"\n\n'
        "Verified context:\n"
        f"- Video summary: {fact_bundle.get('video_summary', 'none')}\n"
        f"- Transcript highlights: {fact_bundle.get('transcript_highlights', 'none')}\n"
        f"- Anchors: {fact_bundle.get('fact_anchors', 'none')}\n"
        f"- Lexical sync words: {fact_bundle.get('lexical_sync_words', 'none')}\n\n"
        "Task:\n"
        "Rewrite only if needed so the reply is fact-consistent with verified context, supportive/non-argumentative, and naturally uses sync words. "
        "If already fine, return the draft reply unchanged.\n"
        "Return reply text only."
    )

    rewrite = await groq_generate_text(
        model=model,
        prompt=prompt,
        system_msg=(
            "You are a factual safety layer for social reply generation. "
            "Never add facts not present in verified context. "
            "Never disagree with the commenter; keep a warm, neutral, or appreciative tone. "
            "For emoji_only comments, output must remain emoji-only."
        ),
        temperature=0.2,
        max_completion_tokens=140,
    )

    if not rewrite:
        return draft_reply

    cleaned = rewrite.strip().strip('"').strip("'")
    if comment_type == "emoji_only" and not _is_emoji_only_comment(cleaned):
        return draft_reply
    if _is_disagreeing_reply(cleaned):
        return draft_reply
    return cleaned or draft_reply


def _normalize_phrase_list(value: Any) -> List[str]:
    """Normalize list-like avoid/signature fields into unique trimmed phrases."""
    if value is None:
        return []
    raw_items: List[Any]
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value)
        raw_items = [part for part in text.replace("\n", ",").split(",")]

    cleaned: List[str] = []
    seen: set = set()
    for item in raw_items:
        token = str(item).strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(token)
    return cleaned


def _get_exact_avoid_words(tone_profile: Dict[str, Any], traits: Dict[str, Any]) -> List[str]:
    form_data = tone_profile.get("form_data", {}) if tone_profile else {}
    form_avoid = _normalize_phrase_list(form_data.get("avoid_words", []))
    if form_avoid:
        return form_avoid
    return _normalize_phrase_list((traits or {}).get("avoid_topics", []))


def _contains_forbidden_terms(text: str, forbidden_terms: List[str]) -> bool:
    if not text:
        return False
    for term in forbidden_terms:
        term = term.strip()
        if not term:
            continue
        if re.search(_build_forbidden_pattern(term), text, flags=re.IGNORECASE):
            return True
    return False


def _hard_remove_forbidden_terms(text: str, forbidden_terms: List[str]) -> str:
    cleaned = text or ""
    for term in forbidden_terms:
        term = term.strip()
        if not term:
            continue
        cleaned = re.sub(_build_forbidden_pattern(term), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = cleaned.strip(" \t\n\r-,:;")
    return cleaned or "Noted."


def _build_forbidden_pattern(term: str) -> str:
    escaped = re.escape(term)
    if " " not in term and re.fullmatch(r"[A-Za-z0-9_']+", term):
        return rf"\b{escaped}\b"
    return escaped


def _normalize_for_spam(text: str) -> str:
    """
    [SECURITY] Normalize Unicode and strip lookalike characters before spam matching.
    Prevents bypasses like '𝒷𝓊𝔂 followers' slipping through keyword filters.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


async def _is_spam(comment_text: str, settings: Dict[str, Any]) -> bool:
    """
    Check if a comment is spam using keyword matching.
    Combines default keywords + user's blacklisted words.

    [SECURITY] Normalizes Unicode before matching to prevent lookalike bypasses.
    """
    if _is_blank_comment(comment_text):
        return True

    if not settings.get("spam_filter_enabled", True):
        return False

    # [SECURITY] Normalize before matching
    normalized_text = _normalize_for_spam(comment_text)
    raw_text_lower = (comment_text or "").lower()

    all_keywords = _normalize_spam_keywords(
        DEFAULT_SPAM_KEYWORDS + _normalize_spam_keywords(settings.get("blacklisted_words", []))
    )
    for keyword in all_keywords:
        normalized_keyword = _normalize_for_spam(keyword)

        # Critical fix: never allow empty normalized tokens to match everything.
        if normalized_keyword:
            if normalized_keyword in normalized_text:
                return True
            continue

        # Emoji/symbol keywords may normalize to empty; match against raw text.
        if keyword.lower() in raw_text_lower:
            return True

    # Excessive emoji burst check (≥8 emojis, no text)
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]'
    )
    text_no_emoji = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF]+',
        '', comment_text,
    ).strip()
    emoji_count = len(emoji_pattern.findall(comment_text))
    if not text_no_emoji and emoji_count >= 8:
        return True

    return False


def _normalize_spam_keywords(value: Any) -> List[str]:
    """Normalize blacklist values into a clean, unique list of keywords."""
    if value is None:
        return []

    raw_items: List[Any]
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value)
        raw_items = [part for part in text.replace("\n", ",").split(",")]

    cleaned: List[str] = []
    seen = set()
    for item in raw_items:
        token = str(item).strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(token)

    return cleaned


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
    """
    [FINE-TUNE] Positive/negative checks now run before '?' check.
    Previously a comment like "this is amazing, what do you think?" always
    returned 'curious' even though it's clearly positive-curious. Now it
    correctly returns 'positive'.
    """
    text = (comment_text or "").lower()
    if not text:
        return "neutral"
    # Check sentiment first — more informative signal than punctuation
    if any(w in text for w in ["love", "awesome", "great", "amazing", "🔥", "😍", "👏"]):
        return "positive"
    if any(w in text for w in ["bad", "hate", "worst", "boring", "terrible", "not good"]):
        return "critical"
    if "?" in text:
        return "curious"
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


async def _load_video_cache_context(
    user_id: str,
    post_id: str,
    post_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Best-effort lookup of video cache for this post."""
    media_type = (post_context or {}).get("media_type", "")
    if media_type and media_type.lower() not in ("video", "reel", "reels"):
        return None

    post_doc = None
    try:
        from bson import ObjectId
        try:
            post_doc = await scheduled_posts_col.find_one(
                {"_id": ObjectId(post_id), "userId": user_id},
                {"videoHash": 1, "mediaType": 1},
            )
        except Exception:
            post_doc = None
    except Exception:
        post_doc = None

    if not post_doc:
        post_doc = await scheduled_posts_col.find_one(
            {"postId": post_id, "userId": user_id},
            {"videoHash": 1, "mediaType": 1},
        )

    video_hash = (post_doc or {}).get("videoHash") or (post_context or {}).get("video_hash")
    if not video_hash:
        return None

    return await get_cached_video_analysis(video_hash)


def _build_fact_bundle(
    post_context: Optional[Dict[str, Any]],
    video_cache: Optional[Dict[str, Any]],
    comment_text: str,
) -> Dict[str, Any]:
    """Create compact fact/context payload for grounded reply generation."""
    summary = (post_context or {}).get("content_summary", "").strip()
    transcript = (video_cache or {}).get("transcript", "").strip()
    visual_summary = (video_cache or {}).get("visual_summary", "").strip()
    objects = (video_cache or {}).get("objects", [])[:6]
    actions = (video_cache or {}).get("actions", [])[:6]
    person = (video_cache or {}).get("detected_person")

    anchors = []
    if summary:
        anchors.append(f"Post: {summary}")
    if visual_summary:
        anchors.append(f"Visual: {visual_summary[:180]}")
    if person:
        anchors.append(f"Person: {person}")
    if objects:
        anchors.append("Objects: " + ", ".join(str(x) for x in objects))
    if actions:
        anchors.append("Actions: " + ", ".join(str(x) for x in actions))

    sync_words = _extract_sync_words(comment_text, " ".join(anchors), transcript)

    return {
        "has_facts": bool(summary or transcript or visual_summary or anchors),
        "video_summary": visual_summary or summary or "none",
        "transcript_highlights": _first_sentences(transcript, max_chars=280) or "none",
        "fact_anchors": " | ".join(anchors[:5]) if anchors else "none",
        "lexical_sync_words": ", ".join(sync_words) if sync_words else "none",
    }


def _extract_sync_words(comment_text: str, anchors_text: str, transcript_text: str) -> List[str]:
    """Pick high-signal shared words to keep phrasing in sync with comment + video context."""
    comment_words = _extract_keywords(comment_text)
    video_words = _extract_keywords(f"{anchors_text} {transcript_text[:500]}")
    video_set = set(video_words)

    shared = [w for w in comment_words if w in video_set]
    if shared:
        return shared[:6]

    merged = comment_words + [w for w in video_words if w not in comment_words]
    return merged[:6]


def _extract_keywords(text: str) -> List[str]:
    stop_words = {
        "this", "that", "with", "from", "your", "have", "about", "just",
        "really", "very", "there", "their", "them", "they", "what", "when",
        "where", "which", "will", "would", "could", "should", "into", "also",
        "thanks", "thank", "great", "awesome", "video", "post", "comment",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z']{2,}", (text or "").lower())
    deduped: List[str] = []
    seen: set = set()
    for w in words:
        if w in stop_words or w in seen:
            continue
        seen.add(w)
        deduped.append(w)
    return deduped


def _first_sentences(text: str, max_chars: int = 280) -> str:
    if not text:
        return ""
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    out: List[str] = []
    length = 0
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        next_len = length + len(c) + (1 if out else 0)
        if next_len > max_chars:
            break
        out.append(c)
        length = next_len
    if out:
        return " ".join(out)
    return text[:max_chars].strip()


def _build_contextual_emoji_reply(
    post_context: Optional[Dict[str, Any]],
    tone_profile: Optional[Dict[str, Any]],
) -> str:
    """
    Build a context-aware emoji-only fallback reply.

    [FINE-TUNE] Uses random.sample instead of random.choice in a loop,
    preventing repeated emoji like 🔥🔥🔥 in the output.
    """
    context_text = ""
    if post_context:
        context_text = (
            f"{post_context.get('content_summary', '')} "
            f"{' '.join(post_context.get('key_topics', []))} "
            f"{post_context.get('sentiment', '')}"
        ).lower()

    topic_map = [
        (["workout", "fitness", "gym", "training"], ["💪", "🔥", "🏋️", "⚡"]),
        (["food", "recipe", "cooking", "meal"], ["😋", "🍽️", "🔥", "👨‍🍳"]),
        (["travel", "trip", "vacation", "adventure"], ["✈️", "🌍", "✨", "🗺️"]),
        (["music", "song", "beat", "dance"], ["🎵", "🎶", "💃", "🎤"]),
        (["business", "startup", "marketing", "sales"], ["📈", "🚀", "🙌", "💡"]),
        (["tech", "ai", "code", "software"], ["🤖", "💡", "🚀", "⚙️"]),
        (["fashion", "style", "outfit", "beauty"], ["✨", "💅", "😍", "👗"]),
        (["pet", "dog", "cat", "puppy"], ["🐾", "❤️", "🥹", "🐶"]),
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

    # [FINE-TUNE] random.sample prevents duplicate emojis in the output
    count = min(count, len(emoji_pool))
    chosen = random.sample(emoji_pool, count)
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
        "comment_text": comment_text,   # stored as value only — never used in queries
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
