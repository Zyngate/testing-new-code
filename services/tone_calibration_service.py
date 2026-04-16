# stelle_backend/services/tone_calibration_service.py
"""
Tone Calibration Service — Phase 1 of Autonomous Engagement.

Handles:
- Saving user tone form data (Method A)
- Extracting Tone DNA via Groq LLM from form input
- Fetching the active tone profile for a user
- Resetting (archiving old, creating new) tone profiles

CHANGES v2:
- [SECURITY]  Input validation: max field lengths enforced on all form_data fields
- [SECURITY]  user_id validated as non-empty, alphanumeric-safe string before any DB call
- [SECURITY]  Signature phrases capped at 20 items; avoid_words capped at 50 items
- [SECURITY]  additional_notes stripped of HTML/script tags before storage/LLM injection
- [SECURITY]  update_engagement_settings: type-validates every allowed field before write
- [SECURITY]  blacklisted_words capped at 200 items, each capped at 100 chars
- [FINE-TUNE] Tone extraction retried once on LLM parse failure before falling back to defaults
- [FINE-TUNE] _sanitize_extracted_traits: formality blended from both style + language (not one-wins)
- [FINE-TUNE] _default_traits_from_form: question_rate now varies by style (not always 0.2)
- [FINE-TUNE] _build_tone_dna: directness now uses formality (more meaningful) instead of exclamation_rate
- [FINE-TUNE] TONE_EXTRACTION_PROMPT_TEMPLATE: clarified emoji/length mappings so LLM guesses less
- [FINE-TUNE] get_active_tone_profile: returns lightweight dict without calibration_pairs by default
"""

import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from database import db
from config import logger
from services.ai_service import groq_generate_text


# ── Collections ──────────────────────────────────────────────
tone_profiles_col = db["user_tone_profiles"]
engagement_settings_col = db["engagement_settings"]


# ── Constants ────────────────────────────────────────────────
TONE_MODEL = "llama-3.3-70b-versatile"

# [SECURITY] Field-level length caps for form_data inputs
_MAX_STYLE_LEN = 50
_MAX_EMOJI_USAGE_LEN = 20
_MAX_REPLY_LENGTH_LEN = 30
_MAX_LANGUAGE_LEN = 50
_MAX_PHRASE_LEN = 100
_MAX_SIGNATURE_PHRASES = 20
_MAX_AVOID_WORDS = 50
_MAX_NOTES_LEN = 1000

# [SECURITY] Valid enum values — reject anything else before it reaches the LLM prompt
_VALID_STYLES = {"casual", "friendly", "warm", "witty", "professional", "blunt"}
_VALID_EMOJI_USAGE = {"none", "moderate", "heavy"}
_VALID_REPLY_LENGTHS = {"short_punchy", "medium", "detailed"}
_VALID_LANGUAGES = {"gen_z", "friendly", "business", "technical"}

TONE_EXTRACTION_SYSTEM_MSG = """You are a tone analysis expert. 
Given a user's self-described communication preferences, extract a precise 
Tone DNA profile as JSON. Be accurate and specific. Only return valid JSON."""

# [FINE-TUNE] Added explicit emoji/length value guidance so LLM doesn't guess
TONE_EXTRACTION_PROMPT_TEMPLATE = """
Analyze the following user-provided communication preferences and generate a Tone DNA profile.

USER FORM DATA:
- Communication Style: {style}
- Emoji Usage: {emoji_usage}  (none=0.0, moderate=0.5, heavy=0.8)
- Response Length Preference: {reply_length}  (short_punchy≈12 words, medium≈22, detailed≈40)
- Language/Slang: {language}
- Signature Phrases: {signature_phrases}
- Words/Topics to Avoid: {avoid_words}
- Additional Notes: {additional_notes}

Important constraints:
- Keep signature_phrases strictly grounded in user input. Do not invent new phrases.
- Keep avoid_topics strictly grounded in user input. Do not add inferred topics.
- Return ONLY valid JSON, with no markdown fences, no extra text.

Produce this exact JSON structure:
{{
  "tone_label": "<2-4 word label e.g. 'Warm & Witty Professional'>",
  "avg_reply_length_words": <integer 5-60 based on reply_length>,
  "emoji_frequency": <0.0-1.0 float based on emoji_usage hint above>,
  "formality": <0.0=very casual, 1.0=very formal>,
  "humor_level": "<none|subtle|moderate|heavy>",
  "gratitude_pattern": <true if style suggests thankfulness, else false>,
  "confrontation_style": "<ignore|deflect_with_humor|polite_correction|direct_rebuttal>",
  "exclamation_rate": <0.0-1.0>,
  "question_rate": <0.0-1.0 how often they ask questions back>,
  "signature_phrases": [<list of phrases from user input only>],
  "vocabulary_level": "<simple|conversational|sophisticated|technical>",
  "avoid_topics": [<from user input only>]
}}
"""


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

async def get_active_tone_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the currently active tone profile for a user.
    Returns None if no profile exists.

    [FINE-TUNE] Excludes large calibration_pairs array from projection —
    engagement_service never needs it and it bloats every reply cycle.
    """
    _validate_user_id(user_id)
    try:
        doc = await tone_profiles_col.find_one(
            {"user_id": user_id, "is_active": True},
            projection={"calibration_pairs": 0},  # not needed at reply time
            sort=[("updated_at", -1), ("created_at", -1)],
        )
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"Error fetching tone profile for {user_id}: {e}")
        return None


async def get_tone_status(user_id: str) -> Dict[str, Any]:
    """
    Check if a user has completed tone setup.
    Returns status info for frontend to decide what to show.
    Field names match the frontend ToneStatus interface:
      is_calibrated, calibrated_at, tone_label, profile_id
    """
    _validate_user_id(user_id)
    profile = await get_active_tone_profile(user_id)
    if profile:
        created = profile.get("created_at", "")
        if hasattr(created, "isoformat"):
            created = created.isoformat()
        return {
            "is_calibrated": True,
            "calibrated_at": created,
            "tone_label": profile.get("tone_label", "Unknown"),
            "profile_id": profile.get("_id", ""),
        }
    return {
        "is_calibrated": False,
        "calibrated_at": None,
        "tone_label": None,
        "profile_id": None,
    }


async def calibrate_tone_from_form(
    user_id: str,
    form_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Method A — Tone Calibration via Form.

    1. Validates and sanitizes form_data inputs
    2. Sends it to Groq LLM to extract Tone DNA traits (retries once on parse failure)
    3. Upserts complete profile in user_tone_profiles
    4. Returns the saved profile

    form_data expected keys:
      - style: str              (e.g. "casual", "professional", "witty", "warm", "blunt")
      - emoji_usage: str        (e.g. "heavy", "moderate", "none")
      - reply_length: str       (e.g. "short_punchy", "medium", "detailed")
      - language: str           (e.g. "gen_z", "business", "friendly", "technical")
      - signature_phrases: list[str] | str
      - avoid_words: list[str] | str
      - additional_notes: str   (optional free text)
    """
    _validate_user_id(user_id)

    # [SECURITY] Sanitize all incoming form data before any processing
    form_data = _sanitize_form_data_input(form_data)

    try:
        now = datetime.now(timezone.utc)

        # 1. Find most recent profile so recalibration updates existing entry
        current = await tone_profiles_col.find_one(
            {"user_id": user_id},
            sort=[("updated_at", -1), ("created_at", -1)],
        )
        new_version = (current.get("version", 0) + 1) if current else 1

        # Merge partial recalibration payload with previous form data
        effective_form_data = _merge_form_data_with_current(form_data, current)

        # Normalize list-like fields for consistent storage
        normalized_signature_phrases = _normalize_text_list(
            effective_form_data.get("signature_phrases", []),
            max_items=_MAX_SIGNATURE_PHRASES,
            max_item_len=_MAX_PHRASE_LEN,
        )
        normalized_avoid_words = _normalize_text_list(
            effective_form_data.get("avoid_words", []),
            max_items=_MAX_AVOID_WORDS,
            max_item_len=_MAX_PHRASE_LEN,
        )
        effective_form_data["signature_phrases"] = normalized_signature_phrases
        effective_form_data["avoid_words"] = normalized_avoid_words

        # 2. Extract Tone DNA via LLM (with one retry on parse failure)
        extracted_traits = await _extract_tone_from_form(effective_form_data)
        extracted_traits = _sanitize_extracted_traits(extracted_traits, effective_form_data)

        # 3. Build the profile payload
        created_at = current.get("created_at", now) if current else now
        profile_doc = {
            "user_id": user_id,
            "version": new_version,
            "tone_label": extracted_traits.get("tone_label", "Custom Tone"),
            "calibration_method": "form",
            "form_data": {
                "style": effective_form_data.get("style", ""),
                "emoji_usage": effective_form_data.get("emoji_usage", ""),
                "reply_length": effective_form_data.get("reply_length", ""),
                "language": effective_form_data.get("language", ""),
                "signature_phrases": normalized_signature_phrases,
                "avoid_words": normalized_avoid_words,
                "additional_notes": effective_form_data.get("additional_notes", ""),
            },
            "extracted_traits": extracted_traits,
            "calibration_pairs": [],
            "is_active": True,
            "created_at": created_at,
            "updated_at": now,
        }

        # 4. Update existing or insert new profile
        if current:
            target_id = current["_id"]
            await tone_profiles_col.update_one(
                {"_id": target_id},
                {
                    "$set": profile_doc,
                    "$unset": {"archived_at": ""},
                },
            )
            # Defensive cleanup: keep exactly one active profile per user
            await tone_profiles_col.update_many(
                {"user_id": user_id, "_id": {"$ne": target_id}},
                {"$set": {"is_active": False, "archived_at": now, "updated_at": now}},
            )
            saved_profile = await tone_profiles_col.find_one({"_id": target_id})
            logger.info(f"✅ Tone profile updated for user {user_id} (v{new_version}) via form")
        else:
            result = await tone_profiles_col.insert_one(profile_doc)
            saved_profile = await tone_profiles_col.find_one({"_id": result.inserted_id})
            logger.info(f"✅ Tone profile created for user {user_id} (v{new_version}) via form")

        if not saved_profile:
            raise RuntimeError("Failed to load saved tone profile")

        saved_profile["_id"] = str(saved_profile["_id"])

        # 5. Ensure engagement_settings exists with defaults
        await _ensure_engagement_settings(user_id)

        # 6. Build tone_dna object matching frontend ToneDNA interface
        saved_profile["tone_dna"] = _build_tone_dna(extracted_traits)

        # Convert datetime fields to ISO strings
        for dt_key in ("created_at", "updated_at"):
            if hasattr(saved_profile.get(dt_key), "isoformat"):
                saved_profile[dt_key] = saved_profile[dt_key].isoformat()

        return saved_profile

    except Exception as e:
        logger.error(f"❌ Error calibrating tone for {user_id}: {e}", exc_info=True)
        raise


async def reset_tone(user_id: str) -> Dict[str, Any]:
    """Archive the current tone profile and return status."""
    _validate_user_id(user_id)
    try:
        result = await tone_profiles_col.update_many(
            {"user_id": user_id, "is_active": True},
            {"$set": {"is_active": False, "archived_at": datetime.now(timezone.utc)}},
        )
        archived_count = result.modified_count
        logger.info(f"🔄 Tone reset for user {user_id} — archived {archived_count} profile(s)")
        return {
            "success": True,
            "archived_count": archived_count,
            "message": "Tone profile archived. Please complete calibration again.",
        }
    except Exception as e:
        logger.error(f"❌ Error resetting tone for {user_id}: {e}")
        raise


async def get_engagement_settings(user_id: str) -> Dict[str, Any]:
    """
    Fetch engagement settings for a user (or create defaults).
    Maps internal DB field names to frontend-expected field names.
    """
    _validate_user_id(user_id)
    await _ensure_engagement_settings(user_id)
    doc = await engagement_settings_col.find_one({"user_id": user_id})
    if doc:
        doc["_id"] = str(doc["_id"])
        doc["enabled"] = doc.pop("auto_reply_enabled", False)
        doc["daily_reply_limit"] = 0
        doc["max_replies_per_post"] = 0
        doc.setdefault("reply_window_hours", 10)
        if isinstance(doc.get("platforms"), list):
            doc["platforms"] = [
                p.lower().strip()
                for p in doc["platforms"]
                if isinstance(p, str) and p.strip()
            ]
        else:
            doc["platforms"] = []
        for dt_key in ("created_at", "updated_at"):
            if hasattr(doc.get(dt_key), "isoformat"):
                doc[dt_key] = doc[dt_key].isoformat()
    return doc


async def update_engagement_settings(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update engagement settings for a user.
    Only selected fields are user-configurable.

    [SECURITY] Every allowed field is now type-validated and range-clamped
    before writing to DB. Unknown fields are silently dropped.
    """
    _validate_user_id(user_id)

    # Map frontend field names to backend field names
    field_mapping = {"enabled": "auto_reply_enabled"}
    mapped_updates: Dict[str, Any] = {}
    for k, v in updates.items():
        backend_key = field_mapping.get(k, k)
        mapped_updates[backend_key] = v

    allowed_fields = {
        "auto_reply_enabled", "reply_mode",
        "platforms", "spam_filter_enabled", "blacklisted_words",
        "reply_window_hours",
    }
    filtered: Dict[str, Any] = {}

    for k, v in mapped_updates.items():
        if k not in allowed_fields:
            continue

        # [SECURITY] Per-field type + value validation
        if k == "auto_reply_enabled":
            if not isinstance(v, bool):
                continue
            filtered[k] = v

        elif k == "reply_mode":
            if v not in {"automatic", "review", "manual"}:
                logger.warning(f"Invalid reply_mode '{v}' from user {user_id}, skipping")
                continue
            filtered[k] = v

        elif k == "platforms":
            if not isinstance(v, list):
                continue
            valid_platforms = {
                "instagram",
                "threads",
                "tiktok",
                "youtube",
                "twitter",
                "facebook",
                "linkedin",
            }
            normalized = [
                p.lower().strip()
                for p in v
                if isinstance(p, str) and p.lower().strip() in valid_platforms
            ]
            # Keep order while removing duplicates.
            filtered[k] = list(dict.fromkeys(normalized))

        elif k == "spam_filter_enabled":
            if not isinstance(v, bool):
                continue
            filtered[k] = v

        elif k == "blacklisted_words":
            if not isinstance(v, list):
                continue
            # [SECURITY] Cap list size and individual word length
            cleaned_words = [
                str(w).strip()[:100]
                for w in v
                if isinstance(w, str) and str(w).strip()
            ][:200]
            filtered[k] = cleaned_words

        elif k == "reply_window_hours":
            try:
                filtered[k] = max(1, min(168, int(v)))
            except (TypeError, ValueError):
                continue

    if not filtered:
        return await get_engagement_settings(user_id)

    filtered["updated_at"] = datetime.now(timezone.utc)
    await engagement_settings_col.update_one(
        {"user_id": user_id},
        {"$set": filtered},
        upsert=True,
    )

    return await get_engagement_settings(user_id)


# ══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════

def _validate_user_id(user_id: str) -> None:
    """
    [SECURITY] Raise ValueError for empty or suspicious user_id values.
    Prevents blank-string DB queries and basic injection attempts.
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string")
    if len(user_id) > 128:
        raise ValueError("user_id exceeds maximum length")
    # Allow alphanumeric, hyphens, underscores, dots — covers UUIDs and ObjectId hex strings
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", user_id):
        raise ValueError(f"user_id contains invalid characters: {user_id!r}")


def _sanitize_form_data_input(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    [SECURITY] Sanitize and validate all user-supplied form fields.
    - Coerces types, enforces length caps, strips dangerous content.
    - Drops fields with invalid enum values and replaces with safe defaults.
    """
    def _safe_str(value: Any, max_len: int, default: str = "") -> str:
        text = str(value).strip() if value is not None else ""
        return _strip_html(text)[:max_len] or default

    def _safe_enum(value: Any, valid_set: set, default: str) -> str:
        raw = str(value).strip().lower() if value is not None else ""
        return raw if raw in valid_set else default

    sanitized: Dict[str, Any] = {}

    sanitized["style"] = _safe_enum(
        form_data.get("style"), _VALID_STYLES, "friendly"
    )
    sanitized["emoji_usage"] = _safe_enum(
        form_data.get("emoji_usage"), _VALID_EMOJI_USAGE, "moderate"
    )
    sanitized["reply_length"] = _safe_enum(
        form_data.get("reply_length"), _VALID_REPLY_LENGTHS, "medium"
    )
    sanitized["language"] = _safe_enum(
        form_data.get("language"), _VALID_LANGUAGES, "friendly"
    )
    sanitized["additional_notes"] = _safe_str(
        form_data.get("additional_notes", ""), _MAX_NOTES_LEN
    )

    # Lists — normalize first, then cap
    sanitized["signature_phrases"] = _normalize_text_list(
        form_data.get("signature_phrases", []),
        max_items=_MAX_SIGNATURE_PHRASES,
        max_item_len=_MAX_PHRASE_LEN,
    )
    sanitized["avoid_words"] = _normalize_text_list(
        form_data.get("avoid_words", []),
        max_items=_MAX_AVOID_WORDS,
        max_item_len=_MAX_PHRASE_LEN,
    )

    return sanitized


def _strip_html(text: str) -> str:
    """
    [SECURITY] Remove HTML tags and script-like patterns from text.
    Prevents prompt injection via <script> or HTML payloads in free-text fields.
    """
    # Remove HTML/XML tags
    cleaned = re.sub(r"<[^>]{0,200}>", "", text)
    # Remove javascript: protocol references
    cleaned = re.sub(r"javascript\s*:", "", cleaned, flags=re.IGNORECASE)
    # Normalize unicode to ASCII-safe form (prevents lookalike bypass)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    return cleaned.strip()


async def _extract_tone_from_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send form data to Groq LLM and extract structured Tone DNA traits.
    [FINE-TUNE] Retries once on JSON parse failure before falling back to defaults.
    Falls back to rule-based defaults if both attempts fail.
    """
    prompt = TONE_EXTRACTION_PROMPT_TEMPLATE.format(
        style=form_data.get("style", "friendly"),
        emoji_usage=form_data.get("emoji_usage", "moderate"),
        reply_length=form_data.get("reply_length", "medium"),
        language=form_data.get("language", "friendly"),
        signature_phrases=_join_text_list_for_prompt(form_data.get("signature_phrases", [])),
        avoid_words=_join_text_list_for_prompt(form_data.get("avoid_words", [])),
        additional_notes=form_data.get("additional_notes", "none"),
    )

    for attempt in range(2):  # [FINE-TUNE] retry once on parse failure
        raw = await groq_generate_text(
            model=TONE_MODEL,
            prompt=prompt,
            system_msg=TONE_EXTRACTION_SYSTEM_MSG,
            temperature=0.3 if attempt == 1 else 0.4,  # lower temp on retry
            max_completion_tokens=600,
        )
        try:
            traits = _parse_llm_json(raw)
            logger.info(f"✅ Tone DNA extracted (attempt {attempt + 1}): {traits.get('tone_label', 'Unknown')}")
            return traits
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"⚠️ Tone LLM parse failed (attempt {attempt + 1}): {e}")

    logger.warning("⚠️ Both LLM attempts failed — using rule-based defaults")
    return _default_traits_from_form(form_data)


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM output, stripping markdown fences if present.
    Raises json.JSONDecodeError on failure.
    """
    cleaned = (raw or "").strip()
    # Strip opening fence
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    # Strip closing fence
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    # Strip "json" language tag sometimes appended by LLMs
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def _default_traits_from_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback: build sensible traits from form data without LLM.
    Used when LLM extraction fails both attempts.

    [FINE-TUNE] question_rate now varies by style instead of always returning 0.2.
    """
    style = form_data.get("style", "friendly").lower()
    emoji = form_data.get("emoji_usage", "moderate").lower()
    length = form_data.get("reply_length", "medium").lower()
    language = form_data.get("language", "friendly").lower()

    emoji_map = {"heavy": 0.8, "moderate": 0.5, "none": 0.0}
    length_map = {"short_punchy": 12, "medium": 22, "detailed": 40}
    formality_map = {
        "casual": 0.2, "friendly": 0.3, "warm": 0.3,
        "witty": 0.3, "professional": 0.7, "blunt": 0.5,
        "gen_z": 0.1, "business": 0.8, "technical": 0.7,
    }
    # [FINE-TUNE] Styles that naturally ask questions back
    question_rate_map = {
        "witty": 0.35,
        "friendly": 0.3,
        "warm": 0.3,
        "casual": 0.2,
        "professional": 0.15,
        "blunt": 0.1,
    }

    phrases = _normalize_text_list(form_data.get("signature_phrases", []))
    avoid = _normalize_text_list(form_data.get("avoid_words", []))

    return {
        "tone_label": f"{style.title()} Communicator",
        "avg_reply_length_words": length_map.get(length, 22),
        "emoji_frequency": emoji_map.get(emoji, 0.5),
        "formality": formality_map.get(style, formality_map.get(language, 0.3)),
        "humor_level": "moderate" if style in ("witty", "casual", "gen_z") else "subtle",
        "gratitude_pattern": style in ("warm", "friendly"),
        "confrontation_style": "polite_correction" if style == "professional" else "deflect_with_humor",
        "exclamation_rate": 0.4 if style in ("casual", "warm", "witty") else 0.2,
        "question_rate": question_rate_map.get(style, 0.2),  # [FINE-TUNE] style-aware
        "signature_phrases": phrases,
        "vocabulary_level": "conversational",
        "avoid_topics": avoid,
    }


def _merge_form_data_with_current(
    incoming_form_data: Dict[str, Any],
    current_profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge partial recalibration input with existing profile form data."""
    incoming_form_data = incoming_form_data or {}
    current_form_data = (current_profile or {}).get("form_data", {}) if current_profile else {}

    merged: Dict[str, Any] = {}

    def _pick_text(field: str, default: str = "") -> str:
        if field in incoming_form_data:
            value = str(incoming_form_data.get(field, "")).strip()
            if value:
                return value
            if field == "additional_notes":
                return ""
        current_value = str(current_form_data.get(field, "")).strip()
        return current_value or default

    merged["style"] = _pick_text("style", "friendly")
    merged["emoji_usage"] = _pick_text("emoji_usage", "moderate")
    merged["reply_length"] = _pick_text("reply_length", "medium")
    merged["language"] = _pick_text("language", "friendly")
    merged["additional_notes"] = _pick_text("additional_notes", "")

    if "signature_phrases" in incoming_form_data:
        merged["signature_phrases"] = _normalize_text_list(incoming_form_data.get("signature_phrases", []))
    else:
        merged["signature_phrases"] = _normalize_text_list(current_form_data.get("signature_phrases", []))

    if "avoid_words" in incoming_form_data:
        merged["avoid_words"] = _normalize_text_list(incoming_form_data.get("avoid_words", []))
    else:
        merged["avoid_words"] = _normalize_text_list(current_form_data.get("avoid_words", []))

    return merged


def _sanitize_extracted_traits(
    extracted_traits: Dict[str, Any],
    form_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Clamp and normalize LLM traits while preserving exact user form intent.

    [FINE-TUNE] Formality is now a weighted blend of style + language signals
    instead of style-wins-all, giving more nuanced results for e.g. a
    'witty' style with 'business' language (result: ~0.5 instead of 0.3).
    """
    extracted_traits = extracted_traits or {}

    style = str(form_data.get("style", "friendly")).strip().lower() or "friendly"
    emoji_usage = str(form_data.get("emoji_usage", "moderate")).strip().lower() or "moderate"
    reply_length = str(form_data.get("reply_length", "medium")).strip().lower() or "medium"
    language = str(form_data.get("language", "friendly")).strip().lower() or "friendly"
    signature_phrases = _normalize_text_list(form_data.get("signature_phrases", []))
    avoid_words = _normalize_text_list(form_data.get("avoid_words", []))

    emoji_map = {"none": 0.0, "moderate": 0.5, "heavy": 0.8}
    length_map = {"short_punchy": 12, "medium": 22, "detailed": 40}
    language_vocab_map = {
        "gen_z": "conversational",
        "friendly": "conversational",
        "business": "sophisticated",
        "technical": "technical",
    }
    style_formality_map = {
        "casual": 0.2, "friendly": 0.3, "warm": 0.3,
        "witty": 0.3, "professional": 0.75, "blunt": 0.55,
    }
    language_formality_map = {
        "gen_z": 0.15, "friendly": 0.3, "business": 0.85, "technical": 0.7,
    }

    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _to_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

    humor = str(extracted_traits.get("humor_level", "subtle")).strip().lower()
    if humor not in {"none", "subtle", "moderate", "heavy"}:
        humor = "moderate" if style in {"witty", "casual"} else "subtle"

    confrontation_style = str(extracted_traits.get("confrontation_style", "polite_correction")).strip().lower()
    if confrontation_style not in {"ignore", "deflect_with_humor", "polite_correction", "direct_rebuttal"}:
        confrontation_style = "polite_correction" if style == "professional" else "deflect_with_humor"

    tone_label = str(extracted_traits.get("tone_label", "")).strip()
    if not tone_label:
        tone_label = f"{style.title()} {language.title()}"

    vocab = str(extracted_traits.get("vocabulary_level", "")).strip().lower()
    if vocab not in {"simple", "conversational", "sophisticated", "technical"}:
        vocab = language_vocab_map.get(language, "conversational")

    # [FINE-TUNE] Blend style + language formality (60/40 split) for nuanced result
    style_f = style_formality_map.get(style, 0.3)
    lang_f = language_formality_map.get(language, 0.3)
    blended_formality = round(style_f * 0.6 + lang_f * 0.4, 3)
    llm_formality = _to_float(extracted_traits.get("formality"), blended_formality)
    # Weight: 50% LLM, 50% deterministic blend — keeps LLM insight but anchors to form
    final_formality = round(llm_formality * 0.5 + blended_formality * 0.5, 3)

    sanitized = {
        "tone_label": tone_label,
        "avg_reply_length_words": length_map.get(reply_length, 22),
        "emoji_frequency": emoji_map.get(emoji_usage, 0.5),
        "formality": _clamp(final_formality, 0.0, 1.0),
        "humor_level": humor,
        "gratitude_pattern": bool(extracted_traits.get("gratitude_pattern", style in {"warm", "friendly"})),
        "confrontation_style": confrontation_style,
        "exclamation_rate": _to_float(extracted_traits.get("exclamation_rate", 0.3), 0.3),
        "question_rate": _to_float(extracted_traits.get("question_rate", 0.2), 0.2),
        "signature_phrases": signature_phrases,
        "vocabulary_level": vocab,
        "avoid_topics": avoid_words,
    }

    sanitized["avg_reply_length_words"] = _to_int(_clamp(float(sanitized["avg_reply_length_words"]), 5, 60), 22)
    sanitized["emoji_frequency"] = _clamp(sanitized["emoji_frequency"], 0.0, 1.0)
    sanitized["formality"] = _clamp(sanitized["formality"], 0.0, 1.0)
    sanitized["exclamation_rate"] = _clamp(sanitized["exclamation_rate"], 0.0, 1.0)
    sanitized["question_rate"] = _clamp(sanitized["question_rate"], 0.0, 1.0)

    return sanitized


async def _ensure_engagement_settings(user_id: str) -> None:
    """Create default engagement settings if none exist."""
    existing = await engagement_settings_col.find_one({"user_id": user_id})
    if not existing:
        default_settings = {
            "user_id": user_id,
            "auto_reply_enabled": False,
            "reply_mode": "automatic",
            "reply_window_hours": 10,
            "platforms": [],
            "spam_filter_enabled": True,
            "max_replies_per_post": 0,
            "max_replies_per_day": 0,
            "blacklisted_words": [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        await engagement_settings_col.insert_one(default_settings)
        logger.info(f"📋 Default engagement settings created for user {user_id}")


def _build_tone_dna(extracted_traits: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a tone_dna object matching the frontend ToneDNA interface.

    [FINE-TUNE] directness now maps from formality (more semantically meaningful)
    instead of exclamation_rate, which is a stylistic quirk, not directness.
    """
    humor_str = str(extracted_traits.get("humor_level", "subtle")).lower()
    humor_map = {"none": 0.0, "subtle": 0.25, "moderate": 0.5, "heavy": 0.8}
    humor_float = humor_map.get(humor_str, 0.25)

    confrontation = extracted_traits.get("confrontation_style", "polite_correction")
    tone_label = extracted_traits.get("tone_label", "Custom Tone")
    vocab = extracted_traits.get("vocabulary_level", "conversational")

    return {
        "tone_label": tone_label,
        "avg_reply_length_words": extracted_traits.get("avg_reply_length_words", 18),
        "emoji_frequency": extracted_traits.get("emoji_frequency", 0.5),
        "formality": extracted_traits.get("formality", 0.3),
        "humor_level": humor_float,
        # [FINE-TUNE] directness = formality, not exclamation_rate
        "directness": extracted_traits.get("formality", 0.3),
        "persona_summary": (
            f"{tone_label} communicator with {vocab} vocabulary. "
            f"Handles confrontation via {confrontation.replace('_', ' ')}."
        ),
        "example_replies": [],
        "vocabulary_preferences": [vocab],
        "signature_phrases": extracted_traits.get("signature_phrases", []),
        "avoid_patterns": extracted_traits.get("avoid_topics", []),
    }


def _normalize_text_list(
    value: Any,
    max_items: int = 50,
    max_item_len: int = 100,
) -> List[str]:
    """
    Accept list or comma/newline-separated text and return a clean unique list.

    [SECURITY] Now accepts max_items and max_item_len caps to prevent oversized inputs.
    """
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
        if len(cleaned) >= max_items:
            break
        token = _strip_html(str(item).strip())[:max_item_len]
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(token)

    return cleaned


def _join_text_list_for_prompt(value: Any) -> str:
    items = _normalize_text_list(value)
    return ", ".join(items) if items else "none specified"
