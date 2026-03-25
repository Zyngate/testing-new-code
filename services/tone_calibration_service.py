# stelle_backend/services/tone_calibration_service.py
"""
Tone Calibration Service — Phase 1 of Autonomous Engagement.

Handles:
- Saving user tone form data (Method A)
- Extracting Tone DNA via Groq LLM from form input
- Fetching the active tone profile for a user
- Resetting (archiving old, creating new) tone profiles
"""

import json
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

TONE_EXTRACTION_SYSTEM_MSG = """You are a tone analysis expert. 
Given a user's self-described communication preferences, extract a precise 
Tone DNA profile as JSON. Be accurate and specific. Only return valid JSON."""

TONE_EXTRACTION_PROMPT_TEMPLATE = """
Analyze the following user-provided communication preferences and generate a Tone DNA profile.

USER FORM DATA:
- Communication Style: {style}
- Emoji Usage: {emoji_usage}
- Response Length Preference: {reply_length}
- Language/Slang: {language}
- Signature Phrases: {signature_phrases}
- Words/Topics to Avoid: {avoid_words}
- Additional Notes: {additional_notes}

Based on these preferences, produce the following JSON (no extra text):
{{
  "tone_label": "<2-4 word label e.g. 'Warm & Witty Professional'>",
  "avg_reply_length_words": <number between 5-60 based on their reply_length preference>,
  "emoji_frequency": <0.0-1.0 float based on emoji_usage>,
  "formality": <0.0=very casual, 1.0=very formal>,
  "humor_level": "<none|subtle|moderate|heavy>",
  "gratitude_pattern": <true if style suggests thankfulness, else false>,
  "confrontation_style": "<ignore|deflect_with_humor|polite_correction|direct_rebuttal>",
  "exclamation_rate": <0.0-1.0>,
  "question_rate": <0.0-1.0 how often they ask questions back>,
  "signature_phrases": [<list of phrases from user input plus inferred ones>],
  "vocabulary_level": "<simple|conversational|sophisticated|technical>",
  "avoid_topics": [<from user input>]
}}
"""


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

async def get_active_tone_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the currently active tone profile for a user.
    Returns None if no profile exists.
    """
    try:
        doc = await tone_profiles_col.find_one({
            "user_id": user_id,
            "is_active": True
        }, sort=[("updated_at", -1), ("created_at", -1)])
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
    profile = await get_active_tone_profile(user_id)
    if profile:
        created = profile.get("created_at", "")
        # Convert datetime to ISO string if needed
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
    form_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Method A — Tone Calibration via Form.
    
    1. Saves raw form data
    2. Sends it to Groq LLM to extract Tone DNA traits
    3. Upserts complete profile in user_tone_profiles
    4. Returns the saved profile

    form_data expected keys:
      - style: str              (e.g. "casual", "professional", "witty", "warm", "blunt")
      - emoji_usage: str        (e.g. "heavy", "moderate", "none")
      - reply_length: str       (e.g. "short_punchy", "medium", "detailed")
      - language: str           (e.g. "gen_z", "business", "friendly", "technical")
      - signature_phrases: str  (free text, comma-separated)
      - avoid_words: str        (free text, comma-separated)
      - additional_notes: str   (optional free text)
    """
    try:
        now = datetime.now(timezone.utc)

        # 1. Find most recent profile (active or archived) so recalibration updates existing entry.
        current = await tone_profiles_col.find_one(
            {"user_id": user_id},
            sort=[("updated_at", -1), ("created_at", -1)]
        )
        new_version = (current.get("version", 0) + 1) if current else 1

        # 2. Extract Tone DNA via LLM
        extracted_traits = await _extract_tone_from_form(form_data)

        # 3. Build the profile payload
        created_at = current.get("created_at", now) if current else now
        profile_doc = {
            "user_id": user_id,
            "version": new_version,
            "tone_label": extracted_traits.get("tone_label", "Custom Tone"),
            "calibration_method": "form",
            "form_data": {
                "style": form_data.get("style", ""),
                "emoji_usage": form_data.get("emoji_usage", ""),
                "reply_length": form_data.get("reply_length", ""),
                "language": form_data.get("language", ""),
                "signature_phrases": form_data.get("signature_phrases", ""),
                "avoid_words": form_data.get("avoid_words", ""),
                "additional_notes": form_data.get("additional_notes", ""),
            },
            "extracted_traits": extracted_traits,
            "calibration_pairs": [],  # Empty — Method A doesn't use calibration pairs
            "is_active": True,
            "created_at": created_at,
            "updated_at": now,
        }

        # 4. Update existing profile if present; otherwise create first profile.
        if current:
            target_id = current["_id"]
            await tone_profiles_col.update_one(
                {"_id": target_id},
                {
                    "$set": profile_doc,
                    "$unset": {"archived_at": ""}
                }
            )

            # Defensive cleanup: keep exactly one active profile per user.
            await tone_profiles_col.update_many(
                {"user_id": user_id, "_id": {"$ne": target_id}},
                {"$set": {"is_active": False, "archived_at": now, "updated_at": now}}
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

        # Convert datetime fields to ISO strings for JSON serialization
        for dt_key in ("created_at", "updated_at"):
            if hasattr(saved_profile.get(dt_key), "isoformat"):
                saved_profile[dt_key] = saved_profile[dt_key].isoformat()

        return saved_profile

    except Exception as e:
        logger.error(f"❌ Error calibrating tone for {user_id}: {e}", exc_info=True)
        raise


async def reset_tone(user_id: str) -> Dict[str, Any]:
    """
    Archive the current tone profile and return status.
    After this, frontend should redirect user to re-run calibration.
    """
    try:
        result = await tone_profiles_col.update_many(
            {"user_id": user_id, "is_active": True},
            {"$set": {"is_active": False, "archived_at": datetime.now(timezone.utc)}}
        )
        archived_count = result.modified_count
        logger.info(f"🔄 Tone reset for user {user_id} — archived {archived_count} profile(s)")
        return {
            "success": True,
            "archived_count": archived_count,
            "message": "Tone profile archived. Please complete calibration again."
        }
    except Exception as e:
        logger.error(f"❌ Error resetting tone for {user_id}: {e}")
        raise


async def get_engagement_settings(user_id: str) -> Dict[str, Any]:
    """
    Fetch engagement settings for a user (or create defaults).
        Maps internal DB field names to frontend-expected field names.
        daily/per-post limits are not configurable.
    """
    await _ensure_engagement_settings(user_id)
    doc = await engagement_settings_col.find_one({"user_id": user_id})
    if doc:
        doc["_id"] = str(doc["_id"])
        # Map backend fields to frontend EngagementSettings interface
        doc["enabled"] = doc.pop("auto_reply_enabled", False)
        # Keep compatibility fields in response
        doc["daily_reply_limit"] = 0
        doc["max_replies_per_post"] = 0
        doc.setdefault("reply_window_hours", 10)
        # Convert datetime fields to ISO strings
        for dt_key in ("created_at", "updated_at"):
            if hasattr(doc.get(dt_key), "isoformat"):
                doc[dt_key] = doc[dt_key].isoformat()
    return doc


async def update_engagement_settings(user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
        Update engagement settings for a user.
        Only selected fields are user-configurable.
    """
    # Map frontend field names to backend field names
    field_mapping = {
        "enabled": "auto_reply_enabled",
    }
    mapped_updates = {}
    for k, v in updates.items():
        backend_key = field_mapping.get(k, k)
        mapped_updates[backend_key] = v

    allowed_fields = {
        "auto_reply_enabled", "reply_mode",
        "platforms", "spam_filter_enabled", "blacklisted_words",
        "reply_window_hours"
    }
    filtered = {k: v for k, v in mapped_updates.items() if k in allowed_fields}
    # Clamp reply_window_hours to a safe range (1–168)
    if "reply_window_hours" in filtered:
        filtered["reply_window_hours"] = max(1, min(168, int(filtered["reply_window_hours"])))
    filtered["updated_at"] = datetime.now(timezone.utc)

    await engagement_settings_col.update_one(
        {"user_id": user_id},
        {"$set": filtered},
        upsert=True
    )

    return await get_engagement_settings(user_id)


# ══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════

async def _extract_tone_from_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send form data to Groq LLM and extract structured Tone DNA traits.
    Falls back to sensible defaults if LLM fails.
    """
    prompt = TONE_EXTRACTION_PROMPT_TEMPLATE.format(
        style=form_data.get("style", "friendly"),
        emoji_usage=form_data.get("emoji_usage", "moderate"),
        reply_length=form_data.get("reply_length", "medium"),
        language=form_data.get("language", "friendly"),
        signature_phrases=form_data.get("signature_phrases", "none specified"),
        avoid_words=form_data.get("avoid_words", "none specified"),
        additional_notes=form_data.get("additional_notes", "none"),
    )

    raw = await groq_generate_text(
        model=TONE_MODEL,
        prompt=prompt,
        system_msg=TONE_EXTRACTION_SYSTEM_MSG,
        temperature=0.4,
        max_completion_tokens=600,
    )

    # Try to parse JSON from the LLM output
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        traits = json.loads(cleaned)
        logger.info(f"✅ Tone DNA extracted: {traits.get('tone_label', 'Unknown')}")
        return traits

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"⚠️ Failed to parse LLM tone output, using defaults. Error: {e}")
        return _default_traits_from_form(form_data)


def _default_traits_from_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback: build sensible traits from form data without LLM.
    Used when LLM extraction fails.
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

    sig_text = form_data.get("signature_phrases", "")
    phrases = [p.strip() for p in sig_text.split(",") if p.strip()] if sig_text else []

    avoid_text = form_data.get("avoid_words", "")
    avoid = [w.strip() for w in avoid_text.split(",") if w.strip()] if avoid_text else []

    return {
        "tone_label": f"{style.title()} Communicator",
        "avg_reply_length_words": length_map.get(length, 22),
        "emoji_frequency": emoji_map.get(emoji, 0.5),
        "formality": formality_map.get(style, formality_map.get(language, 0.3)),
        "humor_level": "moderate" if style in ("witty", "casual", "gen_z") else "subtle",
        "gratitude_pattern": style in ("warm", "friendly"),
        "confrontation_style": "polite_correction" if style == "professional" else "deflect_with_humor",
        "exclamation_rate": 0.4 if style in ("casual", "warm", "witty") else 0.2,
        "question_rate": 0.2,
        "signature_phrases": phrases,
        "vocabulary_level": "conversational",
        "avoid_topics": avoid,
    }


async def _ensure_engagement_settings(user_id: str):
    """Create default engagement settings if none exist."""
    existing = await engagement_settings_col.find_one({"user_id": user_id})
    if not existing:
        default_settings = {
            "user_id": user_id,
            "auto_reply_enabled": False,  # Off until user explicitly enables
            "reply_mode": "automatic",    # "automatic" | "review" | "manual"
            "reply_window_hours": 10,
            "platforms": [],              # User must select platforms
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
    Build a tone_dna object matching the frontend ToneDNA interface
    from the LLM-extracted traits.
    """
    # Map humor_level string to float
    humor_str = str(extracted_traits.get("humor_level", "subtle")).lower()
    humor_map = {"none": 0.0, "subtle": 0.25, "moderate": 0.5, "heavy": 0.8}
    humor_float = humor_map.get(humor_str, 0.25)

    return {
        "tone_label": extracted_traits.get("tone_label", "Custom Tone"),
        "avg_reply_length_words": extracted_traits.get("avg_reply_length_words", 18),
        "emoji_frequency": extracted_traits.get("emoji_frequency", 0.5),
        "formality": extracted_traits.get("formality", 0.3),
        "humor_level": humor_float,
        "directness": extracted_traits.get("exclamation_rate", 0.4),
        "persona_summary": (
            f"{extracted_traits.get('tone_label', 'Custom')} communicator with "
            f"{extracted_traits.get('vocabulary_level', 'conversational')} vocabulary. "
            f"Handles confrontation via {extracted_traits.get('confrontation_style', 'polite correction')}."
        ),
        "example_replies": [],  # Could be generated separately
        "vocabulary_preferences": [extracted_traits.get("vocabulary_level", "conversational")],
        "signature_phrases": extracted_traits.get("signature_phrases", []),
        "avoid_patterns": extracted_traits.get("avoid_topics", []),
    }
