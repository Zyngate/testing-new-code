# stelle_backend/services/post_content_analyzer.py
"""
Post Content Analyzer — Phase 2 of Autonomous Engagement.

When a post is published, this module analyzes the content (caption, video, image)
and stores a content_context in post_content_cache so the reply engine can
understand what each post is about.

Reuses existing video_cache_repo for video posts.
If a video was NOT previously analyzed (no transcript in cache),
downloads it and runs the full STT + Vision pipeline.
"""

import os
import json
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import httpx

from database import db
from config import logger
from services.ai_service import groq_generate_text
from services.video_cache_repo import get_cached_video_analysis, save_video_analysis
from services.video_cache_utils import compute_video_hash
from services.video_caption_service import (
    extract_audio_from_video,
    get_transcript_groq,
    extract_frames_from_video,
    analyze_frames_with_groq,
    clean_ocr_text,
    normalize_ocr_spelling,
    clean_transcript_for_caption,
    identify_person_from_frames,
)


# ── Collections ──────────────────────────────────────────────
post_cache_col = db["post_content_cache"]


# ── Constants ────────────────────────────────────────────────
ANALYSIS_MODEL = "llama-3.3-70b-versatile"
CACHE_TTL_DAYS = 30

# Temp directory for downloaded videos (same pattern as video_caption_service)
TEMP_DIR = Path(os.getenv("TEMP", "/tmp")) / "stelle_engagement"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CONTENT_ANALYSIS_SYSTEM = """You are a social media content analyst. 
Analyze the given post content and return analysis as JSON only."""

CONTENT_ANALYSIS_PROMPT = """
Analyze this social media post and return a JSON analysis.

PLATFORM: {platform}
MEDIA TYPE: {media_type}
CAPTION: {caption}
ADDITIONAL CONTEXT (transcript/visual): {additional_context}

Return ONLY this JSON structure:
{{
  "content_summary": "<1-2 sentence description of what the post is about>",
  "key_topics": ["<topic1>", "<topic2>", "<topic3>"],
  "sentiment": "<enthusiastic|educational|inspirational|humorous|calm|controversial|promotional>",
  "likely_comment_themes": [
    "<what people might comment about - theme 1>",
    "<theme 2>",
    "<theme 3>",
    "<theme 4>"
  ]
}}
"""


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

async def analyze_post_content(
    user_id: str,
    post_id: str,
    platform: str,
    media_type: str,
    caption: str = "",
    media_urls: List[str] = None,
    video_hash: str = None,
) -> Dict[str, Any]:
    """
    Analyze a published post and cache the content context.
    
    Called automatically after a post is published.
    For videos, reuses existing video_cache_repo analysis.
    
    Returns the saved content context document.
    """
    try:
        # Check if already analyzed
        existing = await post_cache_col.find_one({
            "post_id": post_id,
            "user_id": user_id
        })
        if existing:
            logger.info(f"📋 Post {post_id} already analyzed, skipping")
            existing["_id"] = str(existing["_id"])
            return existing

        # Gather additional context based on media type
        additional_context = ""

        if media_type.lower() in ("video", "reel", "reels"):
            # ── Try to reuse existing video analysis from cache ──
            cached_video = None
            has_transcript = False

            if video_hash:
                cached_video = await get_cached_video_analysis(video_hash)
                if cached_video:
                    has_transcript = bool(cached_video.get("transcript", "").strip())

            if cached_video and has_transcript:
                # ✅ Cache HIT with transcript — reuse everything
                additional_context = _build_video_context(cached_video)
                logger.info(f"♻️ Reused video cache (with transcript) for post {post_id}")

            else:
                # ❌ No cache OR cache exists but has no transcript
                # → Download video and run full STT + Vision pipeline
                logger.info(
                    f"🔄 Post {post_id}: {'no transcript in cache' if cached_video else 'no cache found'} "
                    f"— running video analysis pipeline"
                )
                analysis_result = await _analyze_video_from_url(
                    media_urls=media_urls,
                    existing_video_hash=video_hash,
                    user_id=user_id,
                    scheduled_post_id=post_id,
                )
                if analysis_result:
                    additional_context = _build_video_context(analysis_result)
                else:
                    additional_context = f"Video post with caption only. Media URLs: {media_urls or 'N/A'}"

        elif media_type.lower() in ("image", "photo", "carousel"):
            additional_context = f"Image post with caption. Media URLs: {media_urls or 'N/A'}"

        else:
            additional_context = "Text-based post, caption is the primary content."

        # Generate content analysis via LLM
        analysis = await _generate_content_analysis(
            platform=platform,
            media_type=media_type,
            caption=caption,
            additional_context=additional_context,
        )

        # Build and save the document
        doc = {
            "post_id": post_id,
            "user_id": user_id,
            "platform": platform.lower(),
            "media_type": media_type.lower(),
            "caption_snippet": caption[:200] if caption else "",
            "content_summary": analysis.get("content_summary", caption[:100] if caption else "Social media post"),
            "key_topics": analysis.get("key_topics", []),
            "sentiment": analysis.get("sentiment", "neutral"),
            "likely_comment_themes": analysis.get("likely_comment_themes", []),
            "analyzed_at": datetime.now(timezone.utc),
            "ttl_expire": datetime.now(timezone.utc) + timedelta(days=CACHE_TTL_DAYS),
        }

        result = await post_cache_col.insert_one(doc)
        doc["_id"] = str(result.inserted_id)

        logger.info(f"✅ Post content analyzed: {post_id} ({platform}/{media_type}) → {analysis.get('content_summary', '')[:60]}")
        return doc

    except Exception as e:
        logger.error(f"❌ Error analyzing post {post_id}: {e}", exc_info=True)
        # Return a minimal fallback so the system doesn't break
        return {
            "post_id": post_id,
            "user_id": user_id,
            "content_summary": caption[:100] if caption else "Social media post",
            "key_topics": [],
            "sentiment": "neutral",
            "likely_comment_themes": [],
        }


async def get_post_context(post_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached post content analysis."""
    try:
        doc = await post_cache_col.find_one({
            "post_id": post_id,
            "user_id": user_id
        })
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"Error fetching post context {post_id}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# INTERNAL
# ══════════════════════════════════════════════════════════════

def _build_video_context(analysis: Dict[str, Any]) -> str:
    """Build a text context string from a video analysis dict."""
    parts = []
    transcript = analysis.get("transcript", "")
    if transcript:
        parts.append(f"Video Transcript: {transcript[:500]}")
    visual_summary = analysis.get("visual_summary", "")
    if visual_summary:
        parts.append(f"Visual Summary: {visual_summary[:300]}")
    objects = analysis.get("objects", [])
    if objects:
        parts.append(f"Objects: {', '.join(objects)}")
    actions = analysis.get("actions", [])
    if actions:
        parts.append(f"Actions: {', '.join(actions)}")
    detected_person = analysis.get("detected_person")
    if detected_person:
        parts.append(f"Detected Person: {detected_person}")
    marketing_prompt = analysis.get("marketing_prompt", "")
    if marketing_prompt:
        parts.append(f"Marketing Context: {marketing_prompt[:400]}")
    return "\n".join(parts) if parts else "Video post (analysis unavailable)"


async def _download_video(url: str) -> Optional[str]:
    """
    Download a video from a URL (Cloudinary, etc.) to a temp file.
    Returns the local file path, or None on failure.
    """
    try:
        local_path = str(TEMP_DIR / f"{uuid.uuid4().hex}.mp4")
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
        file_size = os.path.getsize(local_path)
        logger.info(f"📥 Downloaded video ({file_size / 1024:.0f} KB) → {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"❌ Failed to download video from {url}: {e}")
        return None


async def _analyze_video_from_url(
    media_urls: List[str] = None,
    existing_video_hash: str = None,
    user_id: Optional[str] = None,
    scheduled_post_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Download a video from its URL, run the full STT + Vision pipeline,
    save results to video_analysis_cache, and return the analysis dict.

    Reuses the exact same functions from video_caption_service.py.
    """
    if not media_urls:
        return None

    # Pick the first video URL
    video_url = media_urls[0]
    local_path = None

    try:
        # ── Download video ──
        local_path = await _download_video(video_url)
        if not local_path:
            return None

        # ── Compute hash (for caching) ──
        video_hash = existing_video_hash or await asyncio.to_thread(
            compute_video_hash, local_path
        )

        # ── Double-check cache with the computed hash ──
        # (the caller may not have had the hash if the post didn't store it)
        cached = await get_cached_video_analysis(video_hash)
        if cached and cached.get("transcript", "").strip():
            logger.info(f"♻️ Cache HIT after hash computation — skipping analysis")
            return cached

        # ── Extract audio + frames IN PARALLEL ──
        async def _audio_task():
            try:
                return await extract_audio_from_video(local_path)
            except Exception as e:
                logger.warning(f"Audio extraction failed: {e}")
                return None

        async def _frames_task():
            try:
                return await asyncio.to_thread(
                    extract_frames_from_video, local_path, None, 1, 3
                )
            except Exception as e:
                logger.warning(f"Frame extraction failed: {e}")
                return []

        audio_path, frame_paths = await asyncio.gather(
            _audio_task(), _frames_task()
        )
        if isinstance(audio_path, Exception):
            audio_path = None
        if isinstance(frame_paths, Exception):
            frame_paths = []

        # ── STT + Visual analysis IN PARALLEL ──
        async def _transcribe_task():
            if not audio_path:
                return ""
            try:
                return await get_transcript_groq(audio_path)
            except Exception as e:
                logger.warning(f"Transcription failed: {e}")
                return ""

        async def _visual_task():
            if not frame_paths:
                return {"visual_captions": [], "visual_summary": ""}
            try:
                return await analyze_frames_with_groq(frame_paths)
            except Exception as e:
                logger.warning(f"Visual analysis failed: {e}")
                return {"visual_captions": [], "visual_summary": ""}

        transcript, visual_result = await asyncio.gather(
            _transcribe_task(), _visual_task()
        )

        visual_summary = visual_result.get("visual_summary", "")
        visual_captions = visual_result.get("visual_captions", [])
        detected_texts = visual_result.get("detected_text", [])
        objects_detected = visual_result.get("objects", [])
        actions_detected = visual_result.get("actions", [])

        # ── OCR cleanup ──
        raw_ocr = "\n".join([t for t in detected_texts if t]).strip()
        ocr_text_combined = clean_ocr_text(raw_ocr)
        if ocr_text_combined:
            ocr_text_combined = await normalize_ocr_spelling(ocr_text_combined)

        # ── Build marketing prompt (same logic as video_caption_service) ──
        cleaned_transcript = clean_transcript_for_caption(transcript) if transcript else ""
        merge_parts = []

        if cleaned_transcript and len(cleaned_transcript.split()) > 20:
            merge_parts.append(
                "This video contains a real spoken conversation. "
                "Base understanding primarily on what is being said."
            )
            merge_parts.append(f"Conversation:\n{cleaned_transcript}")
            if ocr_text_combined:
                merge_parts.append(f"On-screen text:\n{ocr_text_combined}")
            if visual_summary:
                merge_parts.append(f"Visual context (secondary): {visual_summary}")
        else:
            if visual_summary:
                merge_parts.append(f"Scene: {visual_summary}")
            if visual_captions:
                frame_caps = "; ".join([cap for _, cap in visual_captions if cap][:5])
                merge_parts.append(f"Visual details: {frame_caps}")
            if ocr_text_combined:
                merge_parts.append(f"On-screen text: {ocr_text_combined}")

        marketing_prompt = "\n".join(merge_parts)

        # ── Identify person ──
        detected_person = None
        try:
            identified = await identify_person_from_frames(
                visual_captions=visual_captions,
                ocr_texts=detected_texts,
                transcript=transcript,
            )
            if identified and identified.lower() != "unknown":
                detected_person = identified
                marketing_prompt = (
                    f"Context: The video includes {detected_person}.\n\n"
                    + marketing_prompt
                )
        except Exception as e:
            logger.warning(f"Person identification failed: {e}")

        # ── Save to video_analysis_cache ──
        analysis_data = {
            "transcript": transcript,
            "visual_summary": visual_summary,
            "visual_captions": visual_captions,
            "detected_texts": detected_texts,
            "ocr_text_combined": ocr_text_combined,
            "detected_person": detected_person,
            "marketing_prompt": marketing_prompt,
            "objects": objects_detected,
            "actions": actions_detected,
        }
        await save_video_analysis(
            video_hash,
            analysis_data,
            user_id=user_id,
            scheduled_post_id=scheduled_post_id,
        )
        logger.info(f"✅ Video analyzed and cached (hash={video_hash[:16]}…)")

        return analysis_data

    except Exception as e:
        logger.error(f"❌ Video analysis pipeline failed: {e}", exc_info=True)
        return None

    finally:
        # ── Cleanup temp files ──
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass


async def _generate_content_analysis(
    platform: str,
    media_type: str,
    caption: str,
    additional_context: str,
) -> Dict[str, Any]:
    """Call Groq LLM to analyze post content."""
    prompt = CONTENT_ANALYSIS_PROMPT.format(
        platform=platform,
        media_type=media_type,
        caption=caption[:500] if caption else "No caption",
        additional_context=additional_context[:800] if additional_context else "None",
    )

    raw = await groq_generate_text(
        model=ANALYSIS_MODEL,
        prompt=prompt,
        system_msg=CONTENT_ANALYSIS_SYSTEM,
        temperature=0.3,
        max_completion_tokens=400,
    )

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        return json.loads(cleaned)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"⚠️ Failed to parse post analysis JSON: {e}")
        return {
            "content_summary": caption[:100] if caption else "Social media post",
            "key_topics": [],
            "sentiment": "neutral",
            "likely_comment_themes": [],
        }
