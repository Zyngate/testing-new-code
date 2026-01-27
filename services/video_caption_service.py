import os
import subprocess
import uuid
import shlex
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import glob
import asyncio
import base64
import re

from config import logger
from groq import Groq
from services.ai_service import groq_generate_text
from services.post_generator_service import (
    generate_keywords_post,
    fetch_platform_hashtags,
    generate_caption_post,
)

# LLM / Vision model names
VISION_MODEL = "llava-v1.5-7b"  # change if you prefer another Groq vision model
STT_MODEL = "whisper-large-v3"
SUMMARIZE_MODEL = "llama-3.3-70b-versatile" # used only for text-compression if desired

# TEMP directory
# TEMP directory (outside project to avoid reload loop)
TEMP_DIR = Path(os.getenv("TEMP", "/tmp")) / "stelle_video"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Utility: run shell command
def run_cmd(cmd: str) -> Tuple[int, str, str]:
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    return proc.returncode, out.decode(errors="ignore"), err.decode(errors="ignore")


# -----------------------------------------------------
# 1. EXTRACT AUDIO (FFmpeg)
# -----------------------------------------------------
async def extract_audio_from_video(video_path: str, out_audio: Optional[str] = None) -> str:
    if out_audio is None:
        out_audio = str(TEMP_DIR / f"{uuid.uuid4().hex}.wav")

    video_quoted = shlex.quote(video_path)
    audio_quoted = shlex.quote(out_audio)

    # create 16k mono WAV (LINEAR16) which works well with STT
    cmd = f'ffmpeg -y -i {video_quoted} -vn -ac 1 -ar 16000 -sample_fmt s16 {audio_quoted} -hide_banner -loglevel error'
    logger.info(f"Running ffmpeg to extract audio: {cmd}")
    returncode, out, err = await asyncio.to_thread(run_cmd, cmd)
    if returncode != 0:
        logger.error(f"ffmpeg failed extracting audio: {err.strip()}")
        raise RuntimeError("Audio extraction failed. Ensure ffmpeg is installed and video file exists.")
    return out_audio

async def identify_person(transcript: str, ocr_text: str, visual_summary: str) -> str:
    prompt = f"""
You are an entity recognition system.

From the following data, identify if a WELL-KNOWN PUBLIC FIGURE is present.
Only return the name if confidence is HIGH.
Otherwise return "unknown".
Priority order for confidence:
1. Spoken transcript (highest confidence)
2. OCR text (may contain spelling errors)
3. Visual summary (lowest confidence)

Only return the name if confidence is HIGH.
Otherwise return "unknown".

Transcript:
{transcript}

OCR:
{ocr_text}

Visual summary:
{visual_summary}

Return only ONE value:
- Full name (e.g., Elon Musk)
- OR "unknown"
"""

    resp = groq_generate_text("llama-3.3-70b-versatile", prompt)
    if hasattr(resp, "__await__"):
        resp = await resp
    return resp.strip()


# -----------------------------------------------------
# 1b. EXTRACT FRAMES (FFmpeg)
# -----------------------------------------------------
def extract_frames_from_video(video_path: str, out_dir: Optional[str] = None, fps: int = 1, max_frames: int = 6) -> List[str]:
    if out_dir is None:
        out_dir = str(TEMP_DIR / f"frames_{uuid.uuid4().hex}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frame_pattern = os.path.join(out_dir, "frame_%04d.png")
    cmd = f'ffmpeg -y -i {shlex.quote(video_path)} -vf "fps={fps}" -vsync 0 {shlex.quote(frame_pattern)} -hide_banner -loglevel error'
    logger.info(f"Running ffmpeg to extract frames: {cmd}")
    returncode, out, err = run_cmd(cmd)
    if returncode != 0:
        logger.error(f"ffmpeg frame extraction failed: {err.strip()}")
        raise RuntimeError("Frame extraction failed. Ensure ffmpeg is installed and video file exists.")

    all_frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
    if not all_frames:
        return []
    if len(all_frames) > max_frames:
        step = len(all_frames) / float(max_frames)
        selected = [all_frames[int(i * step)] for i in range(max_frames)]
    else:
        selected = all_frames
    return selected


# -----------------------------------------------------
# 2. TRANSCRIPTION USING GROQ WHISPER (STT)
# -----------------------------------------------------
async def get_transcript_groq(audio_path: str) -> str:
    api_key = os.getenv("GROQ_API_KEY_VIDEO_CAPTION")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY_VIDEO_CAPTION is not set in environment.")

    client = Groq(api_key=api_key)

    # Groq client is sync in many installs; wrap in thread
    def _blocking_transcribe(path: str) -> str:
        with open(path, "rb") as f:
            # Use the correct Groq API endpoint for audio transcriptions
            resp = client.audio.transcriptions.create(
                file=(os.path.basename(path), f),
                model=STT_MODEL
            )
        # response shape varies; try common fields
        if hasattr(resp, "text"):
            return (resp.text or "").strip()
        if isinstance(resp, dict):
            return (resp.get("text") or resp.get("transcript") or "").strip()
        return str(resp)

    try:
        transcript = await asyncio.to_thread(_blocking_transcribe, audio_path)
        return transcript
    except AttributeError as e:
        # If 'audio' attribute doesn't exist, log and try alternative
        logger.error(f"Groq client doesn't support audio attribute: {e}")
        logger.warning("Audio transcription is not available with current Groq SDK version")
        return ""
    except Exception as e:
        logger.error(f"Groq STT transcription failed: {e}")
        return ""


# -----------------------------------------------------
# Helpers: image to data URL
# -----------------------------------------------------
def image_to_data_url(path: str) -> str:
    """Encode local image to data URL (PNG)."""
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9 .,!?\n]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
async def normalize_ocr_spelling(text: str) -> str:
    if not text or len(text.split()) < 2:
        return text

    prompt = f"""
You are correcting OCR text extracted from a video frame.

Rules:
- Fix spelling mistakes
- Preserve proper nouns and names
- Do NOT add new words
- Do NOT change meaning
- Output ONLY the corrected text

OCR text:
{text}
"""

    resp = groq_generate_text("llama-3.3-70b-versatile", prompt)

    # groq_generate_text may be sync or async
    if hasattr(resp, "__await__"):
        resp = await resp

    return (resp or "").strip()


# -----------------------------------------------------
# 3. VISUAL ANALYSIS + OCR USING GROQ SCOUT VISION MODEL
# -----------------------------------------------------
async def analyze_frames_with_groq(frame_paths: List[str]) -> Dict[str, Any]:
    """
    Uses Groq Scout (meta-llama/llama-4-scout-17b-16e-instruct) in JSON mode to:
      - produce a one-line caption per frame
      - extract OCR text (ocr_text)
      - list objects and actions
      - return a short visual summary
    Returns dict with keys:
      - visual_captions: [(basename, caption_str), ...]
      - visual_summary: "one-line summary"
      - detected_text: [ocr_text_for_frame, ...]
      - objects: unique list of objects
      - actions: unique list of actions
    """
    api_key = os.getenv("GROQ_API_KEY_VIDEO_CAPTION")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY_VIDEO_CAPTION is not set.")

    client = Groq(api_key=api_key)

    visual_captions: List[Tuple[str, str]] = []
    detected_text_list: List[str] = []
    objects_list: List[str] = []
    actions_list: List[str] = []

    for fp in frame_paths:
        try:
            # encode frame to base64 data URL
            data_url = image_to_data_url(fp)

            # JSON-mode prompt content (image + instruction)
            prompt_content = [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image and return a JSON object with keys:\n"
                        "caption: one-line description\n"
                        "ocr_text: full extracted visible text from the image (empty string if none)\n"
                        "objects: list of main objects (strings)\n"
                        "actions: list of visible actions (strings)\n"
                        "scene: short scene description\n"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                }
            ]

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt_content}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            # response.choices[0].message.content should already be JSON (string or dict)
            parsed = response.choices[0].message.content
            # parsed may be a string or dict; handle both
            if isinstance(parsed, str):
                parsed_json = json.loads(parsed)
            elif isinstance(parsed, dict):
                parsed_json = parsed
            else:
                parsed_json = json.loads(str(parsed))

            caption = parsed_json.get("caption", "") or ""
            ocr_text = parsed_json.get("ocr_text", "") or ""
            objs = parsed_json.get("objects", []) or []
            acts = parsed_json.get("actions", []) or []

            visual_captions.append((os.path.basename(fp), caption.strip() if caption else ""))

            if ocr_text:
                detected_text_list.append(ocr_text.strip())

            if objs:
                # normalize strings
                objects_list.extend([str(o).strip() for o in objs if o])
            if acts:
                actions_list.extend([str(a).strip() for a in acts if a])

        except Exception as e:
            logger.warning(f"Vision analysis failed for {fp}: {e}")
            visual_captions.append((os.path.basename(fp), ""))

    # ---- COMBINE FOR VISUAL SUMMARY ----
    all_caps = " | ".join([cap for _, cap in visual_captions if cap])
    objects_unique = list(dict.fromkeys([o for o in objects_list if o]))
    actions_unique = list(dict.fromkeys([a for a in actions_list if a]))

    visual_summary = ""
    if all_caps:
        visual_summary_prompt = (
            "Summarize this scene using the combined captions, objects, and actions below.\n\n"
            f"Captions: {all_caps}\n"
            f"Objects: {objects_unique}\n"
            f"Actions: {actions_unique}\n\n"
            "Write ONE line (max 20 words)."
        )
        try:
            ai_resp = groq_generate_text(SUMMARIZE_MODEL, visual_summary_prompt)
            if hasattr(ai_resp, "__await__"):
                ai_resp = await ai_resp
            visual_summary = (ai_resp or "").strip()
        except Exception as e:
            logger.warning(f"Could not generate visual summary: {e}")
            # fallback: pick first caption or truncated combined captions
            visual_summary = all_caps.split(" | ")[0][:120]
    else:
        visual_summary = ""

    return {
        "visual_captions": visual_captions,
        "visual_summary": visual_summary,
        "detected_text": detected_text_list,
        "objects": objects_unique,
        "actions": actions_unique,
    }

async def identify_person_from_frames(
    visual_captions: List[tuple],
    ocr_texts: List[str],
    transcript: str
) -> str:
    prompt = f"""
You are identifying whether a well-known public figure appears in a video.

Signals:
- Visual frame descriptions
- OCR text from frames
- Spoken transcript

Rules:
- Only return a name if confidence is HIGH
- Otherwise return "unknown"
- Do NOT guess

Visual captions:
{visual_captions}

OCR text:
{ocr_texts}

Transcript:
{transcript}

Return ONE value:
- Full name
- OR "unknown"
"""
    resp = groq_generate_text("llama-3.3-70b-versatile", prompt)
    if hasattr(resp, "__await__"):
        resp = await resp
    return resp.strip()


# -----------------------------------------------------
# 4. SUMMARIZE TRANSCRIPT & SCENE
# -----------------------------------------------------
async def summarize_transcript_for_caption(transcript: str) -> Dict[str, str]:
    prompt = f"""
You are a senior marketing content strategist. Given the transcript below,
produce:
1) A short (1–2 line) summary.
2) A line describing the visible scene.
3) A marketing prompt useful for hashtag/caption generation.

Transcript:
\"\"\"{transcript}\"\"\" 

Output JSON with keys: summary, scene, marketing_prompt.
"""
    try:
        ai_resp = groq_generate_text(SUMMARIZE_MODEL, prompt)
        if hasattr(ai_resp, "__await__"):
            ai_resp = await ai_resp
        try:
            parsed = json.loads(ai_resp)
            return {
                "summary": parsed.get("summary", "").strip(),
                "scene": parsed.get("scene", "").strip(),
                "marketing_prompt": parsed.get("marketing_prompt", "").strip(),
            }
        except Exception:
            return {
                "summary": (ai_resp or "").strip().split("\n")[0],
                "scene": "",
                "marketing_prompt": (ai_resp or "").strip(),
            }
    except Exception as e:
        logger.error(f"Error summarizing transcript: {e}", exc_info=True)
        return {
            "summary": transcript[:150],
            "scene": "",
            "marketing_prompt": transcript[:200],
        }

def clean_transcript_for_caption(transcript: str) -> str:
    if not transcript:
        return ""

    lines = re.split(r"[.!?]", transcript)
    cleaned = []
    seen = set()

    for line in lines:
        l = line.strip()
        if len(l.split()) < 4:
            continue
        key = l.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(l)

    return ". ".join(cleaned)[:900]


# -----------------------------------------------------
# 5. MAIN PIPELINE (Groq STT + Vision)
# -----------------------------------------------------
async def caption_from_video_file(video_filepath: str, platforms: List[str], client: Optional[Groq] = None) -> Dict[str, Any]:
    # 🔒 Defensive fix: flatten platforms if nested
    if platforms and isinstance(platforms[0], list):
        platforms = platforms[0]

    # 1. audio
    try:
        audio_path = await extract_audio_from_video(video_filepath)
    except Exception as e:
        logger.error("extract_audio_from_video failed: " + str(e))
        raise

    # 2. frames
    frame_paths = []
    try:
        frame_paths = extract_frames_from_video(video_filepath, fps=1, max_frames=6)
    except Exception as e:
        logger.warning(f"Frame extraction failed or produced no frames: {e}")
        frame_paths = []

    # 3. STT (Groq Whisper)
    transcript = ""
    try:
        transcript = await get_transcript_groq(audio_path)
    except Exception as e:
        logger.warning(f"Transcription failed or produced empty result: {e}")
        transcript = ""

    # 4. Visual analysis
    visual_result = {"visual_captions": [], "visual_summary": ""}
    try:
        visual_result = await analyze_frames_with_groq(frame_paths)
    except Exception as e:
        logger.warning(f"Visual analysis failed: {e}")
        visual_result = {"visual_captions": [], "visual_summary": ""}

    visual_summary = visual_result.get("visual_summary", "")
    visual_captions = visual_result.get("visual_captions", [])
    detected_texts = visual_result.get("detected_text", [])
    # ---- OCR COMBINATION (DEFINE ONCE) ----
    raw_ocr_text = "\n".join([t for t in detected_texts if t]).strip()
    ocr_text_combined = clean_ocr_text(raw_ocr_text)
    ocr_text_combined = await normalize_ocr_spelling(ocr_text_combined)

    # -----------------------------------------------------
    # 6. MERGE SIGNALS (TRANSCRIPT-FIRST, VISUAL-SAFE)
    # -----------------------------------------------------

    cleaned_transcript = clean_transcript_for_caption(transcript)
    merge_parts = []
    # 🧠 CASE 1: PEOPLE ARE TALKING → TRANSCRIPT DOMINATES
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

# 🎵 CASE 2: NO REAL SPEECH → VISUAL MODE
    else:
        if visual_summary:
            merge_parts.append(f"Scene: {visual_summary}")
        
        if visual_captions:
            frame_caps = "; ".join([cap for _, cap in visual_captions if cap][:5])
            merge_parts.append(f"Visual details: {frame_caps}")

        if ocr_text_combined:
            merge_parts.append(f"On-screen text: {ocr_text_combined}")

    marketing_prompt = "\n".join(merge_parts)


    # -----------------------------------------------------
    # 6b. IDENTIFY PERSON IN VIDEO (PUBLIC FIGURE)
    # -----------------------------------------------------

    identified_person = await identify_person_from_frames(
        visual_captions=visual_captions,
        ocr_texts=detected_texts,
        transcript=transcript
    )

    detected_person = (
        identified_person
        if identified_person and identified_person.lower() != "unknown"
       else None
    )



    if detected_person:
        marketing_prompt = (
        f"Context: The video includes {detected_person}.\n\n"
        + marketing_prompt
    )

    # -----------------------------------------------------
    # BUILD STRONG VIDEO EVIDENCE FOR CAPTION GENERATION
    # (THIS IS THE MAIN FIX)
    # -----------------------------------------------------

    visual_lines = [
        f"- {cap}" for _, cap in visual_captions if cap
    ]
    visual_block = "\n".join(visual_lines[:5]) or "None"
    evidence_prompt = f"""
    VIDEO EVIDENCE (MUST BE USED):

    TRANSCRIPT (EXCERPTS):
    {transcript[:800]}

    VISUAL FRAME DESCRIPTIONS:
    {visual_block}

ON-SCREEN TEXT (OCR):
{ocr_text_combined or "None"}

RULES:
- Base the caption ONLY on this evidence
- Do NOT generalize beyond the video
- If a public figure is mentioned, it must be supported by the transcript or visuals
"""



    # -----------------------------------------------------
    # 6c. INJECT IDENTITY INTO MARKETING PROMPT (IMPORTANT)
    # -----------------------------------------------------

    

    # 7. Generate keywords
    try:
        keywords = await generate_keywords_post(client, marketing_prompt)
    except Exception as e:
        logger.error("Keyword generation failed: " + str(e))
        keywords = ["", "", ""]

    # 8. Platform hashtags
    platform_hashtags: Dict[str, List[str]] = {}
    for p in platforms:
        try:
            try:
                tags = await fetch_platform_hashtags(client, keywords, p, marketing_prompt)
            except TypeError:
                tags = await fetch_platform_hashtags(client, keywords, p)
            platform_hashtags[p] = tags or []
        except Exception as e:
            logger.warning(f"fetch_platform_hashtags failed for {p}: {e}")
            platform_hashtags[p] = []

    # 9. Captions
    try:
        creator_context = f"""
You are writing a caption for a SHORT-FORM VIDEO (Reels / TikTok / Shorts).

IMPORTANT:
- This is NOT a summary task.
- Do NOT describe scenes or list events.
- Do NOT sound like a report or documentary.

WHAT MATTERS:
- Viewer curiosity
- Emotional tension
- Why this moment matters

VIDEO SIGNALS (for understanding only):
{marketing_prompt}

Focus on ONE strong idea or reaction.
"""

        captions_result = await generate_caption_post(
            creator_context,
            keywords,
            platforms
        )


        captions = captions_result.get("captions") if isinstance(captions_result, dict) else captions_result
    except Exception as e:
        logger.error(f"generate_caption_post failed: {e}", exc_info=True)
        captions = {p: f"Short {p} caption: {marketing_prompt[:100]}" for p in platforms}

    # 10. Cleanup
    try:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        pass
    try:
        if frame_paths:
            frames_dir = os.path.dirname(frame_paths[0])
            for f in glob.glob(os.path.join(frames_dir, "*")):
                try:
                    os.remove(f)
                except Exception:
                    pass
            try:
                os.rmdir(frames_dir)
            except Exception:
                pass
    except Exception:
        pass

    return {
        "detected_person": detected_person,
        "transcript": transcript,
        "visual_summary": visual_summary,
        "visual_captions": visual_captions,
        "detected_texts": detected_texts,
        "marketing_prompt": marketing_prompt,
        "keywords": keywords,
        "captions": captions,
        "platform_hashtags": platform_hashtags,
    }


# Example usage:
# import asyncio
# asyncio.run(caption_from_video_file(r"C:\path\to\video.mp4", ["instagram", "linkedin"]))

# NOTE:
# - Set GROQ_API_KEY_VIDEO_CAPTION env var to your Groq API key.
# - This file uses Groq's chat completions (Scout) for vision and Groq audio.transcriptions for STT.
# - Adjust model names if you want to use a different vision or summarization model.
