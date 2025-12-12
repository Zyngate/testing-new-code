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

from config import logger
from config import GROQ_API_KEY_VIDEO_CAPTION
from groq import Groq
from services.ai_service import groq_generate_text
from services.post_generator_service import (
    generate_keywords_post,
    fetch_platform_hashtags,
    generate_caption_post,
)

# LLM / Vision model names
VISION_MODEL = "llava-v1.5-7b"
STT_MODEL = "whisper-large-v3"
SUMMARIZE_MODEL = "gpt-4o-mini"

# TEMP directory
TEMP_DIR = Path(os.path.join(os.getcwd(), "tmp_stelle_video"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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

    cmd = f'ffmpeg -y -i {video_quoted} -vn -ac 1 -ar 16000 -sample_fmt s16 {audio_quoted} -hide_banner -loglevel error'
    logger.info(f"Running ffmpeg to extract audio: {cmd}")
    returncode, out, err = await asyncio.to_thread(run_cmd, cmd)
    if returncode != 0:
        logger.error(f"ffmpeg failed extracting audio: {err.strip()}")
        raise RuntimeError("Audio extraction failed. Ensure ffmpeg is installed and video file exists.")
    return out_audio

# -----------------------------------------------------
# 1b. EXTRACT FRAMES
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
# 2. TRANSCRIPTION USING GROQ
# -----------------------------------------------------
async def get_transcript_groq(audio_path: str) -> str:
    api_key = GROQ_API_KEY_VIDEO_CAPTION
    if not api_key:
        raise RuntimeError("GROQ_API_KEY_VIDEO_CAPTION is not set in environment.")

    client = Groq(api_key=api_key)

    def _blocking_transcribe(path: str) -> str:
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(file=f, model=STT_MODEL)
        if hasattr(resp, "text"):
            return (resp.text or "").strip()
        if isinstance(resp, dict):
            return (resp.get("text") or resp.get("transcript") or "").strip()
        return str(resp)

    try:
        transcript = await asyncio.to_thread(_blocking_transcribe, audio_path)
        return transcript
    except Exception as e:
        logger.error(f"Groq STT transcription failed: {e}")
        return ""

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# -----------------------------------------------------
# 3. FRAME ANALYSIS
# -----------------------------------------------------
async def analyze_frames_with_groq(frame_paths: List[str]) -> Dict[str, Any]:
    api_key = GROQ_API_KEY_VIDEO_CAPTION
    if not api_key:
        raise RuntimeError("GROQ_API_KEY_VIDEO_CAPTION is not set.")

    client = Groq(api_key=api_key)

    visual_captions = []
    detected_text_list = []
    objects_list = []
    actions_list = []

    for fp in frame_paths:
        try:
            data_url = image_to_data_url(fp)

            prompt_content = [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image and return a JSON object with keys:\n"
                        "caption\nocr_text\nobjects\nactions\nscene\n"
                    )
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ]

            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt_content}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            parsed = response.choices[0].message.content
            if isinstance(parsed, str):
                parsed_json = json.loads(parsed)
            else:
                parsed_json = parsed

            caption = parsed_json.get("caption", "")
            ocr_text = parsed_json.get("ocr_text", "")
            objs = parsed_json.get("objects", [])
            acts = parsed_json.get("actions", [])

            visual_captions.append((os.path.basename(fp), caption.strip()))

            if ocr_text:
                detected_text_list.append(ocr_text.strip())
            objects_list.extend([str(o).strip() for o in objs if o])
            actions_list.extend([str(a).strip() for a in acts if a])

        except Exception as e:
            logger.warning(f"Vision analysis failed for {fp}: {e}")
            visual_captions.append((os.path.basename(fp), ""))

    all_caps = " | ".join([cap for _, cap in visual_captions if cap])
    objects_unique = list(dict.fromkeys(objects_list))
    actions_unique = list(dict.fromkeys(actions_list))

    visual_summary = ""
    if all_caps:
        prompt = (
            f"Summarize this visually:\nCaptions:{all_caps}\nObjects:{objects_unique}\nActions:{actions_unique}\n"
            "One line max."
        )
        try:
            ai_resp = groq_generate_text(SUMMARIZE_MODEL, prompt)
            if hasattr(ai_resp, "__await__"):
                ai_resp = await ai_resp
            visual_summary = ai_resp.strip()
        except:
            visual_summary = all_caps.split("|")[0][:120]

    return {
        "visual_captions": visual_captions,
        "visual_summary": visual_summary,
        "detected_text": detected_text_list,
        "objects": objects_unique,
        "actions": actions_unique,
    }

# -----------------------------------------------------
# 4. TRANSCRIPT SUMMARY
# -----------------------------------------------------
async def summarize_transcript_for_caption(transcript: str) -> Dict[str, str]:
    prompt = f"""
You are a marketing strategist. Summarize the transcript.

Transcript:
\"\"\"{transcript}\"\"\" 

Output JSON: summary, scene, marketing_prompt
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
        except:
            return {
                "summary": ai_resp.strip().split("\n")[0],
                "scene": "",
                "marketing_prompt": ai_resp.strip(),
            }

    except Exception as e:
        logger.error(f"Transcript summary failed: {e}")
        return {
            "summary": transcript[:150],
            "scene": "",
            "marketing_prompt": transcript[:200],
        }

# -----------------------------------------------------
# 5. MAIN PIPELINE
# -----------------------------------------------------
async def caption_from_video_file(video_filepath: str, platforms: List[str], client: Optional[Groq] = None) -> Dict[str, Any]:

    # 1 audio
    audio_path = await extract_audio_from_video(video_filepath)

    # 2 frames
    try:
        frame_paths = extract_frames_from_video(video_filepath, fps=1, max_frames=6)
    except:
        frame_paths = []

    # 3 transcript
    transcript = await get_transcript_groq(audio_path)

    # 4 visual analysis
    visual_result = await analyze_frames_with_groq(frame_paths)

    visual_summary = visual_result.get("visual_summary", "")
    visual_captions = visual_result.get("visual_captions", [])
    detected_texts = visual_result.get("detected_text", [])

    # 5 transcript summary
    summary_obj = await summarize_transcript_for_caption(transcript)
    text_summary = summary_obj["summary"]
    text_scene = summary_obj["scene"]
    marketing_prompt_text = summary_obj["marketing_prompt"] or text_summary

    # 6 merge signals
    ocr_text_combined = "\n".join(detected_texts)
    merge_parts = [marketing_prompt_text]
    if visual_summary:
        merge_parts.append(f"Visual: {visual_summary}")
    if ocr_text_combined:
        merge_parts.append(f"OCR: {ocr_text_combined}")

    marketing_prompt = "\n".join([m for m in merge_parts if m])

    # 7 keywords
    try:
        keywords = await generate_keywords_post(client, marketing_prompt)
    except:
        keywords = ["", "", ""]

    # 8 hashtags
    platform_hashtags = {}
    for p in platforms:
        try:
            tags = await fetch_platform_hashtags(client, keywords, p, marketing_prompt)
            platform_hashtags[p] = tags or []
        except:
            platform_hashtags[p] = []

    # -----------------------------------------------------
    # ✅ FIXED INDENTATION BLOCK — captions generation
    # -----------------------------------------------------
    captions = {}
    for p in platforms:
        try:
            single_caption_result = await generate_caption_post(
                marketing_prompt, keywords, [p]
            )
            if isinstance(single_caption_result, dict):
                captions[p] = single_caption_result.get("captions", {}).get(p)
            else:
                captions[p] = single_caption_result
        except Exception as e:
            logger.error(f"Caption generation failed for {p}: {e}")
            captions[p] = f"Short {p} caption: {marketing_prompt[:100]}"

    # 10 cleanup
    try:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
    except:
        pass

    try:
        if frame_paths:
            frames_dir = os.path.dirname(frame_paths[0])
            for f in glob.glob(os.path.join(frames_dir, "*")):
                try:
                    os.remove(f)
                except:
                    pass
            try:
                os.rmdir(frames_dir)
            except:
                pass
    except:
        pass

    return {
        "transcript": transcript,
        "text_summary": text_summary,
        "text_scene": text_scene,
        "visual_summary": visual_summary,
        "visual_captions": visual_captions,
        "detected_texts": detected_texts,
        "marketing_prompt": marketing_prompt,
        "keywords": keywords,
        "captions": captions,
        "platform_hashtags": platform_hashtags,
    }
