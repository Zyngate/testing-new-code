# stelle_backend/services/image_caption_service.py

import os
import base64
import json
from groq import Groq
from typing import List, Dict, Any, Optional
from config import logger
from services.ai_service import groq_generate_text
from services.post_generator_service import (
    generate_keywords_post,
    fetch_platform_hashtags,
    generate_caption_post,
)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SUMMARIZE_MODEL = "gpt-4o-mini"

def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return f"data:image/png;base64," + base64.b64encode(b).decode()

async def analyze_image_with_groq(image_path: str) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY_CAPTION")
    client = Groq(api_key=api_key)

    data_url = image_to_data_url(image_path)

    prompt_content = [
        {
            "type": "text",
            "text": (
                "Analyze this image and return JSON with keys:\n"
                "caption,\n"
                "ocr_text,\n"
                "objects,\n"
                "actions,\n"
                "scene"
            )
        },
        {
            "type": "image_url",
            "image_url": {"url": data_url}
        }
    ]

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt_content}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    parsed = response.choices[0].message.content
    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    return {
        "caption": parsed.get("caption", ""),
        "ocr_text": parsed.get("ocr_text", ""),
        "objects": parsed.get("objects", []),
        "actions": parsed.get("actions", []),
        "scene": parsed.get("scene", "")
    }

async def caption_from_image_file(image_filepath: str, platforms: List[str], client: Optional[Groq]):
    # 1. Visual analysis
    visual = await analyze_image_with_groq(image_filepath)

    caption = visual.get("caption", "")
    scene_text = visual.get("scene", "")
    ocr_text = visual.get("ocr_text", "")

    marketing_prompt = "\n".join([
        caption,
        f"Scene: {scene_text}" if scene_text else "",
        f"OCR: {ocr_text}" if ocr_text else ""
    ]).strip()

    # 2. Generate keywords
    try:
        keywords = await generate_keywords_post(client, marketing_prompt)
    except:
        keywords = ["", "", ""]

    # 3. Platform hashtags
    platform_hashtags = {}
    for p in platforms:
        try:
            tags = await fetch_platform_hashtags(client, keywords, p, marketing_prompt)
        except:
            tags = []
        platform_hashtags[p] = tags

    # 4. Captions
    try:
        captions_result = await generate_caption_post(marketing_prompt, keywords, platforms)
        captions = captions_result.get("captions") if isinstance(captions_result, dict) else captions_result
    except:
        captions = {p: caption for p in platforms}

    return {
        "caption_detected": caption,
        "scene": scene_text,
        "ocr_text": ocr_text,
        "objects": visual.get("objects", []),
        "actions": visual.get("actions", []),
        "marketing_prompt": marketing_prompt,
        "keywords": keywords,
        "captions": captions,
        "platform_hashtags": platform_hashtags,
    }
