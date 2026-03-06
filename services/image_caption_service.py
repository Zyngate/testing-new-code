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
    PLATFORM_CTAS,
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

async def caption_from_image_file(image_filepath: str, platforms: List[str], client: Optional[Groq], autoposting: bool = False):
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

    # 3. Platform hashtags + 4. Captions (PARALLEL)
    async def get_hashtags_for_platform(p):
        try:
            tags = await fetch_platform_hashtags(client, keywords, p, marketing_prompt, autoposting=autoposting)
        except:
            tags = []
        return (p, tags)

    async def get_captions():
        try:
            captions_result = await generate_caption_post(marketing_prompt, keywords, platforms, autoposting=autoposting)
            return captions_result if isinstance(captions_result, dict) else {"captions": captions_result}
        except:
            return {"captions": {p: caption for p in platforms}}

    # Run all hashtag tasks + caption task in parallel
    import asyncio
    hashtag_tasks = [get_hashtags_for_platform(p) for p in platforms]
    all_results = await asyncio.gather(*hashtag_tasks, get_captions())

    # Extract results
    platform_hashtags = {}
    for result in all_results[:-1]:  # All except last (captions)
        p, tags = result
        platform_hashtags[p] = tags
    
    captions_data = all_results[-1]  # Last result is captions dict
    
    # Handle new combined format for non-autoposting
    if "platforms" in captions_data:
        # Already in combined format from generate_caption_post
        return {
            "caption_detected": caption,
            "scene": scene_text,
            "ocr_text": ocr_text,
            "objects": visual.get("objects", []),
            "actions": visual.get("actions", []),
            "marketing_prompt": marketing_prompt,
            "keywords": captions_data.get("keywords", keywords),
            "platforms": captions_data.get("platforms", {}),
        }
    
    # Legacy format handling for autoposting
    captions = captions_data.get("captions", {}) if isinstance(captions_data, dict) else captions_data
    titles = captions_data.get("titles", {}) if isinstance(captions_data, dict) else {}
    boards = captions_data.get("boards", {}) if isinstance(captions_data, dict) else {}
    platform_ctas = captions_data.get("ctas", {}) if isinstance(captions_data, dict) else {}

    # For NON-AUTOPOSTING: Return only caption + optional title per platform.
    # `caption` contains final composed text in strict order: caption -> hashtags -> CTA dump.
    if not autoposting:
        platforms_combined = {}
        for p in platforms:
            caption_text = captions.get(p, "")
            hashtags_list = platform_hashtags.get(p, [])[:10]  # Exactly 10 hashtags
            # Get ALL CTAs for the platform
            all_ctas = PLATFORM_CTAS.get(p, [])
            title_text = titles.get(p, "") if p in ("youtube", "pinterest") else ""
            
            # Build fully composed caption in strict order: caption -> hashtags -> CTA dump.
            composed_parts = []
            if caption_text:
                composed_parts.append(caption_text.strip())

            # Add hashtag dump after caption
            if hashtags_list:
                hashtags_str = " ".join(hashtags_list)
                composed_parts.append(hashtags_str)

            # Add full CTA dump at the end
            if all_ctas:
                cta_dump = "\n".join([cta.strip() for cta in all_ctas if cta and cta.strip()])
                if cta_dump:
                    composed_parts.append(cta_dump)
            
            # Join with double newlines for clean separation
            composed_caption = "\n\n".join(composed_parts)
            
            platforms_combined[p] = {
                "title": title_text if title_text else None,
                "caption": composed_caption,
            }
        return {
            "caption_detected": caption,
            "scene": scene_text,
            "ocr_text": ocr_text,
            "objects": visual.get("objects", []),
            "actions": visual.get("actions", []),
            "marketing_prompt": marketing_prompt,
            "keywords": keywords,
            "platforms": platforms_combined,
        }

    # For AUTOPOSTING: Keep original separate format
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
        "ctas": platform_ctas, 
        "titles": titles,
        "boards": boards,
    }
