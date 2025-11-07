# stelle_backend/services/post_generator_service.py
# stelle_backend/services/post_generator_service.py
import asyncio
import json
import random
from typing import List, Dict, Any, Union, Tuple
from groq import AsyncGroq
from enum import Enum
import requests
import os # <-- FIX 29: Missing 'os' import

from config import logger, INTERNET_CLIENT_KEY, ASYNC_CLIENT_KEY
from services.ai_service import rate_limited_groq_call, query_internet_via_groq # <-- FIX 27, 28: Missing 'query_internet_via_groq'
from services.common_utils import get_current_datetime

# ... (rest of the file remains the same) ...

# Define Platforms Enum and platform_list (from original code's gen.py concept)
class Platforms(str, Enum):
    Instagram = "Instagram"
    X = "X (Twitter)"
    Reddit = "Reddit"
    LinkedIn = "LinkedIn"
    Facebook = "Facebook"
    
    platform_list = [Instagram, X, Reddit, LinkedIn, Facebook]

# NOTE: The original code re-uses a generic `client` (AsyncGroq) and `internet_client` (Groq) from ai_service.
# We will use the async client from `ai_service` for consistency.

# --- LLM Functions ---

async def generate_keywords_post(client: AsyncGroq, query: str) -> List[str]:
    """Generates seed keywords for content generation."""
    prompt = (
        f"Generate 3 seed keywords based on the following content description: {query}. "
        "Separate the keywords with commas. Output only keywords."
    )
    
    completion = await rate_limited_groq_call(
        client,
        model="llama-3.1-8b-instant", # Use a general model for speed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=100,
    )

    response = completion.choices[0].message.content
    seed_keywords = [kw.strip() for kw in response.split(",") if kw.strip()][:3]

    if len(seed_keywords) != 3:
        logger.warning(f"Adjusting seed keywords: {seed_keywords}")
        seed_keywords = (seed_keywords + [""] * 3)[:3]

    return seed_keywords

async def fetch_trending_hashtags_post(client: AsyncGroq, seed_keywords: list, platforms: list[Platforms]) -> list:
    """Fetches trending hashtags relevant to keywords and specified platforms."""
    hashtags = []
    platform_names = ", ".join([p.value for p in platforms])

    for keyword in seed_keywords:
        prompt = (
            f"Browse {platform_names}, and fetch trending hashtags related to {keyword}. "
            "Ensure they are trending and can boost SEO. Provide up to 10 unique hashtags separated by spaces. ONLY provide hashtags, no explanations."
        )
        
        # Use a model with browsing capability for better results
        response_content = await query_internet_via_groq(prompt)
        
        keyword_hashtags = [
            ht.strip().replace("#", "") for ht in response_content.split() if ht.strip()
        ]
        hashtags.extend(keyword_hashtags)

    # Deduplicate and limit to 30
    unique_hashtags = list(dict.fromkeys(hashtags))
    return [f"#{ht}" for ht in unique_hashtags[:30]]

async def fetch_seo_keywords_post(client: AsyncGroq, seed_keywords: list) -> list:
    """Fetches general SEO keywords based on seed keywords."""
    seo_keywords = []

    for keyword in seed_keywords:
        prompt = (
            f"Find the top 5 most searched SEO keywords related to {keyword} from top blogs and search results. "
            "Only provide keywords separated by commas. Output only keywords."
        )
        
        # Use a model with browsing capability
        response_content = await query_internet_via_groq(prompt)
        
        keyword_seo = [kw.strip() for kw in response_content.split(",") if kw.strip()]
        seo_keywords.extend(keyword_seo[:5])

    # Deduplicate and limit to 15
    return list(dict.fromkeys(seo_keywords))[:15]

async def generate_caption_post(
    client: AsyncGroq,
    query: str,
    seed_keywords: list,
    trending_hashtags: list,
    platforms: list[Platforms]
) -> Dict[str, str]:
    """Generates platform-optimized captions."""
    caption_dict = {}
    
    # Group prompts for all selected platforms
    platform_prompts = [
        (
            platform.value,
            f"Write a {platform.value} post caption (max 150 words) with a strong starting hook for content about '{query}'. "
            f"Use seed keywords: {', '.join(seed_keywords)}. "
            f"Incorporate the following hashtags naturally: {' '.join(trending_hashtags)}. "
            "Ensure the tone and length are appropriate for this platform."
        )
        for platform in platforms
    ]

    for platform_name, prompt in platform_prompts:
        try:
            completion = await rate_limited_groq_call(
                client,
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_completion_tokens=500,
            )
            caption_dict[platform_name] = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating caption for {platform_name}: {e}")
            caption_dict[platform_name] = f"Error generating caption for {platform_name}."
            
    return caption_dict

async def classify_post_type(client: AsyncGroq, prompt: str) -> str:
    """Classifies the post intent (e.g., informative, promotional, inspirational)."""
    classification_prompt = (
        f"Analyze the user's content prompt: '{prompt}'. "
        "Classify the intended post type into one word: 'Informative', 'Inspirational', 'Promotional', or 'Tutorial'. "
        "Output ONLY the word."
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.3,
            max_completion_tokens=10,
        )
        return completion.choices[0].message.content.strip().capitalize()
    except Exception as e:
        logger.error(f"Error classifying post type: {e}")
        return "Informative"

async def generate_html_code_post(client: AsyncGroq, prompt: str, post_type: str) -> str:
    """Generates clean HTML/CSS for a text-based post visualization (e.g., carousel card)."""
    html_prompt = (
        f"Generate a single-page HTML document for a text-based social media post about '{prompt}' (Type: {post_type}). "
        "Use modern CSS (tailwind-style classes in comments or inline) to create a visually appealing, dark-themed card or carousel slide design. "
        "The content should be concise and highly engaging. Include all CSS inline or in a `<style>` block. "
        "Output ONLY the complete HTML code starting with `<!DOCTYPE html>`."
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": html_prompt}],
            temperature=0.8,
            max_completion_tokens=3000,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating HTML post code: {e}")
        return "<html><body><h1>Error generating visual post content.</h1></body></html>"
    
# --- Pexels Media Service ---
# NOTE: The original code snippet did not define GROQ_API_KEY_PEXELS. Assuming PEXELS_API_KEY from .env
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

async def get_pexels_data(keywords: List[str], media_type: str) -> Dict[str, Any]:
    """Fetches video or photo data from Pexels based on keywords."""
    if not PEXELS_API_KEY:
        logger.error("PEXELS_API_KEY is missing.")
        return {}

    query = "+".join(keywords)
    endpoint = f"search/{media_type}s" if media_type == "photo" else "search/videos"
    url = f"https://api.pexels.com/{endpoint}?query={query}&per_page=5"

    try:
        async with requests.AsyncClient() as client:
            response = await client.get(
                url, headers={"Authorization": PEXELS_API_KEY}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Pexels API fetch failed for {media_type}: {e}")
        return {}

async def parse_media(media_data: Dict[str, Any], media_type: str) -> List[Dict[str, str]]:
    """Parses Pexels JSON response into a simplified list of media objects."""
    parsed_list = []
    
    if media_type == "photo" and "photos" in media_data:
        for photo in media_data["photos"]:
            parsed_list.append({
                "type": "photo",
                "src": photo["src"]["large"] if "src" in photo and "large" in photo["src"] else photo["url"],
                "alt": photo.get("alt", "Pexels Photo"),
                "artist": photo.get("photographer", "N/A")
            })
    elif media_type == "video" and "videos" in media_data:
        for video in media_data["videos"]:
            # Find a suitable video file link (e.g., medium quality MP4)
            video_link = None
            for file in video.get("video_files", []):
                if file.get("quality") == "hd" and file.get("link"):
                    video_link = file["link"]
                    break
            if not video_link:
                 for file in video.get("video_files", []):
                    if file.get("link"): # Fallback to any link
                        video_link = file["link"]
                        break
                        
            if video_link:
                parsed_list.append({
                    "type": "video",
                    "src": video_link,
                    "duration": f"{video.get('duration', 0)}s",
                    "artist": video.get("user", {}).get("name", "N/A")
                })

    return parsed_list