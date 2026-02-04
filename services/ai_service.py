# stelle_backend/services/ai_service.py

import os
import re
import asyncio
import random
import logging
import json
import numpy as np
from typing import Union, Tuple, List, Dict, Any
from groq import Groq, AsyncGroq
from ratelimit import limits, sleep_and_retry
from sentence_transformers import SentenceTransformer

from database import (
    doc_index,
    code_index,
    file_doc_memory_map,
    code_memory_map,
    uploads_collection,
    memory_collection,
    goals_collection
)
from config import (
    CALLS_PER_MINUTE,
    PERIOD,
    FAISS_EMBEDDING_DIM,
    INTERNET_CLIENT_KEY,
    DEEPSEARCH_CLIENT_KEY,
    ASYNC_CLIENT_KEY,
    CONTENT_CLIENT_KEY,
    EXPLANATION_CLIENT_KEY,
    CLASSIFY_CLIENT_KEY,
    BROWSE_ENDPOINT_KEY,
    MEMORY_SUMMARY_KEY,
    PLANNING_KEY,
    GOAL_SETTING_KEY,
    GENERATE_API_KEYS,
    logger,
    GROQ_API_KEY_CAPTION,
    GROQ_API_KEY_VISUALIZE
)
from services.file_service import split_text_into_chunks
import itertools

# Build caption key pool for rotation (10 keys for parallel processing)
CAPTION_API_KEYS = []
for i in [None, "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9"]:
    key_name = f"GROQ_API_KEY_CAPTION{i if i else ''}"
    key_val = os.getenv(key_name)
    if key_val:
        CAPTION_API_KEYS.append(key_val)
if not CAPTION_API_KEYS:
    from config import BASE_GROQ_KEY
    CAPTION_API_KEYS = [BASE_GROQ_KEY]

logger.info(f"Caption key pool: {len(CAPTION_API_KEYS)} keys")

# Caption key rotation cycle
_caption_key_cycle = itertools.cycle(CAPTION_API_KEYS)

def get_caption_client_sync() -> Groq:
    """
    Return a sync Groq client using rotating caption API keys.
    Used for video transcription and frame analysis.
    """
    from groq import Groq
    key = next(_caption_key_cycle)
    logger.debug(f"Using caption key (sync): ...{key[-8:]}")
    return Groq(api_key=key, max_retries=2, timeout=120.0)


# ============================================================
# 🔵 1. GLOBAL RESOURCES (key resolution + clients)
# ============================================================

local_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Resolve keys robustly from multiple possible env var names and config fallbacks.
# This helps when .env uses either old names (INTERNET_CLIENT_KEY) or canonical names (GROQ_API_KEY_BROWSE).
from config import BASE_GROQ_KEY  # ensure we can fall back to base key if needed

def _resolve_key(*candidates: str) -> str:
    """
    Return the first non-empty key from a list of env/config candidate names or values.
    Each candidate can be either an env var name (prefixed 'env:') or a direct value.
    Example: _resolve_key('env:GROQ_API_KEY_BROWSE', INTERNET_CLIENT_KEY, 'env:BASE_GROQ_KEY')
    """
    for cand in candidates:
        if isinstance(cand, str) and cand.startswith("env:"):
            val = os.getenv(cand[4:], "") or None
            if val:
                return val
        else:
            if cand:
                return cand
    return ""


# Candidates order: prefer canonical GROQ env names, then legacy config names, then BASE_GROQ_KEY
resolved_internet_key = _resolve_key(
    "env:GROQ_API_KEY_BROWSE",
    "env:INTERNET_CLIENT_KEY",
    INTERNET_CLIENT_KEY,
    BASE_GROQ_KEY
)

resolved_deepsearch_key = _resolve_key(
    "env:GROQ_API_KEY_DEEPSEARCH",
    DEEPSEARCH_CLIENT_KEY,
    "env:GROQ_API_KEY_RESEARCHAGENT",
    BASE_GROQ_KEY
)

resolved_browse_endpoint_key = _resolve_key(
    "env:GROQ_API_KEY_BROWSE_ENDPOINT",
    BROWSE_ENDPOINT_KEY,
    BASE_GROQ_KEY
)

resolved_async_key = _resolve_key(
    "env:ASYNC_CLIENT_KEY",
    ASYNC_CLIENT_KEY,
    BASE_GROQ_KEY
)

# Log which keys are being used (only log truncated suffix for safety)
def _mask_key(k: str) -> str:
    if not k:
        return "<missing>"
    return f"...{k[-8:]}"

logger.info(f"Resolved Groq keys — internet: {_mask_key(resolved_internet_key)}, deepsearch: {_mask_key(resolved_deepsearch_key)}, browse_endpoint: {_mask_key(resolved_browse_endpoint_key)}, async: {_mask_key(resolved_async_key)}")

# Instantiate clients with resolved keys
try:
    internet_client = Groq(api_key=resolved_internet_key)
except Exception as e:
    logger.error(f"Failed to create internet_client Groq client: {e}")
    internet_client = None

try:
    deepsearch_client = Groq(api_key=resolved_deepsearch_key)
except Exception as e:
    logger.error(f"Failed to create deepsearch_client Groq client: {e}")
    deepsearch_client = None

try:
    client = AsyncGroq(api_key=resolved_async_key)
except Exception as e:
    logger.error(f"Failed to create async Groq client: {e}")
    client = None


# ============================================================
# 🔵 2. RATE LIMIT WRAPPER
# ============================================================

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
async def rate_limited_groq_call(
    client_or_model: Union[AsyncGroq, Groq, str],
    *args,
    **kwargs
) -> Any:
    if isinstance(client_or_model, str):
        client_to_use = client
        model = client_or_model
    else:
        client_to_use = client_or_model
        model = kwargs.pop("model", "llama-3.1-8b-instant")

    is_async = isinstance(client_to_use, AsyncGroq)

    try:
        if is_async:
            completion = await client_to_use.chat.completions.create(
                model=model, *args, **kwargs
            )
        else:
            completion = await asyncio.to_thread(
                client_to_use.chat.completions.create,
                model=model,
                *args,
                **kwargs
            )
        return completion

    except Exception as e:
        logger.error(f"Rate-limited Groq API call failed: {e}")
        raise


# ============================================================
# 🔵 3. DEDICATED CAPTION GENERATOR CLIENT (with key rotation)
# ============================================================

def get_caption_client() -> AsyncGroq:
    """
    Return an AsyncGroq client using rotating caption API keys.
    Distributes load across multiple keys to avoid rate limits.
    Disables SDK retry logic so we can rotate keys on 429 errors.
    """
    key = next(_caption_key_cycle)
    logger.debug(f"Using caption key: ...{key[-8:]}")
    return AsyncGroq(
        api_key=key,
        max_retries=0  # Disable SDK retries, we handle rotation ourselves
    )


async def groq_generate_text(model: str, prompt: str, system_msg: str = "You are a helpful assistant.", **kwargs) -> str:
    """
    Uses rotating caption clients to generate text with rate limit handling.
    Rotates through different API keys on 429 errors.
    Returns empty string on failure (caller should fallback).
    """
    max_retries = 6  # Try multiple keys if needed
    
    for attempt in range(max_retries):
        try:
            client_caption = get_caption_client()  # Gets next key in rotation
            response = await client_caption.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", 0.8),
                max_completion_tokens=kwargs.get("max_completion_tokens", 300),
                top_p=kwargs.get("top_p", 0.95),
                stream=False,
                timeout=60.0,  # Increased timeout
            )
            return response.choices[0].message.content or ""
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error (429) or timeout
            if "429" in str(e) or "rate_limit" in error_str or "timed out" in error_str or "timeout" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(f"API issue (attempt {attempt + 1}): {str(e)[:50]}... rotating to next key")
                    # Stagger delay to avoid overwhelming
                    await asyncio.sleep(1.0 + (attempt * 0.5))
                    continue
                else:
                    logger.error(f"All {max_retries} attempts exhausted")
                    return ""
            else:
                # Non-rate-limit error, log and fail
                logger.error(f"GROQ Caption Error: {e}")
                return ""
    
    return ""

# ============================================================
# 🔵 4. EXISTING CORE FUNCTIONS (unchanged except get_groq_client)
# ============================================================

async def get_groq_client() -> AsyncGroq:
    """
    IMPORTANT: for caption generation flows we want to use the dedicated caption key.
    This function now returns the caption client so all caption-related callers that
    call get_groq_client() (e.g. routes -> generate_keywords_post -> generate_caption_post)
    will consistently use GROQ_API_KEY_CAPTION.

    NOTE: If you have other non-caption parts of the app that relied on random GENERATE_API_KEYS,
    consider using a separate helper. For caption flows this ensures stability.
    """
    return get_caption_client()

def get_visualize_groq_client() -> Groq:
    """
    Dedicated Groq client for visualization flows.
    Keeps caption and visualization traffic isolated.
    """
    return Groq(api_key=GROQ_API_KEY_VISUALIZE)


async def generate_text_embedding(text: str | None) -> list:
    if not text:
        logger.warning("generate_text_embedding called with empty or None text; returning empty embedding.")
        return []
    try:
        embedding = await asyncio.to_thread(
            local_embedding_model.encode, text, convert_to_numpy=True
        )
        embedding_list = embedding.tolist()
        if len(embedding_list) != FAISS_EMBEDDING_DIM:
            logger.error(f"Embedding has unexpected length: {len(embedding_list)}")
            return []
        return embedding_list
    except Exception as e:
        logger.error(f"Local embedding generation error: {e}")
        return []


@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
async def query_internet_via_groq(query: str, return_sources: bool = False):
    try:
        completion = await asyncio.to_thread(
            internet_client.chat.completions.create,
            messages=[{"role": "user", "content": query}],
            model="compound-beta",
        )
        content = completion.choices[0].message.content

        if not return_sources:
            return content

        sources = []
        executed_tools = (
            getattr(completion.choices[0].message, "executed_tools", []) or []
        )
        for tool in executed_tools:
            if (
                tool.type == "search"
                and hasattr(tool, "search_results")
                and tool.search_results
            ):
                raw = tool.search_results
                hits = getattr(raw, "results", None) or raw.get("results", [])
                for result in hits:
                    title = result.get("title")
                    url = result.get("url")
                    if title and url:
                        sources.append({"title": title, "url": url})

        return content, sources

    except Exception as e:
        logger.error(f"Error querying Groq API: {e}")
        if return_sources:
            return "Error accessing internet information.", []
        return "Error accessing internet information."


async def content_for_website(content: str) -> str:
    prompt = (
        f"Summarize the following content concisely:\n\n{content}\n\n"
        "List key themes and provide a brief final summary."
    )
    try:
        client_local = Groq(api_key=CONTENT_CLIENT_KEY)
        response = await asyncio.to_thread(
            client_local.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are a content analysis expert."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=700,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Content summary error: {e}")
        return "Error generating content summary."


async def detailed_explanation(content: str) -> str:
    prompt = (
        "Provide a detailed explanation by listing key themes and challenges, "
        "and then generate a comprehensive summary of the content below:\n\n" + content
    )
    try:
        client_local = Groq(api_key=EXPLANATION_CLIENT_KEY)
        response = await asyncio.to_thread(
            client_local.chat.completions.create,
            messages=[
                {"role": "system", "content": "You are an expert analysis assistant."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=700,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Detailed explanation error: {e}")
        return "Error generating detailed explanation."


async def classify_prompt(prompt: str) -> str:
    try:
        client_local = Groq(api_key=CLASSIFY_CLIENT_KEY)
        response = await asyncio.to_thread(
            client_local.chat.completions.create,
            messages=[
                {
                    "role": "system",
                    "content": "Determine if this query requires real time research. Respond with 'research' or 'no research'.",
                },
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
        )
        reply = response.choices[0].message.content.strip().lower()
        return reply
    except Exception as e:
        logger.error(f"Classify prompt error: {e}")
        return "no research"


async def retrieve_multimodal_context(query: str, session_id: str, filenames: list[str] = None):
    try:
        embedding = await generate_text_embedding(query)
        contexts = []
        used_filenames = set()

        if not embedding or len(embedding) != FAISS_EMBEDDING_DIM:
            return "", set()

        query_vector = np.array(embedding, dtype="float32").reshape(1, -1)

        # Document index
        if doc_index.ntotal > 0:
            k = 10
            distances, indices = doc_index.search(query_vector, k)

            for rank, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                meta = file_doc_memory_map.get(idx)

                if meta:
                    filename = meta["filename"]

                    if filenames and filename not in filenames:
                        continue

                    chunk = await uploads_collection.find_one(
                        {
                            "user_id": meta["user_id"],
                            "session_id": session_id,
                            "filename": filename,
                            "chunk_index": meta.get("chunk_index"),
                        }
                    )

                    if chunk and chunk.get("query_count", 0) < 15:
                        snippet = f"From {filename}: {meta['text_snippet']}"
                        contexts.append((distances[0][rank], snippet, filename))
                        used_filenames.add(filename)

        # Code index
        if code_index.ntotal > 0:
            k = 8
            distances, indices = code_index.search(query_vector, k)

            for rank, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                meta = code_memory_map.get(idx)

                if meta:
                    filename = meta["filename"]

                    if filenames and filename not in filenames:
                        continue

                    seg = await uploads_collection.find_one(
                        {
                            "user_id": meta["user_id"],
                            "session_id": session_id,
                            "filename": filename,
                            "segment_name": meta.get("segment_name"),
                        }
                    )

                    if seg and seg.get("query_count", 0) < 15:
                        snippet = f"From {filename}: {meta['text_snippet']}"
                        contexts.append((distances[0][rank], snippet, filename))
                        used_filenames.add(filename)

        final_contexts = sorted(contexts, key=lambda x: x[0])
        context_text = "\n\n".join([c[1] for c in final_contexts])

        return context_text, used_filenames

    except Exception as e:
        logger.error(f"Error during multimodal retrieval: {e}")
        return "", set()


async def efficient_summarize(previous_summary: str, new_messages: list, user_id: str, max_summary_length: int = 500):
    try:
        user_queries = "\n".join([m["content"] for m in new_messages if m["role"] == "user"])

        context_text = f"Previous Summary:\n{previous_summary}\n\nNew User Queries:\n{user_queries}"

        active_goals = await goals_collection.find({"user_id": user_id, "status": "active"}).to_list(None)

        goals_text = ""
        for goal in active_goals:
            goals_text += f"- {goal['title']}\n"
            for task in goal["tasks"]:
                goals_text += f"  - {task['title']}\n"

        context_text += "\n" + goals_text

        prompt = (
            f"Summarize the following context in under {max_summary_length} characters:\n\n{context_text}"
        )

        client_local = Groq(api_key=MEMORY_SUMMARY_KEY)

        response = await asyncio.to_thread(
            client_local.chat.completions.create,
            messages=[
                {"role": "system", "content": "You summarize user behavior."},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.1-8b-instant",
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Memory summarization error: {e}")
        return previous_summary


async def store_long_term_memory(user_id: str, session_id: str, new_messages: list):
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        previous = mem_entry.get("summary", "") if mem_entry else ""

        new_summary = await efficient_summarize(previous, new_messages, user_id)
        new_vector = await generate_text_embedding(new_summary)

        if not new_vector or len(new_vector) != FAISS_EMBEDDING_DIM:
            return

        new_vector_np = np.array(new_vector).reshape(1, -1)

        idx = doc_index.ntotal
        doc_index.add(new_vector_np)

        update_data = {
            "summary": new_summary,
            "session_id": session_id,
            "vector": new_vector,
        }

        if mem_entry:
            await memory_collection.update_one({"user_id": user_id}, {"$set": update_data})
        else:
            await memory_collection.insert_one({"user_id": user_id, **update_data})

    except Exception as e:
        logger.error(f"store_long_term_memory error: {e}")


async def clarify_query(query: str) -> str:
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": f"Refine this query: {query}"}],
        )
        return completion.choices[0].message.content.strip()
    except:
        return query


async def clarify_response(query: str) -> str:
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": f"Interpret: {query}"}],
        )
        return completion.choices[0].message.content.strip()
    except:
        return query


async def generate_keywords(clarified_query: str) -> List[str]:
    try:
        prompt = (
            f"Generate 1-3 Google search keywords from: {clarified_query}"
        )
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
        )
        text = completion.choices[0].message.content.strip()
        parts = re.split(r'[, \n]+', text)
        return [p for p in parts if p][:3]
    except:
        return [clarified_query]


async def understanding_query(clarify_response: str) -> str:
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": f"Understand this: {clarify_response}"}],
        )
        return completion.choices[0].message.content.strip()
    except:
        return clarify_response


async def generate_subqueries(main_query: str, num_subqueries: int = 3):
    try:
        prompt = (
            f"Generate {num_subqueries} subqueries for: {main_query}"
        )
        completion = await rate_limited_groq_call(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        lines = [l.strip() for l in completion.choices[0].message.content.split("\n")]
        return [l for l in lines if l][:num_subqueries]
    except:
        return []


async def synthesize_result(main_query: str, contents: list[str], max_context: int = 4000):
    try:
        trimmed = " ".join([c[:1000] for c in contents])[:max_context]
        prompt = (
            f"Synthesize information for query: {main_query}\n\n{trimmed}"
        )
        completion = await rate_limited_groq_call(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        return completion.choices[0].message.content.strip()
    except:
        return "Error generating summary."

# ============================================================
# 🔵 5. DeepSearch + Visualization (Required by chat_routes.py)
# ============================================================
async def query_deepsearch(query: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Clean, depth-first DeepSearch.
    Uses reasoning for concepts and browsing only when needed.
    """

    try:
        client = AsyncGroq(api_key=resolved_deepsearch_key)

        # ---- DEPTH ROUTER ----
        philosophy_keywords = [
            "ikigai", "purpose", "meaning", "life",
            "happiness", "mindset", "philosophy", "psychology"
        ]

        q = query.lower()
        needs_browsing = not any(k in q for k in philosophy_keywords)

        model = "compound-beta" if needs_browsing else "llama-3.3-70b-versatile"

        # ---- STRUCTURE-FIRST SYSTEM PROMPT ----
        system_prompt = (
            "You must answer in a deep, structured, human way.\n\n"
            "Follow this structure strictly:\n"
            "1. Start with a clear, elegant core definition.\n"
            "2. Expand with cultural, historical, or psychological context.\n"
            "3. Explicitly address common misconceptions or shallow interpretations.\n"
            "4. Provide practical, real-world applications or steps.\n"
            "5. End with a simple, relatable example or insight.\n\n"
            "Avoid Wikipedia tone. Write for an intelligent human."
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.4,
            max_completion_tokens=1500,
        )

        content = response.choices[0].message.content.strip()
        return content, []

    except Exception as e:
        logger.error(f"DeepSearch failed: {e}")
        return "DeepSearch unavailable.", []

async def generate_thinking_steps(prompt: str) -> list[str]:
    """
    Generates user-visible reflective thinking.
    Not a plan. Not instructions. Just interpretation.
    """

    client = AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are simulating a natural inner monologue.\n\n"
                    "Rules:\n"
                    "- Write in first-person.\n"
                    "- Sound thoughtful and human.\n"
                    "- Reflect on scope and intent.\n"
                    "- Convert ambiguity into confident framing.\n"
                    "- Do NOT ask questions (no question marks).\n"
                    "- Do NOT describe actions or plans.\n"
                    "- Do NOT provide answers or conclusions.\n"
                    "- This is internal reflection, not a conversation.\n\n"
                    "Style example:\n"
                    "\"The topic is broad and complex. A structured, high-level framing "
                    "helps capture its essence without oversimplifying. Balancing context "
                    "and nuance creates clarity while leaving room for deeper exploration.\""
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.35,
        max_completion_tokens=220,
    )

    text = response.choices[0].message.content.strip()

    # HARD CHARACTER CONTROL
    MIN_CHARS = 600
    MAX_CHARS = 900

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0]

    return [text]




async def visualize_content(context_text: str) -> Dict[str, Any]:
    """
    Fast + reliable visualization using llama-3.3-70b-versatile.
    """
    try:
        prompt = (
            "Analyze the following content and return JSON:\n"
            "summary: short overview\n"
            "themes: list of main themes\n"
            "insights: 3–5 actionable insights\n\n"
            f"CONTENT:\n{context_text}"
        )

        completion = await asyncio.to_thread(
            internet_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        raw = completion.choices[0].message.content.strip()

        try:
            return json.loads(raw)
        except:
            return {
                "summary": raw[:400],
                "themes": [],
                "insights": []
            }

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return {"summary": "", "themes": [], "insights": []}
