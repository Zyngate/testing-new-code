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
    GROQ_API_KEY_CAPTION
)
from services.file_service import split_text_into_chunks


# ============================================================
# 🔵 1. GLOBAL RESOURCES
# ============================================================

local_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

internet_client = Groq(api_key=INTERNET_CLIENT_KEY)
deepsearch_client = Groq(api_key=DEEPSEARCH_CLIENT_KEY)
client = AsyncGroq(api_key=ASYNC_CLIENT_KEY)


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
# 🔵 3. DEDICATED CAPTION GENERATOR CLIENT (ADDED)
# ============================================================

def get_caption_client():
    key = GROQ_API_KEY_CAPTION
    if not key:
        logger.error("GROQ_API_KEY_CAPTION missing — using fallback key")
    return AsyncGroq(api_key=key)


async def groq_generate_text(
    model: str,
    prompt: str,
    system_msg: str = "You are a helpful assistant."
):
    """
    Safe wrapper for caption text generation only.
    Does NOT affect any other logic in ai_service.py
    """
    try:
        client = get_caption_client()

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_completion_tokens=300,
            top_p=0.95,
            stream=False
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"GROQ Caption Error: {e}")
        return ""


# ============================================================
# 🔵 4. EXISTING CORE FUNCTIONS
# ============================================================

async def get_groq_client() -> AsyncGroq:
    selected_key = random.choice(GENERATE_API_KEYS)
    return AsyncGroq(api_key=selected_key)


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
        client = Groq(api_key=CONTENT_CLIENT_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
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
        client = Groq(api_key=EXPLANATION_CLIENT_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
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
        client = Groq(api_key=CLASSIFY_CLIENT_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
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

        client = Groq(api_key=MEMORY_SUMMARY_KEY)

        response = await asyncio.to_thread(
            client.chat.completions.create,
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
