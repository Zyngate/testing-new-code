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
    logger
)
from services.file_service import split_text_into_chunks # Import utility function

# --- Global/Shared Resources ---
local_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# Initializing Groq clients using selected keys from config.py
internet_client = Groq(api_key=INTERNET_CLIENT_KEY)
deepsearch_client = Groq(api_key=DEEPSEARCH_CLIENT_KEY)
client = AsyncGroq(api_key=ASYNC_CLIENT_KEY)

# --- Rate Limiting Decorator ---
@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
async def rate_limited_groq_call(
    client_or_model: Union[AsyncGroq, Groq, str],
    *args,
    **kwargs
) -> Any:
    """Rate-limited wrapper for Groq client calls (handles both sync and async clients)."""
    if isinstance(client_or_model, str):
        # Allow passing the model name directly and use the main async client
        client_to_use = client
        model = client_or_model
    else:
        client_to_use = client_or_model
        model = kwargs.pop("model", "llama-3.1-8b-instant") # Default model

    # Determine if the client is synchronous or asynchronous
    is_async = isinstance(client_to_use, AsyncGroq)

    try:
        if is_async:
            completion = await client_to_use.chat.completions.create(model=model, *args, **kwargs)
        else:
            # Run synchronous call in a separate thread
            completion = await asyncio.to_thread(
                client_to_use.chat.completions.create, model=model, *args, **kwargs
            )
        return completion
    except Exception as e:
        logger.error(f"Rate-limited Groq API call failed: {e}")
        raise

# --- Core LLM & Utility Functions ---

async def get_groq_client() -> AsyncGroq:
    """Returns a randomly selected Groq client for streaming/non-critical tasks."""
    selected_key = random.choice(GENERATE_API_KEYS)
    return AsyncGroq(api_key=selected_key)

async def generate_text_embedding(text: str | None) -> list:
    """Generates an embedding vector using the local SentenceTransformer model."""
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
async def query_internet_via_groq(
    query: str, return_sources: bool = False
) -> Union[str, Tuple[str, List[dict]]]:
    """Sends query to Groq with internet browsing capability."""
    try:
        completion = await asyncio.to_thread(
            internet_client.chat.completions.create,
            messages=[{"role": "user", "content": query}],
            model="compound-beta", # Assumed model with browsing capability
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
                    title = result.get("title") if isinstance(result, dict) else getattr(result, "title", None)
                    url = result.get("url") if isinstance(result, dict) else getattr(result, "url", None)

                    if title and url:
                        sources.append({"title": title, "url": url})

        return content, sources

    except Exception as e:
        logger.error(f"Error querying Groq API (internet): {e}")
        if return_sources:
            return "Error accessing internet information.", []
        return "Error accessing internet information."


async def content_for_website(content: str) -> str:
    """Summarizes content concisely for website use."""
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
    """Provides a detailed explanation/summary."""
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
    """Determines if a query requires real-time research."""
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
        logger.info(f"Classify prompt response: {reply}")
        return reply
    except Exception as e:
        logger.error(f"Classify prompt error: {e}")
        return "no research"

async def retrieve_multimodal_context(
    query: str, session_id: str, filenames: list[str] = None
) -> tuple[str, set]:
    """Retrieves relevant text/code chunks from FAISS vector stores."""
    try:
        embedding = await generate_text_embedding(query)
        contexts = []
        used_filenames = set()

        if not embedding or len(embedding) != FAISS_EMBEDDING_DIM:
            return "", set()

        query_vector = np.array(embedding, dtype="float32").reshape(1, -1)
        
        # 1. Document Index Search
        if doc_index.ntotal > 0:
            k = 10
            distances, indices = doc_index.search(query_vector, k)
            
            for rank, idx in enumerate(indices[0]):
                if idx < 0: continue # Invalid index
                meta = file_doc_memory_map.get(idx)
                
                if meta and (
                    meta.get("session_id") == session_id or meta.get("session_id") != session_id # Allow context from other sessions temporarily
                ):
                    filename = meta["filename"]
                    if filenames and filename not in filenames:
                        continue
                        
                    # Fetch chunk details from MongoDB
                    chunk = await uploads_collection.find_one(
                        {
                            "user_id": meta["user_id"],
                            "session_id": session_id,
                            "filename": filename,
                            "chunk_index": meta.get("chunk_index"),
                        }
                    )
                    
                    # Heuristic to combine related chunks and limit max usage
                    if chunk and chunk.get("query_count", 0) < 15:
                        snippet = f"From {filename} (Chunk {meta.get('chunk_index', 'N/A')}):\n{meta['text_snippet']}"
                        contexts.append((distances[0][rank], snippet, filename))
                        used_filenames.add(filename)
                        
        # 2. Code Index Search
        if code_index.ntotal > 0:
            k = 8
            distances, indices = code_index.search(query_vector, k)
            
            for rank, idx in enumerate(indices[0]):
                if idx < 0: continue # Invalid index
                meta = code_memory_map.get(idx)
                
                if meta and meta.get("session_id") == session_id:
                    filename = meta["filename"]
                    if filenames and filename not in filenames:
                        continue
                        
                    # Fetch segment details from MongoDB
                    segment_chunk = await uploads_collection.find_one(
                        {
                            "user_id": meta["user_id"],
                            "session_id": session_id,
                            "filename": filename,
                            "segment_name": meta.get("segment_name"),
                        }
                    )
                    
                    if segment_chunk and segment_chunk.get("query_count", 0) < 15:
                        snippet = f"From {filename} (Code Segment: {meta.get('segment_name', 'N/A')}):\n{meta['text_snippet']}"
                        contexts.append((distances[0][rank], snippet, filename))
                        used_filenames.add(filename)

        # Sort combined contexts by distance (closer matches first)
        final_contexts = sorted(contexts, key=lambda x: x[0])
        # Join the snippet strings (only keep the snippet part)
        context_text = "\n\n".join([c[1] for c in final_contexts])
        
        return context_text, used_filenames
    except Exception as e:
        logger.error(f"Error during multimodal retrieval: {e}")
        return "", set()

async def efficient_summarize(
    previous_summary: str,
    new_messages: list,
    user_id: str,
    max_summary_length: int = 500,
) -> str:
    """Generates a concise summary for long-term memory."""
    user_queries = "\n".join(
        [msg["content"] for msg in new_messages if msg["role"] == "user"]
    )
    context_text = f"User ID: {user_id}\n"
    if previous_summary:
        context_text += f"Previous Summary:\n{previous_summary}\n\n"
    context_text += f"New User Queries:\n{user_queries}\n\n"
    
    # Include active goals for context
    active_goals = await goals_collection.find(
        {"user_id": user_id, "status": "active"}
    ).to_list(None)
    goals_context = ""
    if active_goals:
        goals_context = "User's current goals and tasks:\n"
        for goal in active_goals:
            goals_context += f"- Goal: {goal['title']} ({goal['status']})\n"
            for task in goal["tasks"]:
                goals_context += f"  - Task: {task['title']} ({task['status']})\n"
    context_text += goals_context
    
    summary_prompt = (
        f"Based on the following context, generate a concise summary (max {max_summary_length} characters) "
        f"that captures the user's interests, style, and ongoing goals:\n\n{context_text}"
    )
    try:
        client = Groq(api_key=MEMORY_SUMMARY_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI that creates personalized conversation summaries.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            model="llama-3.1-8b-instant",
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Long-term memory summarization error: {e}")
        return previous_summary if previous_summary else "Summary unavailable."


async def store_long_term_memory(user_id: str, session_id: str, new_messages: list):
    """Updates the user's long-term memory summary and FAISS vector."""
    from database import user_memory_map # Import dynamically to avoid circular dependency
    from datetime import datetime, timezone

    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        previous_summary = mem_entry.get("summary", "") if mem_entry else ""
        new_summary = await efficient_summarize(previous_summary, new_messages, user_id)
        new_vector = await generate_text_embedding(new_summary)
        
        if not new_vector or len(new_vector) != FAISS_EMBEDDING_DIM:
            logger.warning(f"Skipping memory update for {user_id}: invalid embedding generated.")
            return

        new_vector_np = np.array(new_vector, dtype="float32").reshape(1, -1)
        
        if user_id in user_memory_map:
            # 1. Remove old vector from FAISS
            old_index = user_memory_map.pop(user_id)
            # FAISS doesn't directly support index removal in this way for FlatL2, 
            # but we assume a robust underlying index implementation or simply add the new one.
            # For FlatL2 in production, a full rebuild might be needed, but for simplicity here:
            pass # Skipping explicit removal for performance, will add the new one below
        
        # 2. Add new vector to FAISS
        idx = doc_index.ntotal
        doc_index.add(new_vector_np)
        user_memory_map[user_id] = idx

        # 3. Update or insert into MongoDB
        update_data = {
            "summary": new_summary,
            "session_id": session_id,
            "vector": new_vector,
            "timestamp": datetime.now(timezone.utc),
        }
        
        if mem_entry:
            await memory_collection.update_one(
                {"user_id": user_id},
                {"$set": update_data},
            )
        else:
            await memory_collection.insert_one({"user_id": user_id, **update_data})

        logger.info(f"Long-term memory updated for user {user_id}. New FAISS index: {idx}")
    except Exception as e:
        logger.error(f"Error storing long-term memory: {e}")


# --- Deep Search Helpers ---
async def clarify_query(query: str) -> str:
    """Refines and optimizes the user's query into an effective search prompt."""
    prompt = (
        "You are an expert prompt engineer. Refine and optimize the user’s original query "
        "into the most effective, concise search prompt possible, considering the user's context.\n"
        f"Original query: {query}\n"
        "Return ONLY the optimized query—no explanations or extra text."
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=100,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error clarifying query: {e}")
        return query

async def clarify_response(query: str) -> str:
    """Generates a conversational response about the next steps for the user query."""
    prompt = (
        "You are an expert person who understands the user and returns what you understand by the user query and what your next steps are to address the answers to the query. "
        "Also make sure your response is in proper slack markdown.\n"
        f"Original query: {query}\n"
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=900,
            temperature=0.6,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error clarifying query for response: {e}")
        return f"Understanding query: {query}" # Fallback

async def generate_keywords(clarified_query: str) -> List[str]:
    """Generates search keywords/queries from a clarified query."""
    from services.common_utils import get_current_datetime
    current_date = get_current_datetime()
    prompt = (
        "Today's date is "
        + current_date
        + ". Act as an expert web Browse agent and give 1-3 search queries from the following query as a JSON array of strings, "
        'e.g. ["term1", "term2"]:\n\n'
        f"{clarified_query}"
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=100,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        keywords_text = completion.choices[0].message.content.strip()
        try:
            # Try to parse as JSON array
            keywords = json.loads(keywords_text)
            return [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()][:3]
        except (json.JSONDecodeError, TypeError):
            # Fallback for non-JSON or incorrect format
            parts = re.split(r'[\[\]",\n]+', keywords_text)
            return [kw.strip().strip('"') for kw in parts if kw.strip()][:3]
    except Exception as e:
        logger.error(f"Error generating keywords: {e}")
        return [clarified_query]

async def understanding_query(clarify_response: str) -> str:
    """Generates an intermediate processing message based on the clarified query."""
    prompt = (
        "You are a preceding agent. You understand the query that comes to you and act like you are doing some research on the query. "
        "In the end, conclude the response like you are proceeding to generate the final answer. "
        "Also make sure your response is in proper slack markdown.\n"
        f"Query: {clarify_response}"
    )
    try:
        completion = await rate_limited_groq_call(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=900,
            temperature=0.6,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating query understanding: {e}")
        return f"Processing request based on your query: {clarify_response}"
    
async def generate_subqueries(main_query: str, num_subqueries: int = 3) -> list[str]:
    """Generates distinct subqueries for general research."""
    from services.common_utils import get_current_datetime
    current_dt = get_current_datetime()
    prompt = (
        f"Today's date {current_dt} Provide exactly {num_subqueries} distinct search queries related to the following topic:\n"
        f"'{main_query}'\n"
        "Write each query on a separate line, with no numbering or additional text."
    )
    try:
        chat_completion = await rate_limited_groq_call(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_completion_tokens=150,
            temperature=0.7,
        )
        response = chat_completion.choices[0].message.content.strip()
        raw_queries = [line.strip() for line in response.splitlines() if line.strip()]
        clean_queries = []
        for q in raw_queries:
            q = re.sub(r"^['\"]|['\"]$", "", q).strip()
            if q:
                clean_queries.append(q)
        unique_subqueries = list(dict.fromkeys(clean_queries))[:num_subqueries]
        logger.info(f"Generated subqueries for '{main_query}': {unique_subqueries}")
        return unique_subqueries
    except Exception as e:
        logger.error(f"Subquery generation error for '{main_query}': {e}")
        return []

async def synthesize_result(
    main_query: str, contents: list[str], max_context: int = 4000
) -> str:
    """Synthesizes collected research content into a final answer."""
    trimmed_contents = [c[:1000] for c in contents if c]
    combined_content = " ".join(trimmed_contents)[:max_context]
    prompt = (
        f"Based on the following collected information, provide a concise, accurate, and well-structured answer to the query:\n"
        f"'{main_query}'\n\nInformation:\n{combined_content}"
    )
    try:
        chat_completion = await rate_limited_groq_call(
            client,
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_completion_tokens=500,
            temperature=0.7,
        )
        result = chat_completion.choices[0].message.content.strip()
        logger.info(f"Synthesized result for '{main_query}'.")
        return result
    except Exception as e:
        logger.error(f"Error during synthesis for '{main_query}': {e}")
        return "Error generating the final result."