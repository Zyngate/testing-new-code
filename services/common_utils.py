# stelle_backend/services/common_utils.py
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any

from config import logger

def get_current_datetime() -> str:
    """Returns the current formatted datetime string."""
    return datetime.now().strftime("%B %d, %Y, %I:%M %p")

def sanitize_chat_history(messages):
    """
    Remove non-standard messages before sending to LLM
    """
    clean = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        # must have role + content only
        if "role" not in msg or "content" not in msg:
            continue

        # ❌ exclude deepsearch / visualize / system-only artifacts
        if msg.get("type") in ("deepsearch", "visualize", "thinking"):
            continue

        clean.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return clean


def filter_think_messages(messages: list) -> list:
    """Removes internal <think>...</think> blocks from message content."""
    filtered = []
    for msg in messages:
        content = msg.get("content") or ""
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if cleaned:
            new_msg = msg.copy()
            new_msg["content"] = cleaned
            # Remove embedding from history for lighter payload, unless explicitly needed later
            new_msg.pop("embedding", None) 
            filtered.append(new_msg)
    return filtered

def convert_object_ids(document: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively converts MongoDB ObjectId to string for JSON serialization."""
    for key, value in document.items():
        if key == "_id":
            document[key] = str(value)
        elif isinstance(value, dict):
            document[key] = convert_object_ids(value)
        elif isinstance(value, list):
            document[key] = [
                convert_object_ids(item) if isinstance(item, dict) else item
                for item in value
            ]
    return document

# Temporary storage for WebSocket research IDs (not persistent)
websocket_queries: Dict[str, str] = {}
deepsearch_queries: Dict[str, Dict[str, Any]] = {}