import asyncio
from typing import Any, Dict, Optional

from services import video_cache_repo as repo


class _FakeCacheCollection:
    def __init__(self, doc: Optional[Dict[str, Any]]):
        self._doc = doc

    async def find_one(self, query: Dict[str, Any]):
        expected_hash = self._doc.get("video_hash") if self._doc else None
        if expected_hash and query.get("video_hash") == expected_hash:
            return self._doc
        return None


class _FakeDB:
    def __init__(self, doc: Optional[Dict[str, Any]]):
        self._collection = _FakeCacheCollection(doc)

    def __getitem__(self, name: str):
        if name != "video_analysis_cache":
            raise KeyError(name)
        return self._collection


async def test_cache_result_hides_identifier_fields() -> None:
    test_doc = {
        "video_hash": "abc123",
        "transcript": "hello world",
        "visual_summary": "person speaking",
        "visual_captions": [["f1", "a speaker on stage"]],
        "detected_texts": ["hello"],
        "ocr_text_combined": "hello",
        "detected_person": "Public Figure",
        "marketing_prompt": "prompt",
        "objects": ["microphone"],
        "actions": ["talking"],
        "user_id": "user_private_001",
        "scheduled_post_id": "67cbf8c2197f2fc3fcb9a001",
    }

    original_db = repo.db
    repo.db = _FakeDB(test_doc)
    try:
        result = await repo.get_cached_video_analysis("abc123")
    finally:
        repo.db = original_db

    if result is None:
        raise AssertionError("Expected cached result, got None")

    forbidden_keys = {"user_id", "scheduled_post_id", "session_id", "personal_id"}
    leaked = sorted(forbidden_keys.intersection(set(result.keys())))
    if leaked:
        raise AssertionError(f"Privacy leak detected in cached output: {leaked}")

    print("PASS: Cache result does not expose identifier fields.")


async def test_cache_miss_returns_none() -> None:
    original_db = repo.db
    repo.db = _FakeDB(None)
    try:
        result = await repo.get_cached_video_analysis("missing_hash")
    finally:
        repo.db = original_db

    if result is not None:
        raise AssertionError("Expected None for cache miss")

    print("PASS: Cache miss returns None.")


async def main() -> None:
    await test_cache_result_hides_identifier_fields()
    await test_cache_miss_returns_none()
    print("All video cache privacy checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
