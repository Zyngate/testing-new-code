import asyncio
import sys
from typing import Optional

from bson import ObjectId

from database import db


def _parse_target_id() -> Optional[ObjectId]:
    if len(sys.argv) < 2:
        return None
    raw = (sys.argv[1] or "").strip()
    if not raw:
        return None
    if not ObjectId.is_valid(raw):
        raise ValueError(f"Invalid ObjectId: {raw}")
    return ObjectId(raw)


async def _pick_latest_scheduled_post_id() -> Optional[ObjectId]:
    post = await db["scheduledposts"].find_one(sort=[("createdAt", -1)])
    if not post:
        return None
    post_id = post.get("_id")
    return post_id if isinstance(post_id, ObjectId) else None


async def main() -> None:
    target_id = _parse_target_id()
    if target_id is None:
        target_id = await _pick_latest_scheduled_post_id()
        if target_id is None:
            print("No scheduledposts documents found.")
            return
        print(f"Using latest scheduled post _id: {target_id}")
    else:
        print(f"Using provided scheduled post _id: {target_id}")

    scheduled_post = await db["scheduledposts"].find_one({"_id": target_id})
    if not scheduled_post:
        print(f"No document found in scheduledposts for _id={target_id}")
        return

    status = str(scheduled_post.get("status", "")).lower()
    media_type = str(scheduled_post.get("mediaType", "")).lower()
    platform_post_id = str(scheduled_post.get("platformPostId", "") or "").strip()
    video_hash = str(scheduled_post.get("videoHash", "") or "").strip()

    print("Scheduled post snapshot:")
    print(f"- _id: {target_id}")
    print(f"- status: {status or 'N/A'}")
    print(f"- mediaType: {media_type or 'N/A'}")
    print(f"- platformPostId present: {'yes' if platform_post_id else 'no'}")
    print(f"- videoHash present: {'yes' if video_hash else 'no'}")

    expected_to_have_video_cache = (
        status == "posted" and media_type in {"video", "reel", "short"}
    )
    if not expected_to_have_video_cache:
        print(
            "Note: This post may not produce video_analysis_cache yet "
            "(expected for non-posted or non-video posts)."
        )

    cache_doc = await db["video_analysis_cache"].find_one({"scheduled_post_id": target_id})

    if not cache_doc:
        print("No matching video_analysis_cache document found for this scheduled post _id.")

        if video_hash:
            hash_doc = await db["video_analysis_cache"].find_one({"video_hash": video_hash})
            if hash_doc:
                print("Found cache by video_hash, but scheduled_post_id is missing or different:")
                print(f"- cache._id: {hash_doc.get('_id')}")
                print(f"- cache.scheduled_post_id: {hash_doc.get('scheduled_post_id')}")
                print(f"- cache.video_hash: {hash_doc.get('video_hash')}")
        return

    cache_id = cache_doc.get("_id")
    cache_video_hash = cache_doc.get("video_hash")
    cache_post_id = cache_doc.get("scheduled_post_id")

    print("MATCH FOUND")
    print(f"scheduledposts._id: {target_id}")
    print(f"video_analysis_cache._id: {cache_id}")
    print(f"video_analysis_cache.scheduled_post_id: {cache_post_id}")
    print(f"video_analysis_cache.video_hash: {cache_video_hash}")


if __name__ == "__main__":
    asyncio.run(main())
