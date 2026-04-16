# stelle_backend/services/comment_poller.py
"""
Comment Poller — Phase 3 Background Scheduler of Autonomous Engagement.

Runs as a perpetual async loop (started from main.py on startup).
Every POLL_INTERVAL_SECONDS it:
  1. Finds all users with auto_reply_enabled = True
  2. For each user, finds posts within the reply_window_hours
  3. Fetches comments via platform APIs
  4. Filters out already-replied and spam
  5. Passes new comments to engagement_service for reply generation

Follows the same pattern as post_scheduler.py but uses async/await
and asyncio.gather() for multi-tenant concurrency.
"""

import asyncio
import re as re_module
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import httpx

from database import db
from config import logger
from services.social_engagement_api import fetch_comments, get_user_auth, check_token_health
from services.engagement_service import (
    generate_reply_for_comment,
)
from services.post_content_analyzer import analyze_post_content


# ── Configuration ────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 300  # 5 minutes
MAX_CONCURRENT_USERS = 20    # Limit how many users are processed in parallel

# ── Collections ──────────────────────────────────────────────
engagement_settings_col = db["engagement_settings"]
scheduled_posts_col = db["scheduledposts"]
auth_col = db["oauthcredentials"]
replies_log_col = db["comment_replies_log"]


def _caption_similarity(a: str, b: str) -> float:
    """Token Jaccard similarity for lightweight caption matching."""
    a_tokens = set((a or "").lower().split())
    b_tokens = set((b or "").lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _parse_graph_datetime(value: str) -> datetime:
    """Parse graph API timestamp into UTC datetime; fallback to now on parse errors."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _to_utc_datetime(value: Any) -> datetime:
    """Normalize datetime values from DB/API to timezone-aware UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        return _parse_graph_datetime(value)
    return datetime.now(timezone.utc)


async def _fetch_threads_posts_for_linking(user_id: str) -> List[Dict[str, Any]]:
    """Fetch recent Threads posts for ID backfilling of scheduledposts rows."""
    auth = await get_user_auth(user_id, "threads")
    if not auth:
        return []

    access_token = (auth.get("accessToken", "") or "").strip()
    account_id = (auth.get("accountId", "") or "").strip()
    if not access_token or not account_id:
        return []
    if not access_token.startswith("THAA"):
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await _fetch_all_threads_posts(
                client=client,
                account_id=account_id,
                access_token=access_token,
                max_items=200,
            )
    except Exception as e:
        logger.error(f"❌ Failed to fetch Threads posts for linking user {user_id}: {e}")
        return []


async def _fetch_all_threads_posts(
    client: httpx.AsyncClient,
    account_id: str,
    access_token: str,
    max_items: int = 200,
) -> List[Dict[str, Any]]:
    """
    Fetch Threads posts with pagination.
    Threads API may return only a small first page; this pulls additional pages.
    """
    url = f"https://graph.threads.net/v1.0/{account_id}/threads"
    params = {
        "fields": "id,text,timestamp,permalink",
        "limit": 100,
        "access_token": access_token,
    }

    posts: List[Dict[str, Any]] = []
    next_url: str | None = None

    while len(posts) < max_items:
        if next_url:
            resp = await client.get(next_url)
        else:
            resp = await client.get(url, params=params)

        resp.raise_for_status()
        data = resp.json() or {}
        page_posts = data.get("data", []) or []
        if not page_posts:
            break

        posts.extend(page_posts)
        next_url = (data.get("paging", {}) or {}).get("next")
        if not next_url:
            break

    return posts[:max_items]


async def _auto_link_missing_threads_platform_ids(user_id: str, posts: List[Dict[str, Any]]) -> int:
    """
    Best-effort auto-link for Threads posts missing platformPostId.
    Matches by caption similarity + publish-time proximity.
    """
    missing_posts = [
        p for p in posts
        if p.get("platform", "").lower() == "threads"
        and not _is_valid_platform_id(p.get("platformPostId", ""), "threads")
        and not str(p.get("_id", "")).startswith("threads:")
    ]
    if not missing_posts:
        return 0

    threads_posts = await _fetch_threads_posts_for_linking(user_id)
    if not threads_posts:
        return 0

    used_ids: set[str] = set()
    linked_count = 0
    now = datetime.now(timezone.utc)

    for post in missing_posts:
        post_caption = str(post.get("caption", "") or "")
        post_time = _to_utc_datetime(post.get("updatedAt") or post.get("scheduledAt"))

        best_score = 0.0
        best_item = None

        for item in threads_posts:
            thread_id = str(item.get("id", "") or "").strip()
            if not thread_id or thread_id in used_ids:
                continue

            caption_score = _caption_similarity(post_caption, str(item.get("text", "") or ""))
            item_time = _to_utc_datetime(item.get("timestamp"))
            post_epoch = post_time.timestamp() if isinstance(post_time, datetime) else datetime.now(timezone.utc).timestamp()
            item_epoch = item_time.timestamp() if isinstance(item_time, datetime) else datetime.now(timezone.utc).timestamp()
            hours_delta = abs(post_epoch - item_epoch) / 3600.0
            time_score = max(0.0, 1.0 - min(hours_delta / 48.0, 1.0))
            score = (0.7 * caption_score) + (0.3 * time_score)

            if score > best_score:
                best_score = score
                best_item = item

        if not best_item or best_score < 0.2:
            continue

        thread_id = str(best_item.get("id", "") or "").strip()
        if not thread_id:
            continue

        await scheduled_posts_col.update_one(
            {"_id": post["_id"], "userId": user_id},
            {"$set": {"platformPostId": thread_id, "updatedAt": now}},
        )
        post["platformPostId"] = thread_id
        used_ids.add(thread_id)
        linked_count += 1

        logger.info(
            f"🔗 Auto-linked Threads platformPostId for post {post['_id']} -> {thread_id} "
            f"(score={best_score:.2f})"
        )

    return linked_count


async def _fetch_recent_threads_posts_for_poller(
    user_id: str,
    window_start: datetime,
    existing_platform_ids: set[str],
) -> List[Dict[str, Any]]:
    """
    Fetch recent Threads posts directly from Threads Graph API so autonomous
    engagement can run even when scheduledposts does not contain Threads entries.
    """
    auth = await get_user_auth(user_id, "threads")
    if not auth:
        return []

    access_token = (auth.get("accessToken", "") or "").strip()
    account_id = (auth.get("accountId", "") or "").strip()
    if not access_token or not account_id:
        return []
    if not access_token.startswith("THAA"):
        logger.warning(f"⚠️ Threads token for user {user_id} is invalid (must start with THAA)")
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            all_threads_posts = await _fetch_all_threads_posts(
                client=client,
                account_id=account_id,
                access_token=access_token,
                max_items=200,
            )

        synthetic_posts: List[Dict[str, Any]] = []
        skipped_existing = 0
        skipped_old = 0
        skipped_invalid_id = 0
        for item in all_threads_posts:
            thread_post_id = str(item.get("id", "") or "").strip()
            if not thread_post_id:
                skipped_invalid_id += 1
                continue
            if thread_post_id in existing_platform_ids:
                skipped_existing += 1
                continue

            ts = _parse_graph_datetime(str(item.get("timestamp", "") or ""))
            if ts < window_start:
                skipped_old += 1
                continue

            synthetic_posts.append({
                "_id": f"threads:{thread_post_id}",
                "userId": user_id,
                "status": "posted",
                "platform": "threads",
                "platformPostId": thread_post_id,
                "caption": item.get("text", "") or "",
                "mediaType": "TEXT",
                "mediaUrls": [],
                "scheduledAt": ts,
                "updatedAt": ts,
            })

        logger.info(
            f"🧵 Threads scan for user {user_id}: fetched={len(all_threads_posts)}, "
            f"added={len(synthetic_posts)}, skipped_existing={skipped_existing}, "
            f"skipped_old={skipped_old}, skipped_invalid_id={skipped_invalid_id}"
        )

        return synthetic_posts
    except Exception as e:
        logger.error(f"❌ Failed to fetch recent Threads posts for user {user_id}: {e}")
        return []


async def _fetch_instagram_media_for_linking(user_id: str) -> List[Dict[str, Any]]:
    """
    Fetch the user's recent Instagram media for auto-linking posts that
    are missing platformPostId.
    """
    auth = await get_user_auth(user_id, "instafb")
    if not auth:
        auth = await get_user_auth(user_id, "instagram")
    if not auth:
        return []

    access_token = auth.get("accessToken", "")
    ig_user_id = auth.get("accountId", "")
    if not access_token or not ig_user_id:
        return []

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Try Business Graph path first (accountId may be a Page ID)
        try:
            page_resp = await client.get(
                f"https://graph.facebook.com/v24.0/{ig_user_id}",
                params={"fields": "instagram_business_account", "access_token": access_token},
            )
            if page_resp.is_success:
                biz_id = page_resp.json().get("instagram_business_account", {}).get("id", "")
                if biz_id:
                    media_resp = await client.get(
                        f"https://graph.facebook.com/v24.0/{biz_id}/media",
                        params={
                            "fields": "id,caption,timestamp,media_type,permalink",
                            "limit": 100,
                            "access_token": access_token,
                        },
                    )
                    if media_resp.is_success:
                        return media_resp.json().get("data", [])
        except Exception:
            pass

        # Fallback for Basic Display tokens
        try:
            media_resp = await client.get(
                "https://graph.instagram.com/me/media",
                params={
                    "fields": "id,caption,timestamp,media_type,permalink",
                    "limit": 100,
                    "access_token": access_token,
                },
            )
            if media_resp.is_success:
                return media_resp.json().get("data", [])
        except Exception:
            pass

    return []


async def _auto_link_missing_instagram_platform_ids(user_id: str, posts: List[Dict[str, Any]]) -> int:
    """
    Best-effort auto-link for Instagram posts that are already in MongoDB but
    still missing platformPostId. Returns number of posts linked.
    """
    missing_posts = [
        p for p in posts
        if p.get("platform", "").lower() == "instagram"
        and not _is_valid_platform_id(p.get("platformPostId", ""), "instagram")
    ]
    if not missing_posts:
        return 0

    ig_media = await _fetch_instagram_media_for_linking(user_id)
    if not ig_media:
        return 0

    used_media_ids = set()
    linked_count = 0
    now = datetime.now(timezone.utc)
    min_match_score = 0.4

    for post in missing_posts:
        post_caption = post.get("caption", "")
        best_score = 0.0
        best_item = None

        for item in ig_media:
            media_id = item.get("id", "")
            if not media_id or media_id in used_media_ids:
                continue
            score = _caption_similarity(post_caption, item.get("caption", ""))
            if score > best_score:
                best_score = score
                best_item = item

        if best_item and best_score >= min_match_score:
            media_id = best_item.get("id", "")
            if not media_id:
                continue

            await scheduled_posts_col.update_one(
                {"_id": post["_id"], "userId": user_id},
                {"$set": {"platformPostId": media_id, "updatedAt": now}},
            )
            post["platformPostId"] = media_id
            used_media_ids.add(media_id)
            linked_count += 1

            logger.info(
                f"🔗 Auto-linked platformPostId for post {post['_id']} -> {media_id} "
                f"(score={best_score:.2f})"
            )

    return linked_count


# ══════════════════════════════════════════════════════════════
# PLATFORM ID VALIDATION
# ══════════════════════════════════════════════════════════════

def _is_valid_platform_id(platform_id: str, platform: str) -> bool:
    """
    Validate that a platform post ID is a real platform-issued ID,
    not a MongoDB ObjectId (24-char hex string that falls back when
    platformPostId was never saved).

    MongoDB ObjectIds are exactly 24 hexadecimal chars.
    Platform IDs:
      Instagram  → purely numeric strings (e.g. '17846368219941196')
      YouTube    → 11-char alphanumeric (e.g. 'dQw4w9WgXcQ')
      Facebook   → numeric or 'page_post' style
      TikTok     → large numeric string
      LinkedIn   → URN-style (e.g. 'urn:li:activity:...')
      Threads    → numeric
    """
    if not platform_id:
        return False

    # Detect a MongoDB ObjectId: exactly 24 hex characters
    import re as _re
    if _re.fullmatch(r'[0-9a-fA-F]{24}', platform_id):
        return False  # This is a MongoDB _id, not a real platform ID

    # Platform-specific checks
    platform_lower = platform.lower()
    if platform_lower in ("instagram", "threads", "tiktok", "facebook"):
        # These platforms use purely numeric IDs
        if not platform_id.isdigit():
            return False
    elif platform_lower == "youtube":
        # YouTube IDs are 11 alphanumeric chars (plus - and _)
        if not _re.fullmatch(r'[A-Za-z0-9_\-]{11}', platform_id):
            return False

    return True


# ══════════════════════════════════════════════════════════════
# MAIN POLLER LOOP
# ══════════════════════════════════════════════════════════════

async def comment_poller_loop():
    """
    Main entry point — runs forever as a background async task.
    Start this from main.py on_event("startup").
    """
    logger.info("🚀 Autonomous Engagement — Comment Poller started")

    while True:
        try:
            await _run_engagement_cycle()
        except Exception as e:
            logger.error(f"🔥 Error in engagement poller cycle: {e}", exc_info=True)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _run_engagement_cycle():
    """
    One polling cycle:
    1. Fetch all users with engagement enabled
    2. Process all users concurrently (bounded by MAX_CONCURRENT_USERS)
    """
    # Find all users with auto-reply enabled
    cursor = engagement_settings_col.find({
        "auto_reply_enabled": True,
    })

    active_users = []
    async for settings in cursor:
        active_users.append(settings)

    if not active_users:
        return  # No active users, nothing to do

    logger.info(f"⏰ Engagement cycle: {len(active_users)} active user(s)")

    # Process users concurrently with a semaphore to limit parallelism
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)

    async def process_with_limit(settings):
        async with semaphore:
            await _process_user(settings)

    tasks = [process_with_limit(s) for s in active_users]
    await asyncio.gather(*tasks, return_exceptions=True)


async def _process_user(settings: Dict[str, Any]):
    """
    Process all eligible posts and comments for a single user.
    Completely isolated — one user's error doesn't affect others.
    """
    user_id = settings.get("user_id", "")
    reply_window_hours = settings.get("reply_window_hours", 48)
    enabled_platforms = [
        str(p).strip().lower()
        for p in settings.get("platforms", [])
        if isinstance(p, str) and str(p).strip()
    ]
    reply_mode = settings.get("reply_mode", "automatic")

    try:
        # ── Find posts within the reply window ──
        window_start = datetime.now(timezone.utc) - timedelta(hours=reply_window_hours)

        # Query for posts that:
        # 1. Belong to this user
        # 2. Were posted (status = "posted") within the reply window
        # 3. Are on enabled platforms
        # Use updatedAt first because posted jobs may have older scheduledAt values.
        post_query = {
            "userId": user_id,
            "status": "posted",
            "$or": [
                {"updatedAt": {"$gte": window_start}},
                {"scheduledAt": {"$gte": window_start}},
            ],
        }
        if enabled_platforms:
            # Case-insensitive match on platforms
            platform_regex = "|".join(re_module.escape(p) for p in enabled_platforms)
            post_query["platform"] = {"$regex": platform_regex, "$options": "i"}

        posts_cursor = scheduled_posts_col.find(post_query)
        posts = []
        async for post in posts_cursor:
            posts.append(post)

        auto_linked_threads = await _auto_link_missing_threads_platform_ids(user_id, posts)
        if auto_linked_threads:
            logger.info(f"🔁 Auto-linked {auto_linked_threads} Threads post(s) for user {user_id}")

        existing_platform_ids = {
            str(p.get("platformPostId", "") or "").strip()
            for p in posts
            if str(p.get("platformPostId", "") or "").strip()
        }

        if "threads" in enabled_platforms:
            threads_posts = await _fetch_recent_threads_posts_for_poller(
                user_id=user_id,
                window_start=window_start,
                existing_platform_ids=existing_platform_ids,
            )
            if threads_posts:
                posts.extend(threads_posts)
                logger.info(
                    f"🧵 Added {len(threads_posts)} recent Threads post(s) from Threads API for user {user_id}"
                )

        # Best-effort automatic backfill for missing Instagram platform IDs.
        auto_linked = await _auto_link_missing_instagram_platform_ids(user_id, posts)
        if auto_linked:
            logger.info(f"🔁 Auto-linked {auto_linked} Instagram post(s) for user {user_id}")

        if not posts:
            return

        logger.info(f"👤 User {user_id}: found {len(posts)} active post(s)")

        for post in posts:
            post_id = str(post.get("_id", ""))
            platform = post.get("platform", "").lower()
            caption = post.get("caption", "")
            media_type = post.get("mediaType", "IMAGE")
            media_urls = post.get("mediaUrls", [])

            # ── Validate platformPostId ──────────────────────────────
            # Must be the real platform-issued ID, not the MongoDB _id.
            # Posts published before platformPostId was stored must be
            # skipped to avoid sending invalid IDs to platform APIs.
            platform_post_id = post.get("platformPostId", "")
            if not _is_valid_platform_id(platform_post_id, platform):
                logger.warning(
                    f"⚠️ Post {post_id} has no valid platformPostId "
                    f"(got: '{platform_post_id or 'None'}') — skipping comment fetch. "
                    f"Update the post with the real {platform} media ID."
                )
                continue

            # ── Ensure post is analyzed (Phase 2 — lazy analysis) ──
            # Synthetic Threads posts may not exist in scheduledposts with ObjectId.
            # Analysis is best-effort and should never block reply generation.
            try:
                await analyze_post_content(
                    user_id=user_id,
                    post_id=post_id,
                    platform=platform,
                    media_type=media_type,
                    caption=caption,
                    media_urls=media_urls,
                    video_hash=post.get("videoHash"),
                )
            except Exception as e:
                logger.warning(f"⚠️ Post analysis skipped for {post_id} ({platform}): {e}")

            # ── Get OAuth token ──
            # For Instagram, comment operations require a Business Graph API
            # token (EAA...) stored under "instafb", not the Basic Display
            # token (IGAB...) stored under "instagram".
            auth = None
            if platform == "instagram":
                auth = await get_user_auth(user_id, "instafb")
                if auth:
                    logger.debug(f"Using instafb (Business Graph) token for user {user_id}")
            if not auth:
                auth = await get_user_auth(user_id, platform)
            if not auth:
                logger.warning(f"⚠️ No OAuth for user {user_id} platform {platform}")
                continue

            # ── Check token health (expiry + type) ──
            if platform == "instagram":
                health = check_token_health(auth, platform)
                if health["is_expired"]:
                    logger.error(
                        f"🔴 Instagram token EXPIRED for user {user_id} "
                        f"(expired: {health['expires_at']}). Fix: {health['fix']}"
                    )
                    break  # Same token used for all posts — bail out of post loop
                if health["token_type"] == "basic_display":
                    logger.error(
                        f"🔴 Instagram Basic Display token for user {user_id} "
                        f"cannot fetch comments. Fix: {health['fix']}"
                    )
                    break

            access_token = auth.get("accessToken", "")
            account_id = auth.get("accountId", "")

            # ── Fetch comments from platform ──
            comments = await fetch_comments(
                platform=platform,
                post_id=platform_post_id,
                access_token=access_token,
                limit=100,
            )

            if not comments:
                continue

            # ── Filter already-replied comments + own account comments ──
            own_usernames = [account_id] if account_id else []
            new_comments = await _filter_new_comments(user_id, post_id, comments, own_usernames)

            if not new_comments:
                continue

            logger.info(f"💬 Post {post_id}: {len(new_comments)} new comment(s) to process")

            # ── Process each new comment ──
            for comment in new_comments:
                result = await generate_reply_for_comment(
                    user_id=user_id,
                    post_id=post_id,
                    platform=platform,
                    comment=comment,
                    access_token=access_token,
                    user_name=user_id,  # Could be fetched from users collection
                )

                status = result.get("status", "")
                logger.debug(f"   Comment {comment.get('comment_id', '')}: {status}")

    except Exception as e:
        logger.error(f"❌ Error processing user {user_id}: {e}", exc_info=True)


async def _filter_new_comments(
    user_id: str,
    post_id: str,
    comments: List[Dict[str, Any]],
    own_usernames: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter out comments that have already been processed (replied, queued, etc.)
    and the user's own comments.

        Excludes comments that already have a reply lifecycle entry for this post,
        including user-rejected replies, so they are not regenerated in later cycles.
        Excluded statuses:
            posted, approved_posted, pending_review, draft, rejected
        Non-excluded statuses (for retry):
            spam_skipped, failed
    """
    if not comments:
        return []

    # ── Filter out the account owner's own comments ──
    own_set = {u.lower() for u in (own_usernames or []) if u}
    if own_set:
        comments = [
            c for c in comments
            if c.get("author", "").lower() not in own_set
        ]

    if not comments:
        return []

    comment_ids = [c.get("comment_id", "") for c in comments if c.get("comment_id")]

    # Exclude comments that are already handled or explicitly rejected by user.
    existing_cursor = replies_log_col.find(
        {
            "user_id": user_id,
            "post_id": post_id,
            "comment_id": {"$in": comment_ids},
            "status": {"$in": ["posted", "approved_posted", "pending_review", "draft", "rejected"]},
        },
        {"comment_id": 1}
    )
    existing_ids = set()
    async for doc in existing_cursor:
        existing_ids.add(doc.get("comment_id", ""))

    # Filter
    new = [
        c for c in comments
        if c.get("comment_id") and c["comment_id"] not in existing_ids
    ]

    return new


async def _process_user_with_stats(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process all eligible posts/comments for a single user and return detailed stats.
    Used by the manual trigger endpoint.
    """
    user_id = settings.get("user_id", "")
    reply_window_hours = settings.get("reply_window_hours", 48)
    enabled_platforms = [
        str(p).strip().lower()
        for p in settings.get("platforms", [])
        if isinstance(p, str) and str(p).strip()
    ]

    total_posts = 0
    total_comments = 0
    total_replies = 0
    total_already_replied = 0
    total_skipped_no_id = 0

    try:
        # ── Find posts within the reply window ──
        # updatedAt is set to the actual publish time when status becomes "posted"
        window_start = datetime.now(timezone.utc) - timedelta(hours=reply_window_hours)

        post_query = {
            "userId": user_id,
            "status": "posted",
            "$or": [
                {"updatedAt": {"$gte": window_start}},
                {"scheduledAt": {"$gte": window_start}},
            ],
        }
        if enabled_platforms:
            platform_regex = "|".join(re_module.escape(p) for p in enabled_platforms)
            post_query["platform"] = {"$regex": platform_regex, "$options": "i"}

        posts_cursor = scheduled_posts_col.find(post_query)
        posts = []
        async for post in posts_cursor:
            posts.append(post)

        auto_linked_threads = await _auto_link_missing_threads_platform_ids(user_id, posts)
        if auto_linked_threads:
            logger.info(f"🔁 Auto-linked {auto_linked_threads} Threads post(s) for user {user_id}")

        existing_platform_ids = {
            str(p.get("platformPostId", "") or "").strip()
            for p in posts
            if str(p.get("platformPostId", "") or "").strip()
        }

        if "threads" in enabled_platforms:
            threads_posts = await _fetch_recent_threads_posts_for_poller(
                user_id=user_id,
                window_start=window_start,
                existing_platform_ids=existing_platform_ids,
            )
            if threads_posts:
                posts.extend(threads_posts)
                logger.info(
                    f"🧵 Added {len(threads_posts)} recent Threads post(s) from Threads API for user {user_id}"
                )

        total_posts = len(posts)

        if not posts:
            return {
                "status": "success",
                "posts_processed": 0,
                "comments_processed": 0,
                "replies_generated": 0,
                "already_replied": 0,
                "message": "No active posts found within the reply window",
            }

        for post in posts:
            post_id = str(post.get("_id", ""))
            platform = post.get("platform", "").lower()
            caption = post.get("caption", "")
            media_type = post.get("mediaType", "IMAGE")
            media_urls = post.get("mediaUrls", [])

            # ── Validate platformPostId ──────────────────────────────────────
            platform_post_id = post.get("platformPostId", "")
            if not _is_valid_platform_id(platform_post_id, platform):
                logger.warning(
                    f"⚠️ Post {post_id} has no valid platformPostId "
                    f"(got: '{platform_post_id or 'None'}') — skipping comment fetch. "
                    f"Update the post with the real {platform} media ID."
                )
                total_skipped_no_id += 1
                continue

            # ── Ensure post is analyzed ──
            try:
                await analyze_post_content(
                    user_id=user_id,
                    post_id=post_id,
                    platform=platform,
                    media_type=media_type,
                    caption=caption,
                    media_urls=media_urls,
                    video_hash=post.get("videoHash"),
                )
            except Exception as e:
                logger.warning(f"⚠️ Post analysis skipped for {post_id} ({platform}): {e}")

            # ── Get OAuth token ──
            # For Instagram, comment operations require a Business Graph API
            # token (EAA...) stored under "instafb", not the Basic Display
            # token (IGAB...) stored under "instagram".
            auth = None
            if platform == "instagram":
                auth = await get_user_auth(user_id, "instafb")
                if auth:
                    logger.debug(f"Using instafb (Business Graph) token for user {user_id}")
            if not auth:
                auth = await get_user_auth(user_id, platform)
            if not auth:
                logger.warning(f"⚠️ No OAuth for user {user_id} platform {platform}")
                continue

            # ── Check token health (expiry + type) ──
            if platform == "instagram":
                health = check_token_health(auth, platform)
                if health["is_expired"]:
                    logger.error(
                        f"🔴 Instagram token EXPIRED for user {user_id} "
                        f"(expired: {health['expires_at']}). Fix: {health['fix']}"
                    )
                    break  # Same token used for all posts — bail out of post loop
                if health["token_type"] == "basic_display":
                    logger.error(
                        f"🔴 Instagram Basic Display token for user {user_id} "
                        f"cannot fetch comments. Fix: {health['fix']}"
                    )
                    break

            access_token = auth.get("accessToken", "")

            # ── Fetch comments from platform ──
            comments = await fetch_comments(
                platform=platform,
                post_id=platform_post_id,
                access_token=access_token,
                limit=100,
            )

            if not comments:
                continue

            total_comments += len(comments)

            # ── Filter already-replied comments + own account comments ──
            own_usernames = [auth.get("accountId", "")] if auth.get("accountId") else []
            new_comments = await _filter_new_comments(user_id, post_id, comments, own_usernames)
            total_already_replied += len(comments) - len(new_comments)

            if not new_comments:
                continue

            # ── Process each new comment ──
            for comment in new_comments:
                result = await generate_reply_for_comment(
                    user_id=user_id,
                    post_id=post_id,
                    platform=platform,
                    comment=comment,
                    access_token=access_token,
                    user_name=user_id,
                )

                status = result.get("status", "")
                if status in ("posted", "pending_review", "draft"):
                    total_replies += 1

        return {
            "status": "success",
            "posts_processed": total_posts,
            "comments_processed": total_comments,
            "replies_generated": total_replies,
            "already_replied": total_already_replied,
            "skipped_no_platform_id": total_skipped_no_id,
            "message": (
                f"Cycle completed: {total_replies} replies generated from "
                f"{total_posts - total_skipped_no_id} posts "
                f"({total_skipped_no_id} skipped — missing platformPostId)"
                if total_skipped_no_id else
                f"Cycle completed: {total_replies} replies generated from {total_posts} posts"
            ),
        }

    except Exception as e:
        logger.error(f"❌ Error in trigger for user {user_id}: {e}", exc_info=True)
        return {
            "status": "error",
            "posts_processed": total_posts,
            "comments_processed": total_comments,
            "replies_generated": total_replies,
            "already_replied": total_already_replied,
            "skipped_no_platform_id": total_skipped_no_id,
            "message": str(e),
        }


# ══════════════════════════════════════════════════════════════
# MANUAL TRIGGER (for testing / on-demand)
# ══════════════════════════════════════════════════════════════

async def trigger_engagement_for_user(user_id: str) -> Dict[str, Any]:
    """
    Manually trigger one engagement cycle for a specific user.
    Useful for testing or on-demand processing from the API.
    Returns detailed results matching the frontend TriggerResult interface:
      status, posts_processed, comments_processed, replies_generated, already_replied, message
    """
    settings = await engagement_settings_col.find_one({"user_id": user_id})
    if not settings:
        return {
            "status": "error",
            "posts_processed": 0,
            "comments_processed": 0,
            "replies_generated": 0,
            "already_replied": 0,
            "message": "No engagement settings found. Please configure engagement first.",
        }

    try:
        result = await _process_user_with_stats(settings)
        return result
    except Exception as e:
        return {
            "status": "error",
            "posts_processed": 0,
            "comments_processed": 0,
            "replies_generated": 0,
            "already_replied": 0,
            "message": str(e),
        }
