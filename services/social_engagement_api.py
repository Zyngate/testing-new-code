# stelle_backend/services/social_engagement_api.py
"""
Social Engagement API — Platform-specific wrappers for fetching comments
and posting replies.

Supports: Instagram, YouTube, Facebook, TikTok, LinkedIn, Threads.

Each platform has:
  - fetch_comments(post_id, access_token, ...) → List[Comment]
  - post_reply(comment_id, reply_text, access_token, ...) → bool

Uses the OAuth tokens stored in the oauthcredentials collection.
"""

import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from database import db
from config import logger


# ── OAuth collection ─────────────────────────────────────────
auth_col = db["oauthcredentials"]

# ── HTTP client timeout ──────────────────────────────────────
HTTP_TIMEOUT = 30.0


# ══════════════════════════════════════════════════════════════
# COMMENT DATA MODEL (standardized across platforms)
# ══════════════════════════════════════════════════════════════

def _normalize_comment(
    platform: str,
    comment_id: str,
    author: str,
    text: str,
    timestamp: str = "",
    parent_id: str = "",
    like_count: int = 0,
    replies: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Standardize comment format across platforms."""
    result = {
        "platform": platform,
        "comment_id": comment_id,
        "author": author,
        "text": text,
        "timestamp": timestamp,
        "parent_id": parent_id,  # Empty for top-level comments
        "like_count": like_count,
    }
    if replies is not None:
        result["replies"] = replies
    return result


# ══════════════════════════════════════════════════════════════
# OAUTH HELPER
# ══════════════════════════════════════════════════════════════

async def get_user_auth(user_id: str, platform: str) -> Optional[Dict[str, Any]]:
    """
    Fetch OAuth credentials for a user/platform pair.
    Matches case-insensitively since DB may store 'Instagram' vs 'instagram'.
    """
    import re as re_module
    try:
        doc = await auth_col.find_one({
            "userId": user_id,
            "platform": {"$regex": f"^{re_module.escape(platform)}$", "$options": "i"}
        })
        return doc
    except Exception as e:
        logger.error(f"Error fetching auth for {user_id}/{platform}: {e}")
        return None


def check_token_health(auth_doc: dict, platform: str = "instagram") -> dict:
    """
    Inspect a stored OAuth credential document and return a health report.

    Returns a dict with:
        is_expired     : bool
        expires_at     : str | None  (ISO datetime)
        token_type     : 'basic_display' | 'business_graph' | 'unknown'
        can_manage_comments: bool
        error          : str | None  (human-readable problem description)
        fix            : str | None  (how to fix it)
    """
    from datetime import datetime, timezone as tz

    access_token = (auth_doc or {}).get("accessToken", "")
    token_type = _ig_token_type(access_token)

    # ── Expiry check ────────────────────────────────────────────
    expires_raw = (auth_doc or {}).get("tokenExpiresAt")
    expires_at: Optional[datetime] = None
    is_expired = False

    if expires_raw:
        # Motor returns datetime objects; string fallback just in case
        if isinstance(expires_raw, datetime):
            expires_at = expires_raw if expires_raw.tzinfo else expires_raw.replace(tzinfo=tz.utc)
        else:
            try:
                from dateutil import parser as _dp
                expires_at = _dp.parse(str(expires_raw))
            except Exception:
                pass

        if expires_at:
            is_expired = datetime.now(tz.utc) > expires_at

    # ── Build report ────────────────────────────────────────────
    report: dict = {
        "token_type": token_type,
        "is_expired": is_expired,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "can_manage_comments": token_type == "business_graph" and not is_expired,
        "error": None,
        "fix": None,
    }

    if is_expired:
        report["error"] = (
            f"Your Instagram access token expired on "
            f"{expires_at.strftime('%Y-%m-%d') if expires_at else 'an unknown date'}. "
            "All API calls will fail until you reconnect."
        )
        report["fix"] = (
            "Re-authenticate via your Instagram OAuth flow in the app to get a fresh token, "
            "or go to https://developers.facebook.com/tools/explorer, generate a new token "
            "with the required permissions, then call POST /auth/update-token."
        )
    elif token_type == "basic_display":
        report["error"] = (
            "Your Instagram token is a Basic Display API token (IGAB...). "
            "It can list media but CANNOT fetch, reply to, or delete comments."
        )
        report["fix"] = (
            "You need an Instagram Business/Creator account connected to a Facebook Page "
            "and a Business Graph API token (starts with EAA). "
            "Steps: 1) Instagram Settings → Account → Switch to Professional. "
            "2) Connect to a Facebook Page. "
            "3) Get a token at https://developers.facebook.com/tools/explorer with permissions: "
            "instagram_basic, instagram_manage_comments, pages_read_engagement. "
            "4) Save via POST /auth/update-token {userId, platform, accessToken, accountId}."
        )

    return report


def _ig_token_type(access_token: str) -> str:
    """
    Detect which Instagram API family a token belongs to.

    Returns:
        'basic_display'  — token starts with 'IGAB' (Instagram Basic Display API).
                           Only works on graph.instagram.com. Cannot manage comments.
        'business_graph' — token starts with 'EAA' (Meta Facebook/Business token).
                           Works on graph.facebook.com. Required for comments.
        'unknown'        — unrecognised prefix.
    """
    token = (access_token or "").strip()
    if token.startswith("IGAB"):
        return "basic_display"
    if token.startswith("EAA"):
        return "business_graph"
    return "unknown"


_BUSINESS_GRAPH_REQUIRED = (
    "Your Instagram token is a Basic Display API token (starts with 'IGAB'). "
    "Comment management (fetch / reply / delete) requires an Instagram Business "
    "or Creator account connected to a Facebook Page, and a Business Graph API token "
    "(starts with 'EAA'). "
    "Steps to fix: "
    "1) Convert your Instagram account to Business/Creator in Instagram Settings. "
    "2) Connect it to a Facebook Page. "
    "3) In Meta Developer console (graph.facebook.com/v18.0/me/accounts) get a Page "
    "   access token, then exchange it for a long-lived token. "
    "4) Save it via POST /auth/update-token."
)


# ══════════════════════════════════════════════════════════════
# INSTAGRAM (Meta Graph API)
# ══════════════════════════════════════════════════════════════

async def fetch_comments_instagram(
    media_id: str,
    access_token: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch comments on an Instagram media object (flat list, no replies).
    Uses Meta Business Graph API: GET /{media-id}/comments
    Requires a Business/Creator account with an EAA... token.
    """
    token_type = _ig_token_type(access_token)
    if token_type == "basic_display":
        logger.error(
            f"\u274c Instagram Basic Display token cannot fetch comments for {media_id}. "
            "A Business Graph API token (EAA...) is required."
        )
        raise ValueError(_BUSINESS_GRAPH_REQUIRED)

    url = f"https://graph.facebook.com/v24.0/{media_id}/comments"
    params = {
        "fields": "id,text,username,timestamp,like_count",
        "limit": min(limit, 100),
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                error_body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                logger.error(f"❌ Instagram comments API error for {media_id}: HTTP {resp.status_code} — {error_body}")
                return []
            data = resp.json()

        comments = []
        for c in data.get("data", []):
            comments.append(_normalize_comment(
                platform="instagram",
                comment_id=c.get("id", ""),
                author=c.get("username", "unknown"),
                text=c.get("text", ""),
                timestamp=c.get("timestamp", ""),
                like_count=c.get("like_count", 0),
            ))

        logger.info(f"📸 Instagram: fetched {len(comments)} comments for media {media_id}")
        return comments

    except Exception as e:
        logger.error(f"❌ Instagram fetch_comments error for {media_id}: {e}")
        return []


async def fetch_comments_with_replies_instagram(
    media_id: str,
    access_token: str,
    limit: int = 25,
) -> Dict[str, Any]:
    """
    Fetch comments WITH nested replies for an Instagram media object.
    Mirrors the CommentsModal.tsx pattern from publisher analytics:
      1. GET /{media-id}/comments?fields=id,username,text,timestamp,like_count&limit=25
      2. For each comment, GET /{comment-id}/replies?fields=id,username,text,timestamp,like_count
      3. Returns comments with replies[] and pagination cursor.

    Returns:
        {
            "comments": [...],
            "paging": {"next": "..." | null}
        }
    """
    token_type = _ig_token_type(access_token)
    if token_type == "basic_display":
        raise ValueError(_BUSINESS_GRAPH_REQUIRED)

    url = f"https://graph.facebook.com/v24.0/{media_id}/comments"
    params = {
        "fields": "id,username,text,timestamp,like_count",
        "limit": min(limit, 100),
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                error_body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                logger.error(f"❌ Instagram comments API error for {media_id}: HTTP {resp.status_code} — {error_body}")
                return {"comments": [], "paging": {"next": None}, "error": error_body}
            data = resp.json()

        comments_raw = data.get("data", [])
        next_url = data.get("paging", {}).get("next")

        # Fetch replies for each comment concurrently
        comments = await _fetch_replies_for_comments(
            comments_raw, access_token
        )

        logger.info(f"📸 Instagram: fetched {len(comments)} comments with replies for media {media_id}")
        return {
            "comments": comments,
            "paging": {"next": next_url},
        }

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"❌ Instagram fetch_comments_with_replies error for {media_id}: {e}")
        return {"comments": [], "paging": {"next": None}}


async def fetch_more_comments_instagram(
    next_url: str,
    access_token: str,
) -> Dict[str, Any]:
    """
    Load more comments using the pagination URL from a previous response.
    Also fetches replies for each new comment.
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(next_url)
            if resp.status_code != 200:
                error_body = resp.text
                logger.error(f"❌ Instagram pagination error: HTTP {resp.status_code} — {error_body}")
                return {"comments": [], "paging": {"next": None}}
            data = resp.json()

        comments_raw = data.get("data", [])
        next_cursor = data.get("paging", {}).get("next")

        comments = await _fetch_replies_for_comments(
            comments_raw, access_token
        )

        return {
            "comments": comments,
            "paging": {"next": next_cursor},
        }

    except Exception as e:
        logger.error(f"❌ Instagram pagination error: {e}")
        return {"comments": [], "paging": {"next": None}}


async def _fetch_replies_for_comments(
    comments_raw: List[Dict[str, Any]],
    access_token: str,
) -> List[Dict[str, Any]]:
    """
    For each comment, fetch its replies from the Graph API.
    Returns a list of normalized comments with replies[] attached.
    Reuses a single httpx client for all reply fetches (efficiency + rate limiting).
    """
    async def _get_replies(
        http_client: httpx.AsyncClient, comment: Dict[str, Any]
    ) -> Dict[str, Any]:
        comment_id = comment.get("id", "")
        reply_url = f"https://graph.facebook.com/v24.0/{comment_id}/replies"
        reply_params = {
            "fields": "id,username,text,timestamp,like_count",
            "access_token": access_token,
        }
        replies = []
        try:
            resp = await http_client.get(reply_url, params=reply_params)
            if resp.status_code != 200:
                error_body = resp.text
                logger.warning(f"⚠️ Failed to fetch replies for comment {comment_id}: HTTP {resp.status_code} — {error_body}")
            else:
                reply_data = resp.json()
                for r in reply_data.get("data", []):
                    replies.append({
                        "id": r.get("id", ""),
                        "username": r.get("username", "unknown"),
                        "text": r.get("text", ""),
                        "timestamp": r.get("timestamp", ""),
                        "like_count": r.get("like_count", 0),
                    })
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch replies for comment {comment_id}: {e}")

        return _normalize_comment(
            platform="instagram",
            comment_id=comment_id,
            author=comment.get("username", "unknown"),
            text=comment.get("text", ""),
            timestamp=comment.get("timestamp", ""),
            like_count=comment.get("like_count", 0),
            replies=replies,
        )

    # Fetch replies concurrently using a shared HTTP client
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        tasks = [_get_replies(client, c) for c in comments_raw]
        results = await asyncio.gather(*tasks)
    return list(results)


async def post_comment_instagram(
    media_id: str,
    message: str,
    access_token: str,
) -> Dict[str, Any]:
    """
    Post a new top-level comment on an Instagram media object.
    POST /{media-id}/comments  with message=...
    """
    token_type = _ig_token_type(access_token)
    if token_type == "basic_display":
        raise ValueError(_BUSINESS_GRAPH_REQUIRED)

    url = f"https://graph.facebook.com/v24.0/{media_id}/comments"
    data = {
        "message": message,
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, data=data)
            resp.raise_for_status()
            result = resp.json()

        logger.info(f"✅ Instagram: posted comment on media {media_id}")
        return {"success": True, "id": result.get("id", "")}

    except Exception as e:
        logger.error(f"❌ Instagram post_comment error on {media_id}: {e}")
        return {"success": False, "error": str(e)}


async def post_reply_instagram(
    comment_id: str,
    reply_text: str,
    access_token: str,
) -> bool:
    """
    Post a reply to a specific comment on Instagram.
    Uses Meta Graph API: POST /{comment-id}/replies with message=...
    """
    url = f"https://graph.facebook.com/v24.0/{comment_id}/replies"
    data = {
        "message": reply_text,
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, data=data)
            resp.raise_for_status()

        logger.info(f"✅ Instagram reply posted on comment {comment_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Instagram post_reply error on comment {comment_id}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# YOUTUBE (YouTube Data API v3)
# ══════════════════════════════════════════════════════════════

async def fetch_comments_youtube(
    video_id: str,
    access_token: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch comment threads on a YouTube video.
    Uses: GET https://www.googleapis.com/youtube/v3/commentThreads
    """
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(limit, 100),
        "order": "time",
        "textFormat": "plainText",
    }
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        comments = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            comments.append(_normalize_comment(
                platform="youtube",
                comment_id=item.get("snippet", {}).get("topLevelComment", {}).get("id", ""),
                author=snippet.get("authorDisplayName", "unknown"),
                text=snippet.get("textDisplay", ""),
                timestamp=snippet.get("publishedAt", ""),
                parent_id=item.get("id", ""),  # Thread ID
                like_count=snippet.get("likeCount", 0),
            ))

        logger.info(f"▶️ YouTube: fetched {len(comments)} comments for video {video_id}")
        return comments

    except Exception as e:
        logger.error(f"❌ YouTube fetch_comments error for {video_id}: {e}")
        return []


async def post_reply_youtube(
    parent_comment_id: str,
    reply_text: str,
    access_token: str,
) -> bool:
    """
    Post a reply to a YouTube comment.
    Uses: POST https://www.googleapis.com/youtube/v3/comments
    """
    url = "https://www.googleapis.com/youtube/v3/comments"
    params = {"part": "snippet"}
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "snippet": {
            "parentId": parent_comment_id,
            "textOriginal": reply_text,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, params=params, headers=headers, json=body)
            resp.raise_for_status()

        logger.info(f"✅ YouTube reply posted on comment {parent_comment_id}")
        return True

    except Exception as e:
        logger.error(f"❌ YouTube post_reply error on {parent_comment_id}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# FACEBOOK (Meta Graph API)
# ══════════════════════════════════════════════════════════════

async def fetch_comments_facebook(
    post_id: str,
    access_token: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch comments on a Facebook post."""
    url = f"https://graph.facebook.com/v24.0/{post_id}/comments"
    params = {
        "fields": "id,message,from,created_time,like_count",
        "limit": min(limit, 100),
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                error_body = resp.text
                logger.error(f"❌ Facebook comments API error for {post_id}: HTTP {resp.status_code} — {error_body}")
                return []
            data = resp.json()

        comments = []
        for c in data.get("data", []):
            author = c.get("from", {}).get("name", "unknown")
            comments.append(_normalize_comment(
                platform="facebook",
                comment_id=c.get("id", ""),
                author=author,
                text=c.get("message", ""),
                timestamp=c.get("created_time", ""),
                like_count=c.get("like_count", 0),
            ))

        logger.info(f"📘 Facebook: fetched {len(comments)} comments for post {post_id}")
        return comments

    except Exception as e:
        logger.error(f"❌ Facebook fetch_comments error for {post_id}: {e}")
        return []


async def post_reply_facebook(
    comment_id: str,
    reply_text: str,
    access_token: str,
) -> bool:
    """Post a reply to a Facebook comment."""
    url = f"https://graph.facebook.com/v24.0/{comment_id}/comments"
    data = {
        "message": reply_text,
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, data=data)
            resp.raise_for_status()

        logger.info(f"✅ Facebook reply posted on comment {comment_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Facebook post_reply error on {comment_id}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# TIKTOK (Content Posting API v2)
# ══════════════════════════════════════════════════════════════

async def fetch_comments_tiktok(
    video_id: str,
    access_token: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch comments on a TikTok video."""
    url = "https://open.tiktokapis.com/v2/comment/list/"
    params = {
        "video_id": video_id,
        "max_count": min(limit, 100),
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=params)
            resp.raise_for_status()
            data = resp.json()

        comments = []
        for c in data.get("data", {}).get("comments", []):
            comments.append(_normalize_comment(
                platform="tiktok",
                comment_id=c.get("comment_id", ""),
                author=c.get("user", {}).get("display_name", "unknown"),
                text=c.get("text", ""),
                timestamp=str(c.get("create_time", "")),
            ))

        logger.info(f"🎵 TikTok: fetched {len(comments)} comments for video {video_id}")
        return comments

    except Exception as e:
        logger.error(f"❌ TikTok fetch_comments error for {video_id}: {e}")
        return []


async def post_reply_tiktok(
    comment_id: str,
    video_id: str,
    reply_text: str,
    access_token: str,
) -> bool:
    """Post a reply to a TikTok comment."""
    url = "https://open.tiktokapis.com/v2/comment/reply/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "comment_id": comment_id,
        "video_id": video_id,
        "text": reply_text,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()

        logger.info(f"✅ TikTok reply posted on comment {comment_id}")
        return True

    except Exception as e:
        logger.error(f"❌ TikTok post_reply error on {comment_id}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# LINKEDIN (LinkedIn API v2)
# ══════════════════════════════════════════════════════════════

async def fetch_comments_linkedin(
    activity_urn: str,
    access_token: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch comments on a LinkedIn post/share."""
    url = f"https://api.linkedin.com/v2/socialActions/{activity_urn}/comments"
    params = {"count": min(limit, 100), "start": 0}
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        comments = []
        for c in data.get("elements", []):
            comments.append(_normalize_comment(
                platform="linkedin",
                comment_id=c.get("$URN", c.get("commentUrn", "")),
                author=c.get("actor", "unknown"),
                text=c.get("message", {}).get("text", ""),
                timestamp=str(c.get("created", {}).get("time", "")),
            ))

        logger.info(f"💼 LinkedIn: fetched {len(comments)} comments for {activity_urn}")
        return comments

    except Exception as e:
        logger.error(f"❌ LinkedIn fetch_comments error for {activity_urn}: {e}")
        return []


async def post_reply_linkedin(
    activity_urn: str,
    parent_comment_urn: str,
    reply_text: str,
    access_token: str,
    author_urn: str = "",
) -> bool:
    """Post a reply to a LinkedIn comment."""
    url = f"https://api.linkedin.com/v2/socialActions/{activity_urn}/comments"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    body = {
        "actor": author_urn,
        "message": {"text": reply_text},
        "parentComment": parent_comment_urn,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()

        logger.info(f"✅ LinkedIn reply posted on {parent_comment_urn}")
        return True

    except Exception as e:
        logger.error(f"❌ LinkedIn post_reply error on {parent_comment_urn}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# THREADS (Meta Graph API)
# ══════════════════════════════════════════════════════════════

async def fetch_comments_threads(
    media_id: str,
    access_token: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Fetch replies on a Threads post."""
    url = f"https://graph.threads.net/v1.0/{media_id}/replies"
    params = {
        "fields": "id,text,username,timestamp,like_count",
        "limit": min(limit, 100),
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        comments = []
        for c in data.get("data", []):
            comments.append(_normalize_comment(
                platform="threads",
                comment_id=c.get("id", ""),
                author=c.get("username", "unknown"),
                text=c.get("text", ""),
                timestamp=c.get("timestamp", ""),
                like_count=c.get("like_count", 0),
            ))

        logger.info(f"🧵 Threads: fetched {len(comments)} replies for {media_id}")
        return comments

    except Exception as e:
        logger.error(f"❌ Threads fetch_comments error for {media_id}: {e}")
        return []


async def post_reply_threads(
    media_id: str,
    reply_text: str,
    access_token: str,
    user_id_threads: str = "",
) -> bool:
    """
    Post a reply on Threads.
    Step 1: Create reply container. Step 2: Publish it.
    """
    # Step 1: Create reply container
    create_url = f"https://graph.threads.net/v1.0/{user_id_threads}/threads"
    create_data = {
        "media_type": "TEXT",
        "text": reply_text,
        "reply_to_id": media_id,
        "access_token": access_token,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            # Create
            resp = await client.post(create_url, data=create_data)
            resp.raise_for_status()
            container_id = resp.json().get("id")

            if not container_id:
                logger.error("Threads: No container ID returned")
                return False

            # Step 2: Publish
            publish_url = f"https://graph.threads.net/v1.0/{user_id_threads}/threads_publish"
            publish_data = {
                "creation_id": container_id,
                "access_token": access_token,
            }
            resp2 = await client.post(publish_url, data=publish_data)
            resp2.raise_for_status()

        logger.info(f"✅ Threads reply posted on {media_id}")
        return True

    except Exception as e:
        logger.error(f"❌ Threads post_reply error on {media_id}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# DELETE COMMENT — Platform Implementations
# ══════════════════════════════════════════════════════════════

async def delete_comment_instagram(comment_id: str, access_token: str) -> bool:
    """
    Delete a comment on Instagram via Meta Graph API.
    DELETE /v24.0/{comment-id}?access_token=...
    Requires: instagram_basic + instagram_manage_comments permissions.
    """
    url = f"https://graph.facebook.com/v24.0/{comment_id}"
    params = {"access_token": access_token}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.delete(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        success = data.get("success", False)
        if success:
            logger.info(f"🗑️ Instagram: deleted comment {comment_id}")
        else:
            logger.warning(f"⚠️ Instagram: delete returned success=False for {comment_id}")
        return bool(success)
    except Exception as e:
        logger.error(f"❌ Instagram delete_comment error for {comment_id}: {e}")
        return False


async def delete_comment_facebook(comment_id: str, access_token: str) -> bool:
    """
    Delete a comment on Facebook via Graph API.
    DELETE /v24.0/{comment-id}
    """
    url = f"https://graph.facebook.com/v24.0/{comment_id}"
    params = {"access_token": access_token}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.delete(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        success = data.get("success", False)
        if success:
            logger.info(f"🗑️ Facebook: deleted comment {comment_id}")
        return bool(success)
    except Exception as e:
        logger.error(f"❌ Facebook delete_comment error for {comment_id}: {e}")
        return False


async def delete_comment_youtube(comment_id: str, access_token: str) -> bool:
    """
    Delete a comment on YouTube Data API v3.
    DELETE https://www.googleapis.com/youtube/v3/comments?id={commentId}
    Returns 204 No Content on success.
    """
    url = "https://www.googleapis.com/youtube/v3/comments"
    params = {"id": comment_id}
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.delete(url, params=params, headers=headers)
        if resp.status_code == 204:
            logger.info(f"🗑️ YouTube: deleted comment {comment_id}")
            return True
        logger.warning(f"⚠️ YouTube: delete returned {resp.status_code} for {comment_id}: {resp.text}")
        return False
    except Exception as e:
        logger.error(f"❌ YouTube delete_comment error for {comment_id}: {e}")
        return False


async def delete_comment_tiktok(comment_id: str, video_id: str, access_token: str) -> bool:
    """
    Delete a TikTok comment via TikTok Research API v2.
    POST /v2/comment/delete/  (TikTok uses POST for deletions)
    """
    url = "https://open.tiktokapis.com/v2/comment/delete/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {"video_id": video_id, "comment_id": comment_id}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        success = data.get("data", {}).get("error_code", -1) == 0
        if success:
            logger.info(f"🗑️ TikTok: deleted comment {comment_id}")
        return success
    except Exception as e:
        logger.error(f"❌ TikTok delete_comment error for {comment_id}: {e}")
        return False


async def delete_comment_threads(comment_id: str, access_token: str) -> bool:
    """
    Delete a Threads reply.
    DELETE /v1.0/{comment-id}?access_token=...
    """
    url = f"https://graph.threads.net/v1.0/{comment_id}"
    params = {"access_token": access_token}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.delete(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        success = data.get("success", False)
        if success:
            logger.info(f"🗑️ Threads: deleted comment {comment_id}")
        return bool(success)
    except Exception as e:
        logger.error(f"❌ Threads delete_comment error for {comment_id}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# UNIFIED DISPATCHER
# ══════════════════════════════════════════════════════════════

PLATFORM_FETCHERS = {
    "instagram": fetch_comments_instagram,
    "youtube": fetch_comments_youtube,
    "facebook": fetch_comments_facebook,
    "tiktok": fetch_comments_tiktok,
    "linkedin": fetch_comments_linkedin,
    "threads": fetch_comments_threads,
}

PLATFORM_REPLIERS = {
    "instagram": post_reply_instagram,
    "youtube": post_reply_youtube,
    "facebook": post_reply_facebook,
    "tiktok": post_reply_tiktok,
    "linkedin": post_reply_linkedin,
    "threads": post_reply_threads,
}

PLATFORM_DELETERS = {
    "instagram": delete_comment_instagram,
    "facebook": delete_comment_facebook,
    "youtube": delete_comment_youtube,
    "tiktok": delete_comment_tiktok,
    "threads": delete_comment_threads,
}


async def fetch_comments(
    platform: str,
    post_id: str,
    access_token: str,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Unified comment fetcher — routes to correct platform API.
    """
    fetcher = PLATFORM_FETCHERS.get(platform.lower())
    if not fetcher:
        logger.warning(f"⚠️ No comment fetcher for platform: {platform}")
        return []
    return await fetcher(post_id, access_token, **kwargs)


async def post_reply(
    platform: str,
    comment_id: str,
    reply_text: str,
    access_token: str,
    **kwargs,
) -> bool:
    """
    Unified reply poster — routes to correct platform API.
    
    Extra kwargs are forwarded to platform-specific functions
    (e.g. video_id for TikTok, activity_urn for LinkedIn).
    """
    platform_lower = platform.lower()

    if platform_lower == "instagram":
        return await post_reply_instagram(comment_id, reply_text, access_token)
    elif platform_lower == "youtube":
        return await post_reply_youtube(comment_id, reply_text, access_token)
    elif platform_lower == "facebook":
        return await post_reply_facebook(comment_id, reply_text, access_token)
    elif platform_lower == "tiktok":
        video_id = kwargs.get("video_id", "")
        return await post_reply_tiktok(comment_id, video_id, reply_text, access_token)
    elif platform_lower == "linkedin":
        activity_urn = kwargs.get("activity_urn", "")
        author_urn = kwargs.get("author_urn", "")
        return await post_reply_linkedin(activity_urn, comment_id, reply_text, access_token, author_urn)
    elif platform_lower == "threads":
        user_id_threads = kwargs.get("user_id_threads", "")
        return await post_reply_threads(comment_id, reply_text, access_token, user_id_threads)
    else:
        logger.warning(f"⚠️ No reply handler for platform: {platform}")
        return False


async def delete_comment(
    platform: str,
    comment_id: str,
    access_token: str,
    **kwargs,
) -> bool:
    """
    Unified comment deleter — routes to correct platform API.

    Extra kwargs:
      video_id  — required for TikTok comment deletion
    """
    platform_lower = platform.lower()

    if platform_lower == "instagram":
        return await delete_comment_instagram(comment_id, access_token)
    elif platform_lower == "facebook":
        return await delete_comment_facebook(comment_id, access_token)
    elif platform_lower == "youtube":
        return await delete_comment_youtube(comment_id, access_token)
    elif platform_lower == "tiktok":
        video_id = kwargs.get("video_id", "")
        return await delete_comment_tiktok(comment_id, video_id, access_token)
    elif platform_lower == "threads":
        return await delete_comment_threads(comment_id, access_token)
    else:
        logger.warning(f"⚠️ No delete handler for platform: {platform}")
        return False
