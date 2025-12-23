# stelle/services/auth_repo.py

from typing import Optional, Dict, Any
from database import db
from config import logger


def get_social_auth(user_id: str, platform: str) -> Optional[Dict[str, Any]]:
    """
    Fetch social platform auth credentials for a user.

    Args:
        user_id (str): Internal user ID
        platform (str): Platform name (instagram / threads / youtube)

    Returns:
        dict | None: Auth document if found, else None
    """
    try:
        platform = platform.lower().strip()

        auth = db.oauthcredentials.find_one({
            "userId": user_id,
            "platform": platform
        })

        if not auth:
            logger.warning(
                f"[AUTH] No credentials found for user={user_id}, platform={platform}"
            )

        return auth

    except Exception as e:
        logger.error(
            f"[AUTH] Error fetching credentials for user={user_id}, platform={platform}: {e}",
            exc_info=True
        )
        return None
