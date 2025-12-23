# stelle/services/social_publish_service.py

from config import logger


def publish_instagram(
    media_url: str,
    caption: str,
    access_token: str,
    account_id: str
) -> bool:
    """
    MVP mock publisher.
    This simulates posting to Instagram.

    Returns:
        True  -> post succeeded
        False -> post failed
    """

    try:
        logger.info("📸 [INSTAGRAM MOCK POST]")
        logger.info(f"Account ID: {account_id}")
        logger.info(f"Media URL: {media_url}")
        logger.info(f"Caption: {caption}")
        logger.info("✅ Mock Instagram post successful")

        # simulate success
        return True

    except Exception as e:
        logger.error(f"❌ Instagram mock post failed: {e}")
        return False
