# services/post_scheduler.py

import time
from datetime import datetime, timezone
from database import db
from .social_publish_service import publish_instagram
from config import logger

CHECK_INTERVAL_SECONDS = 30


def post_scheduler():
    logger.info("🚀 Post Scheduler started")

    posts_col = db["scheduledposts"]
    auth_col = db["oauthcredentials"]

    while True:
        try:
            now = datetime.now(timezone.utc)

            # 1️⃣ Fetch due posts
            posts = posts_col.find({
                "status": "scheduled",
                "scheduledAt": {"$lte": now}
            })

            for post in posts:
                post_id = post["_id"]
                platform = post.get("platform", "").lower()
                user_id = post.get("userId")

                logger.info(f"⏰ Processing post {post_id} for user {user_id}")

                # 2️⃣ Mark as posting
                posts_col.update_one(
                    {"_id": post_id},
                    {
                        "$set": {
                            "status": "posting",
                            "updatedAt": now
                        }
                    }
                )

                # 3️⃣ Fetch OAuth credentials
                # Use case-insensitive regex because DB stores "Instagram" not "instagram"
                import re
                auth = auth_col.find_one({
                    "userId": user_id,
                    "platform": {"$regex": f"^{re.escape(platform)}$", "$options": "i"}
                })

                if not auth:
                    logger.error(f"❌ Auth not found for user={user_id}, platform={platform}")
                    posts_col.update_one(
                        {"_id": post_id},
                        {
                            "$set": {
                                "status": "failed",
                                "failureReason": "AUTH_NOT_FOUND",
                                "updatedAt": datetime.now(timezone.utc)
                            }
                        }
                    )
                    continue

                # 4️⃣ Publish (mock or real)
                success = False

                if platform == "instagram":
                    success = publish_instagram(
                        media_url=post["mediaUrls"][0],
                        caption=post.get("caption", ""),
                        access_token=auth["accessToken"],
                        account_id=auth["accountId"]
                    )
                else:
                    logger.error(f"❌ Unsupported platform: {platform}")

                # 5️⃣ Final status update
                posts_col.update_one(
                    {"_id": post_id},
                    {
                        "$set": {
                            "status": "posted" if success else "failed",
                            "updatedAt": datetime.now(timezone.utc)
                        }
                    }
                )

                logger.info(
                    f"✅ Post {post_id} finished with status={'posted' if success else 'failed'}"
                )

        except Exception as e:
            logger.error("🔥 Error in post scheduler loop", exc_info=True)

        time.sleep(CHECK_INTERVAL_SECONDS)
