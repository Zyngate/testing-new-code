# stelle_backend/services/subscription_service.py
import asyncio
from datetime import datetime, timezone, timedelta
import pytz
from pywebpush import webpush, WebPushException
import json
import logging

from database import users_collection, notifications_collection, goals_collection
from config import logger, VAPID_PRIVATE_KEY, VAPID_CLAIMS

# --- Push Notification Core Helper ---

async def schedule_notification(
    user_id: str,
    message: str,
    scheduled_time: datetime,
    notif_type: str = "general",
):
    """Schedules a push notification to be sent later."""
    notif = {
        "user_id": user_id,
        "message": message,
        # Ensure scheduled_time is timezone-aware UTC
        "scheduled_time": scheduled_time.replace(tzinfo=timezone.utc), 
        "type": notif_type,
        "sent": False,
        "created_at": datetime.now(timezone.utc),
    }
    await notifications_collection.insert_one(notif)
    logger.info(
        f"Notification scheduled for user {user_id} at {scheduled_time} with type '{notif_type}'."
    )

# --- Background Schedulers (Called from main.py startup) ---

async def notification_sender_loop():
    """Continuously checks for and sends due push notifications."""
    while True:
        try: # Necessary try block
            now = datetime.now(timezone.utc)
            cursor = notifications_collection.find(
                {"scheduled_time": {"$lte": now}, "sent": False}
            )
            async for notif in cursor:
                user_id = notif["user_id"]
                message = notif["message"]
                user = await users_collection.find_one({"user_id": user_id})
                
                if not user or "push_subscription" not in user:
                    await notifications_collection.update_one(
                        {"_id": notif["_id"]},
                        {"$set": {"sent": True, "status": "skipped_no_subscription"}},
                    )
                    continue
                    
                subscription_info = user["push_subscription"]
                payload = json.dumps(
                    {
                        "title": "Stelle Team",
                        "body": message,
                        "icon": "https://media-hosting.imagekit.io/2b232c0c6a354b82/2.png?Expires=1839491017&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=...",
                        "data": {"url": "https://stelle.chat"},
                    }
                )
                
                try:
                    # Webpush is synchronous, run it in a thread
                    await asyncio.to_thread(
                        webpush,
                        subscription_info,
                        data=payload,
                        vapid_private_key=VAPID_PRIVATE_KEY,
                        vapid_claims=VAPID_CLAIMS,
                    )
                    await notifications_collection.update_one(
                        {"_id": notif["_id"]},
                        {"$set": {"sent": True, "sent_at": datetime.now(timezone.utc)}},
                    )
                    logger.info(f"Push notification sent to user {user_id}")
                except WebPushException as ex:
                    logger.warning(f"WebPush failed for user {user_id}: {ex}")
                    await asyncio.sleep(60) 
        
        except Exception as e:
             logger.error(f"Error in notification_sender_loop: {e}")
        
        await asyncio.sleep(10)

async def daily_checkin_scheduler():
    """Schedules a daily check-in notification at 9:00 AM local time for each user."""
    while True:
        try:
            async for user in users_collection.find():
                user_id = user["user_id"]
                tz_name = user.get("time_zone", "UTC")
                
                try:
                    user_tz = pytz.timezone(tz_name)
                except Exception:
                    user_tz = pytz.UTC
                    
                now_local = datetime.now(user_tz)
                
                # Target time: 9:00 AM local time check window
                if 8 * 60 + 55 <= (now_local.hour * 60 + now_local.minute) <= 9 * 60 + 5:
                    checkin_time_local = now_local.replace(hour=9, minute=0, second=0, microsecond=0)
                    checkin_time_utc = user_tz.localize(checkin_time_local.replace(tzinfo=None)).astimezone(pytz.UTC)
                    
                    existing = await notifications_collection.find_one(
                        {"user_id": user_id, "type": "daily_checkin", "scheduled_time": checkin_time_utc}
                    )
                    
                    if existing:
                        continue
                        
                    goals_cursor = goals_collection.find(
                        {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
                    )
                    active_goals = [g["title"] for g in await goals_cursor.to_list(None)]
                    
                    if active_goals:
                        message = f"Good morning! Your goals for today: {', '.join(active_goals)}. Keep it up!"
                    else:
                        message = "Good morning! Set some goals today to stay on track!"
                        
                    await schedule_notification(
                        user_id, message, checkin_time_utc, notif_type="daily_checkin"
                    )
        
        except Exception as e:
            logger.error(f"Error in daily_checkin_scheduler: {e}")
            
        await asyncio.sleep(60) 

async def proactive_checkin_scheduler():
    """Schedules proactive notifications at 9 AM, 2 PM, and 7 PM local time."""
    checkin_times = [
        {"hour": 9, "minute": 0, "type": "proactive_morning"},
        {"hour": 14, "minute": 0, "type": "proactive_afternoon"},
        {"hour": 19, "minute": 0, "type": "proactive_evening"},
    ]
    
    while True:
        try:
            async for user in users_collection.find():
                user_id = user["user_id"]
                tz_name = user.get("time_zone", "UTC")
                
                try:
                    user_tz = pytz.timezone(tz_name)
                except Exception:
                    user_tz = pytz.UTC
                    
                now_local = datetime.now(user_tz)
                
                for checkin in checkin_times:
                    target = now_local.replace(
                        hour=checkin["hour"], minute=checkin["minute"], second=0, microsecond=0
                    )
                    
                    if abs((now_local - target).total_seconds()) <= 15 * 60:
                        target_utc = user_tz.localize(target.replace(tzinfo=None)).astimezone(pytz.UTC)
                        
                        existing = await notifications_collection.find_one(
                            {"user_id": user_id, "type": checkin["type"], "scheduled_time": target_utc}
                        )
                        if existing:
                            continue
                            
                        goals_cursor = goals_collection.find(
                            {"user_id": user_id, "status": {"$in": ["active", "in progress"]}}
                        )
                        active_goals = [g["title"] for g in await goals_cursor.to_list(None)]
                        
                        if active_goals:
                            message = f"Hi there! How are you progressing on your goals ({', '.join(active_goals)})? We're here to support you!"
                        else:
                            message = "Hi there! How are you doing today? Let us know if you need any help."
                            
                        await schedule_notification(
                            user_id, message, target_utc, notif_type=checkin["type"]
                        )
        
        except Exception as e:
            logger.error(f"Error in proactive_checkin_scheduler: {e}")
            
        await asyncio.sleep(60)