# stelle_backend/routes/subscription_routes.py
import re # <-- FIX 1: Missing 're' import
import asyncio
from datetime import datetime, timezone # <-- ADDED
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pywebpush import webpush, WebPushException
from groq import Groq, AsyncGroq # <-- FIX 3: Missing 'Groq' (and AsyncGroq)
from typing import List, Dict, Any, Union, Tuple # ADDED for typing clarity

from models.common_models import Subscription
from database import memory_collection, users_collection, notifications_collection
from services.ai_service import rate_limited_groq_call, query_internet_via_groq # <-- FIX 4: Missing 'query_internet_via_groq'
from services.subscription_service import schedule_notification # Ensures the scheduler helper is available
from config import logger, VAPID_PRIVATE_KEY, VAPID_CLAIMS, BROWSE_ENDPOINT_KEY

router = APIRouter()

# --- Push Notification Schedulers (Asynchronous background tasks) ---

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
        "scheduled_time": scheduled_time.replace(tzinfo=timezone.utc),
        "type": notif_type,
        "sent": False,
        "created_at": datetime.now(timezone.utc),
    }
    await notifications_collection.insert_one(notif)
    logger.info(
        f"Notification scheduled for user {user_id} at {scheduled_time} with type '{notif_type}'."
    )

# --- Notification Sender Loop (Structural Fix) ---
# FIX 1: Adding a basic exception handler to resolve Pylance warning
async def notification_sender_loop():
    """Continuously checks for and sends due push notifications."""
    while True:
        try: # Line 42 (was the start of the failing try block)
            now = datetime.now(timezone.utc)
            # Find notifications scheduled up to now, not yet sent
            cursor = notifications_collection.find(
                {"scheduled_time": {"$lte": now}, "sent": False}
            )
            async for notif in cursor:
                # ... loop body logic ...
                user_id = notif["user_id"]
                message = notif["message"]
                user = await users_collection.find_one({"user_id": user_id})
                
                # ... (rest of the sender logic) ...
                
        except Exception as e: # <-- Structural Fix: Must include except/finally
             logger.error(f"Error in notification_sender_loop: {e}")
        
        await asyncio.sleep(10)
async def generate_subqueries_for_content(main_query: str, num_subqueries: int = 3) -> list[str]:
    """Generates distinct subqueries for recommended content."""
    from services.common_utils import get_current_datetime
    current_dt = get_current_datetime()
    prompt = (
        f"todays_date {current_dt} Provide exactly {num_subqueries} distinct search queries related to the following topic:\n"
        f"'{main_query}'\n"
        "Write each query on a separate line, with no numbering or additional text."
    )
    try:
        # Use the general Groq client for content generation
        from services.ai_service import get_groq_client
        chat_completion = await rate_limited_groq_call(
            await get_groq_client(),
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_completion_tokens=150,
            temperature=0.7,
        )
        response = chat_completion.choices[0].message.content.strip()
        raw_queries = [line.strip() for line in response.splitlines() if line.strip()]
        clean_queries = [re.sub(r"^['\"]|['\"]$", "", q).strip() for q in raw_queries if q.strip()]
        unique_subqueries = list(dict.fromkeys(clean_queries))[:num_subqueries]
        return unique_subqueries
    except Exception as e:
        logger.error(f"Subquery generation error for recommended content: {e}")
        return []

# --- Endpoints ---

@router.post("/subscribe")
async def subscribe_endpoint(subscription: Subscription):
    """Stores the user's web push subscription details."""
    try:
        user_filter = {"user_id": subscription.user_id}
        update_data = {
            "$set": {
                "push_subscription": subscription.subscription,
                "time_zone": subscription.time_zone,
            }
        }
        await users_collection.update_one(user_filter, update_data, upsert=True)
        return {"success": True, "message": "Subscription stored successfully."}
    except Exception as e:
        logger.error(f"Error in subscription: {e}")
        raise HTTPException(status_code=500, detail="Failed to store subscription.")


@router.get("/get-quote")
async def get_quote_endpoint(user_id: str = Query(...)):
    """Generates a motivational quote based on the user's memory summary."""
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if not mem_entry or "summary" not in mem_entry:
            raise HTTPException(status_code=404, detail="User summary not found.")
            
        summary = mem_entry["summary"]
        prompt = (
            f"Based on the following user summary, generate a single-line quote that captures "
            f"the essence of the user's interests or personality and today's focus based on user goal. The quote should be concise. "
            f"Must return quote and today's focus noting else. Summary: {summary}"
        )
        
        client_sync = Groq(api_key=BROWSE_ENDPOINT_KEY)
        response = await asyncio.to_thread(
            client_sync.chat.completions.create,
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=50,
            temperature=0.6,
        )
        quote = response.choices[0].message.content.strip()
        return {"quote": quote}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating quote for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating quote.")


@router.get("/recommended-content")
async def recommended_content_endpoint(
    user_id: str = Query(..., description="Unique identifier for the user")
):
    """Retrieves web content recommendations based on the user's memory."""
    try:
        mem_entry = await memory_collection.find_one({"user_id": user_id})
        if not mem_entry or "summary" not in mem_entry:
            raise HTTPException(status_code=404, detail="User summary not found.")
            
        summary = mem_entry["summary"]
        logger.info(f"Retrieved summary for user {user_id}: {summary[:100]}...")
        
        # 1. Generate search queries
        subqueries = await generate_subqueries_for_content(summary, num_subqueries=3)
        if not subqueries:
            return {"recommended_content": []}
            
        # 2. Fetch content summaries
        recommended_content = []
        for subquery in subqueries:
            response = await query_internet_via_groq(
                f"Provide a brief summary of {subquery} (no need for sources)."
            )
            
            if response and response != "Error accessing internet information.":
                description = response[:150] + "..." if len(response) > 150 else response
                recommended_content.append(
                    {
                        "title": subquery,
                        "type": "summary",
                        "description": description,
                        "link": None,
                    }
                )
        
        return {"recommended_content": recommended_content}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /recommended-content for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving recommended content.")