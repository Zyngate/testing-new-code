# stelle_backend/main.py

import asyncio
from datetime import datetime, timezone
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import threading
from config import logger
from database import load_faiss_indices
from routes import image_caption_routes
from routes import recommendation_routes
from routes import video_routes
from routes import visualize_routes



#from routes import recommendation_routes


# Services (Schedulers)
from services.subscription_service import (
    daily_checkin_scheduler,
    proactive_checkin_scheduler,
    notification_sender_loop
)
from services.task_service import start_task_scheduler
from routes import post_generation_routes
from routes.post_routes import router as post_router

# Routers grouped by FEATURES
from routes import (
    auth_routes,
    chat_routes,
    content_routes,
    goal_routes,
    misc_routes,
    subscription_routes,
    task_routes,
    integration_routes, 
    post_generation_routes,
    plan_routes,
    cloudinary_routes
)

app = FastAPI(
    title="Stelle Backend API",
    version="1.0.0",
)

app.include_router(cloudinary_routes.router)

# -------------------------
#   CORS CONFIG
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://www.stelle.chat",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------
#   GLOBAL ERROR HANDLER
# -------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Unexpected server error"}
    )

# ✅ Post Scheduling Module
app.include_router(
    post_router,
    prefix="/posts",
    tags=["Posts"]
)

# ✅ Weekly Plan Manager
app.include_router(
    plan_routes.router,
    prefix="/plans",
    tags=["Plan Manager"]
)


# -------------------------
#   FEATURE-WISE ROUTERS
# -------------------------

# ✅ Authentication Module
app.include_router(
    auth_routes.router,
    prefix="/auth",
    tags=["Authentication"]
)

app.include_router(
    post_generation_routes.router,
    prefix="/caption-generator",
    tags=["Caption & Hashtag Generator"]
)

app.include_router(
    recommendation_routes.router,
    prefix="/recommendation",
    tags=["Recommendation Engine"]
)

# VIDEO Caption Generator (video → captions)
app.include_router(
    video_routes.router,
    prefix="/caption-generator",
    tags=["Video Caption Generator"]
)


# ✅ Chat / AI-Assistance Module
app.include_router(
    chat_routes.router,
    prefix="/aiassist",
    tags=["Chat"]
)

app.include_router(
    visualize_routes.router,
    tags=["Visualization"]
)


# ✅ Content Intelligence (upload, images, visualization)
app.include_router(
    content_routes.router,
    prefix="/content",
    tags=["Content"]
)

# ✅ Goals + Weekly Planning
app.include_router(
    goal_routes.router,
    prefix="/goals",
    tags=["Goals"]
)

# ✅ Research + History + Deepsearch
app.include_router(
    misc_routes.router,
    prefix="/utils",
    tags=["Utilities"]
)

# ✅ Push Notifications + Quotes
app.include_router(
    subscription_routes.router,
    prefix="/notifications",
    tags=["Notifications"]
)

# ✅ Tasks Module
app.include_router(
    task_routes.router,
    prefix="/tasks",
    tags=["Tasks"]
)

# ✅ Integrations (LinkedIn, scraping, etc.)
app.include_router(
    integration_routes.router,
    prefix="/integrations",
    tags=["Integrations"]
)

app.include_router(
    image_caption_routes.router,
    prefix="/caption-generator",
    tags=["Caption & Hashtag Generator"]
)

# -------------------------
#   STARTUP EVENTS
# -------------------------
from database import get_or_init_sync_collections

@app.on_event("startup")
async def startup():
    await load_faiss_indices()

    # Reset stuck tasks (this is fine)
    tasks_col, _ = get_or_init_sync_collections()
    if tasks_col is not None:
        tasks_col.update_many(
            {"status": "running"},
            {"$set": {"status": "scheduled", "retrieved": False}}
        )

    # 🚫 DO NOT run schedulers on web service
    if os.getenv("ENABLE_SCHEDULERS") == "true":
        asyncio.create_task(notification_sender_loop())
        asyncio.create_task(daily_checkin_scheduler())
        asyncio.create_task(proactive_checkin_scheduler())
        start_task_scheduler()

# -------------------------
#   ROOT ENDPOINT
# -------------------------
@app.get("/", tags=["Root"])
async def root():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

