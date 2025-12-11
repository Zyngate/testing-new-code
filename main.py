# stelle_backend/main.py

import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import logger
from database import load_faiss_indices

from routes import recommendation_routes


# Services (Schedulers)
from services.subscription_service import (
    daily_checkin_scheduler,
    proactive_checkin_scheduler,
    notification_sender_loop
)
from services.task_service import task_thread

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
    post_generation_routes
)

app = FastAPI(
    title="Stelle Backend API",
    version="1.0.0",
)

# -------------------------
#   CORS CONFIG
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# ✅ Recommendation Engine Module
app.include_router(
    recommendation_routes.router,
    prefix="/recommendation",
    tags=["Recommendation Engine"]
)


# ✅ Chat / AI-Assistance Module
app.include_router(
    chat_routes.router,
    prefix="/aiassist",
    tags=["Chat"]
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

# ✅ Post Generation (captions, content writing)
app.include_router(
    post_generation_routes.router,
    prefix="/content-generation",
    tags=["Content Generation"]
)

# -------------------------
#   STARTUP EVENTS
# -------------------------
@app.on_event("startup")
async def startup():
    await load_faiss_indices()

    asyncio.create_task(notification_sender_loop())
    asyncio.create_task(daily_checkin_scheduler())
    asyncio.create_task(proactive_checkin_scheduler())

    if not task_thread.is_alive():
        task_thread.start()

# -------------------------
#   ROOT ENDPOINT
# -------------------------
@app.get("/", tags=["Root"])
async def root():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
