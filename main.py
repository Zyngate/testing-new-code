# stelle_backend/main.py

import asyncio
import os
from datetime import datetime, timezone
from config import FROM_EMAIL, VAPID_CLAIMS
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from database import load_faiss_indices

from services.subscription_service import (
    daily_checkin_scheduler,
    proactive_checkin_scheduler,
    notification_sender_loop
)

from services.task_service import task_thread
from config import logger

# Import all routers
from routes import (
    auth_routes,
    chat_routes,
    content_routes,
    goal_routes,
    misc_routes,
    subscription_routes,
    task_routes,
    integration_routes       # ← YOUR UPDATED CAPTION ROUTES ARE HERE
)

# ----------------------------------------------------------------
# FastAPI Initialization
# ----------------------------------------------------------------
app = FastAPI(
    title="Stelle Backend API",
    description="Modularized AI Assistant Backend with RAG, Goal Management, and Scheduling.",
    version="1.0.0",
)

# ----------------------------------------------------------------
# CORS
# ----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# Global Error Handler
# ----------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error, please try again later."},
    )

# ----------------------------------------------------------------
# ROUTERS
# ----------------------------------------------------------------
app.include_router(auth_routes.router, tags=["Authentication"])
app.include_router(chat_routes.router, tags=["Chat & AI Generation"])
app.include_router(content_routes.router, tags=["Content & Visualization"])
app.include_router(goal_routes.router, tags=["Goals & Planning"])
app.include_router(misc_routes.router, tags=["Utilities & Research"])
app.include_router(subscription_routes.router, tags=["Notifications"])
app.include_router(task_routes.router, tags=["Recurring Tasks"])
app.include_router(integration_routes.router, tags=["Integrations"])   # ← CAPTION + LINKEDIN ENDPOINTS

# ----------------------------------------------------------------
# Startup Tasks
# ----------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

    # 1. Load FAISS Indices
    await load_faiss_indices()

    # 2. Start Schedulers
    asyncio.create_task(notification_sender_loop())
    asyncio.create_task(daily_checkin_scheduler())
    asyncio.create_task(proactive_checkin_scheduler())

    # 3. Start Sync Task Thread
    if not task_thread.is_alive():
        task_thread.start()
        logger.info("Synchronous Task Scheduler thread started.")
    else:
        logger.info("Synchronous Task Scheduler thread already running.")

# ----------------------------------------------------------------
# Root Endpoint
# ----------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Stelle Backend is running successfully!",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ----------------------------------------------------------------
# Local development entrypoint
# ----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
