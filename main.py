# stelle_backend/main.py
import asyncio
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Request
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
    integration_routes
)

# --- FastAPI Initialization ---
app = FastAPI(
    title="Stelle Backend API",
    description="Modularized AI Assistant Backend with RAG, Goal Management, and Scheduling.",
    version="1.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error, please try again later."},
    )

# --- Include Routers ---
app.include_router(auth_routes.router, tags=["Authentication"])
app.include_router(chat_routes.router, tags=["Chat & AI Generation"])
app.include_router(content_routes.router, tags=["Content & Visualization"])
app.include_router(goal_routes.router, tags=["Goals & Planning"])
app.include_router(misc_routes.router, tags=["Utilities & Research"])
app.include_router(subscription_routes.router, tags=["Notifications"])
app.include_router(task_routes.router, tags=["Recurring Tasks"])
app.include_router(integration_routes.router, tags=["Integrations"])

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

    # Load FAISS indices asynchronously
    await load_faiss_indices()

    # Start async schedulers for notifications and check-ins
    asyncio.create_task(notification_sender_loop())
    asyncio.create_task(daily_checkin_scheduler())
    asyncio.create_task(proactive_checkin_scheduler())

    # Start synchronous task scheduler thread
    if not task_thread.is_alive():
        task_thread.start()
        logger.info("Synchronous Task Scheduler thread started.")
    else:
        logger.info("Synchronous Task Scheduler thread is already running.")

# --- Root Endpoint (status check) ---
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Stelle Backend is running successfully!",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# --- Main Execution Block for Local Dev & Cloud ---
if __name__ == "__main__":
    import uvicorn

    # Use Render-assigned port if available, fallback to 8000 for local dev
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
