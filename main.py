# stelle_backend/main.py

import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import logger
from database import load_faiss_indices

from services.subscription_service import (
    daily_checkin_scheduler,
    proactive_checkin_scheduler,
    notification_sender_loop
)
from services.task_service import task_thread

# Routers
from routes import (
    auth_routes,
    chat_routes,
    content_routes,
    goal_routes,
    misc_routes,
    subscription_routes,
    task_routes,
    integration_routes   # <-- YOUR CAPTION ENDPOINT IS HERE
)

app = FastAPI(
    title="Stelle Backend API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def handler(request: Request, exc: Exception):
    logger.error(str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error"})


# ALL ROUTERS INCLUDED
app.include_router(auth_routes.router)
app.include_router(chat_routes.router)
app.include_router(content_routes.router)
app.include_router(goal_routes.router)
app.include_router(misc_routes.router)
app.include_router(subscription_routes.router)
app.include_router(task_routes.router)
app.include_router(integration_routes.router)  # ✅ CAPTION ENDPOINT LOADED


@app.on_event("startup")
async def startup():
    await load_faiss_indices()
    asyncio.create_task(notification_sender_loop())
    asyncio.create_task(daily_checkin_scheduler())
    asyncio.create_task(proactive_checkin_scheduler())
    if not task_thread.is_alive():
        task_thread.start()


@app.get("/")
async def root():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
