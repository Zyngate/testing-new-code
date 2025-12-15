import uuid
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException

from config import logger
from services.visualize_service import visualize_content

# ============================================================
# ROUTER CONFIG
# ============================================================

router = APIRouter(
    prefix="/visualize",
    tags=["Visualization"]
)

# ============================================================
# IN-MEMORY JOB STORE
# ============================================================

# NOTE:
# - This is fine for now
# - Later you can move this to Redis if needed
visualize_jobs: Dict[str, str] = {}

# ============================================================
# START VISUALIZATION (REST)
# ============================================================

@router.post("/start")
async def start_visualize(request: Request):
    """
    Starts a visualization job.
    Returns a visualize_id for WebSocket connection.
    """
    data = await request.json()
    text = data.get("text")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    visualize_id = str(uuid.uuid4())
    visualize_jobs[visualize_id] = text

    logger.info(f"[VISUALIZE] Job created: {visualize_id}")

    return {"visualize_id": visualize_id}

# ============================================================
# VISUALIZATION WEBSOCKET
# ============================================================

@router.websocket("/ws/{visualize_id}")
async def visualize_ws(websocket: WebSocket, visualize_id: str):
    """
    WebSocket endpoint that:
    - accepts visualize_id
    - generates HTML visualization
    - sends it back to frontend
    """
    await websocket.accept()

    prompt_text = visualize_jobs.pop(visualize_id, None)

    if not prompt_text:
        await websocket.send_json({
            "status": "failed",
            "error": "Invalid or expired visualize_id"
        })
        await websocket.close()
        return

    try:
        # Notify frontend that processing started
        await websocket.send_json({
            "status": "processing",
            "message": "Generating visualization..."
        })

        # Core visualization logic
        result = await visualize_content(prompt_text)

        if result.get("status") == "success":
            await websocket.send_json({
                "status": "success",
                "html": result["html"]
            })
        else:
            await websocket.send_json({
                "status": "failed",
                "error": "Visualization generation failed"
            })

    except WebSocketDisconnect:
        logger.info(f"[VISUALIZE] WebSocket disconnected: {visualize_id}")

    except Exception as e:
        logger.error(f"[VISUALIZE] Error: {e}", exc_info=True)
        await websocket.send_json({
            "status": "failed",
            "error": "Internal server error"
        })

    finally:
        await websocket.close()
