import uuid
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from config import logger
from services.visualize_service import visualize_content

router = APIRouter(tags=["Visualize"])

# In-memory job store
visualize_jobs: Dict[str, Dict] = {}

# -------------------------------
# START VISUALIZATION
# -------------------------------
@router.post("/start_visualize")
async def start_visualize(request: Request):
    data = await request.json()
    text = data.get("text")

    visualize_id = str(uuid.uuid4())
    visualize_jobs[visualize_id] = {"text": text}

    return {"visualize_id": visualize_id}

# -------------------------------
# WEBSOCKET VISUALIZATION
# -------------------------------
@router.websocket("/ws/visualize/{visualize_id}")
async def visualize_ws(websocket: WebSocket, visualize_id: str):
    await websocket.accept()

    job = visualize_jobs.pop(visualize_id, None)
    if not job:
        await websocket.send_json({
            "status": "failed",
            "error": "Invalid visualize_id"
        })
        await websocket.close()
        return

    try:
        await websocket.send_json({
            "status": "processing",
            "message": "Generating visualization..."
        })

        result = await visualize_content(job["text"])

        if result["status"] == "success":
            await websocket.send_json({
                "status": "success",
                "html": result["html"]
            })
        else:
            await websocket.send_json({
                "status": "failed",
                "error": "Visualization failed"
            })

    except WebSocketDisconnect:
        logger.info("Visualize WS disconnected")

    except Exception as e:
        logger.error(e)
        await websocket.send_json({
            "status": "failed",
            "error": "Internal error"
        })

    finally:
        await websocket.close()
