# stelle_backend/routes/image_caption_routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.image_caption_service import caption_from_image_file
from groq import Groq
import os
from config import GROQ_API_KEY_CAPTION

router = APIRouter(tags=["Image Caption Generator"])

@router.post("/generate_image_caption")
async def generate_image_caption(
    image: UploadFile = File(...),
    platforms: str = "instagram"
):
    try:
        platforms_list = [p.strip().lower() for p in platforms.split(",")]

        # Save temp file
        contents = await image.read()
        temp_path = f"tmp_image_{image.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)

        client = Groq(api_key=GROQ_API_KEY_CAPTION)

        result = await caption_from_image_file(temp_path, platforms_list, client)

        os.remove(temp_path)

        return {"status": "success", "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
