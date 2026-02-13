from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import asyncio
import cloudinary.uploader
from config import logger

router = APIRouter(prefix="/media", tags=["Media Upload"])

MAX_MEDIA_UPLOAD = 20
# Max concurrent Cloudinary uploads to avoid overwhelming the connection
MAX_CONCURRENT_UPLOADS = 5


async def _upload_single_file(file_content: bytes, filename: str, semaphore: asyncio.Semaphore) -> dict:
    """Upload a single file to Cloudinary in a thread (non-blocking), with concurrency control."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: cloudinary.uploader.upload(
                file_content,
                resource_type="auto",
                folder="scheduler_media"
            )
        )
        return {
            "originalName": filename,
            "public_id": result["public_id"],
            "secure_url": result["secure_url"],
            "resource_type": result["resource_type"]
        }


@router.post("/upload-media")
async def upload_media(files: List[UploadFile] = File(...)):
    """
    Upload up to 20 images/videos to Cloudinary in parallel.
    Returns Cloudinary URLs.
    """

    if len(files) > MAX_MEDIA_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_MEDIA_UPLOAD} media files allowed"
        )

    # Validate file types first (fast check before any uploads)
    for file in files:
        if not file.content_type.startswith(("image/", "video/")):
            raise HTTPException(
                status_code=400,
                detail=f"Only image or video files are allowed. '{file.filename}' is {file.content_type}"
            )

    # Read all file contents into memory so Cloudinary uploads don't fight over streams
    file_data = []
    for file in files:
        try:
            content = await file.read()
            file_data.append((content, file.filename))
        except Exception:
            logger.error(f"❌ Failed to read file {file.filename}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to read file: {file.filename}")

    # Upload all files in parallel with a concurrency limit
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
    tasks = [
        _upload_single_file(content, filename, semaphore)
        for content, filename in file_data
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    uploaded_files = []
    failed_files = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"❌ Cloudinary upload failed for {file_data[i][1]}: {result}", exc_info=True)
            failed_files.append(file_data[i][1])
        else:
            uploaded_files.append(result)

    if failed_files and not uploaded_files:
        raise HTTPException(
            status_code=500,
            detail=f"All uploads failed. Failed files: {', '.join(failed_files)}"
        )

    return {
        "success": True,
        "count": len(uploaded_files),
        "failed": failed_files,
        "media": uploaded_files
    }
