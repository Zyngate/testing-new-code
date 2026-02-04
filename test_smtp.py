import time
import asyncio
from services.video_caption_service import (
    extract_audio_from_video,
    get_transcript_groq
)

VIDEO_PATH = r"C:\Users\DELL\Downloads\nipah.mp4"


async def run_stt_test():
    start = time.time()

    audio_path = await extract_audio_from_video(VIDEO_PATH)
    transcript = await get_transcript_groq(audio_path)

    end = time.time()

    print("====== STT ONLY TEST ======")
    print(f"Transcript length: {len(transcript)} characters")
    print(f"STT time: {end - start:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(run_stt_test())
