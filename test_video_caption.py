import asyncio
from services.video_caption_service import caption_from_video_file

async def run_test():
    video_path = r"C:\Users\DELL\Downloads\WhatsApp Video 2026-02-01 at 2.27.19 PM.mp4"

    platforms = ["instagram", "linkedin", "youtube","Tiktok","Threads"]

    result = await caption_from_video_file(video_path, platforms)

    print("\n===== TRANSCRIPT =====")
    print(result["transcript"])

    print("\n===== VISUAL SUMMARY =====")
    print(result["visual_summary"])

    print("\n===== VISUAL CAPTIONS =====")
    for frame, caption in result["visual_captions"]:
        print(frame, " -> ", caption)

    print("\n===== OCR TEXT DETECTED =====")
    for t in result.get("detected_texts", []):
        print(" -", t)

    print("\n===== OBJECTS DETECTED =====")
    for obj in result.get("objects", []):
        print(" -", obj)

    print("\n===== ACTIONS DETECTED =====")
    for act in result.get("actions", []):
        print(" -", act)

    print("\n===== KEYWORDS =====")
    print(result["keywords"])

    print("\n===== CAPTIONS =====")
    print(result["captions"])

    print("\n===== HASHTAGS =====")
    print(result["platform_hashtags"])


if __name__ == "__main__":
    asyncio.run(run_test())
