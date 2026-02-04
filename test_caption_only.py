import time
import asyncio
from services.post_generator_service import generate_caption_post

TRANSCRIPT = """
PASTE THE SAME 477-CHAR TRANSCRIPT HERE
"""

async def run_caption_test():
    start = time.time()

    result = await generate_caption_post(
        effective_query=TRANSCRIPT,
        seed_keywords=["news", "health"],
        platforms=["instagram"]
    )

    end = time.time()

    caption = result["captions"]["instagram"]

    print("====== CAPTION ONLY TEST ======")
    print(f"Caption length: {len(caption)} characters")
    print(f"Caption generation time: {end - start:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(run_caption_test())
