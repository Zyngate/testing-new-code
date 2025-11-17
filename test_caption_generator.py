# test_caption_generator.py

import asyncio
from services.ai_service import groq_generate_text

async def test_caption():
    prompt = "Write a catchy Instagram caption about Oranges."
    system_msg = "You are a creative social media caption generator."
    
    print("🔹 Generating caption...")
    
    caption = await groq_generate_text(
        model="llama-3.1-8b-instant",
        prompt=prompt,
        system_msg=system_msg
    )
    
    if caption:
        print("✅ Caption generated successfully!")
        print("Caption:\n", caption)
    else:
        print("❌ Caption generation failed.")

if __name__ == "__main__":
    asyncio.run(test_caption())
