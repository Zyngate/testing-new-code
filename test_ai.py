import asyncio
import websockets
import json

async def test_caption_only():
    uri = "wss://testing-new-code-9onw.onrender.com/wss/generate-caption"
    async with websockets.connect(uri) as ws:
        print(await ws.recv())  # connected message

        # Send platform indices (Instagram=0, X/Twitter=1)
        await ws.send("0,1")
        print(await ws.recv())

        # Send prompt
        await ws.send("Chocolate dessert recipe")

        # Receive processing messages
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            status = data.get("status")
            if status == "completed":
                print("✅ Caption generation finished!")
                print("Captions:", data.get("captions"))
                break
            elif status == "error":
                print("❌ Error:", data.get("message"))
                break
            else:
                print("Processing:", data.get("message"))

asyncio.run(test_caption_only())
