import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI

EMAIL_TO_TEST = "keerthi.24msd7029@vitapstudent.ac.in"


async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["stelle_db"]

    print("\n🔍 Testing email:", EMAIL_TO_TEST)

    collections_to_check = [
        "WebPush",
        "users",
        "user_profiles",
        "accounts",
        "profiles"
    ]

    found_anywhere = False

    for col_name in collections_to_check:
        collection = db[col_name]
        user = await collection.find_one({"email": EMAIL_TO_TEST})

        if user:
            found_anywhere = True
            print(f"\n✅ FOUND in collection: {col_name}")
            print("Document keys:", list(user.keys()))
        else:
            print(f"❌ Not found in collection: {col_name}")

    if not found_anywhere:
        print("\n🚨 USER NOT FOUND IN ANY COMMON COLLECTIONS")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
