from groq import AsyncGroq
import random
from config import GENERATE_API_KEYS

def get_chat_client():
    return AsyncGroq(api_key=random.choice(GENERATE_API_KEYS))
