# stelle_backend/config.py
import os
import random
import logging
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Assumes the .env file is present in the application root and is loaded.
load_dotenv() 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("stelle_backend")

# --- Database & External Keys (Reading from .env) ---
# Note: The database key has conflicting database names in your .env (stelleDB vs. stelle_db). 
# We prioritize the final line with 'stelle_db'.
MONGO_URI = os.getenv("MONGO_URI") 
if not MONGO_URI:
    logger.error("MONGO_URI environment variable is missing.")

# VAPID Keys
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = {"sub": os.getenv("FROM_EMAIL", "mailto:info@stelle.world")}

# SMTP Configuration
SMTP_CONFIG = {
    "server": "smtpout.secureserver.net", # Hostinger SMTP server (inferred)
    "port": int(os.getenv("SMTP_PORT", 465)),
    "username": os.getenv("SMTP_USERNAME"),
    "password": os.getenv("SMTP_PASSWORD"),
    "from_email": os.getenv("FROM_EMAIL"),
}

# Pexels Key
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Redis Configuration (Optional/Unused in core logic but read for completeness)
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")


# --- Groq API Key Management (Mapping specific roles to .env keys) ---

# Helper function to get multiple keys, prioritizing the specific ones
def get_groq_keys(prefix: str) -> list[str]:
    """Retrieves all Groq API keys starting with a given prefix from environment."""
    keys = [
        value
        for key, value in os.environ.items()
        if key.startswith(prefix) and value
    ]
    return keys

# BASE_GROQ_KEY is the general key used as fallback
BASE_GROQ_KEY = os.getenv("GROQ_API_KEY") 

# Specific clients mapped to specific .env keys (or falling back to BASE_GROQ_KEY)
INTERNET_CLIENT_KEY = os.getenv("GROQ_API_KEY_BROWSE", BASE_GROQ_KEY)
DEEPSEARCH_CLIENT_KEY = os.getenv("GROQ_API_KEY_DEEPSEARCH", BASE_GROQ_KEY)
ASYNC_CLIENT_KEY = BASE_GROQ_KEY 

# Specialized Keys
CONTENT_CLIENT_KEY = os.getenv("GROQ_API_KEY_CONTENT", BASE_GROQ_KEY)
EXPLANATION_CLIENT_KEY = os.getenv("GROQ_API_KEY_EXPLANATION", BASE_GROQ_KEY)
CLASSIFY_CLIENT_KEY = os.getenv("GROQ_API_KEY_CLASSIFY", BASE_GROQ_KEY)
BROWSE_ENDPOINT_KEY = os.getenv("GROQ_API_KEY_BROWSE_ENDPOINT", BASE_GROQ_KEY)
MEMORY_SUMMARY_KEY = os.getenv("GROQ_API_KEY_MEMORY_SUMMARY", BASE_GROQ_KEY)
PLANNING_KEY = os.getenv("GROQ_API_KEY_PLANNING", BASE_GROQ_KEY)
GOAL_SETTING_KEY = os.getenv("GROQ_API_KEY_GOAL_SETTING", BASE_GROQ_KEY)
# In config.py (Final version used)
# ...
GROQ_API_KEY_STELLE_MODEL = os.getenv("GROQ_API_KEY_STELLE_MODEL", BASE_GROQ_KEY) 
# ...

# Keys for Randomized Generation (List)
GENERATION_KEY_PREFIX = "GROQ_API_KEY_GENERATE_"
GENERATION_KEYS_LIST = get_groq_keys(GENERATION_KEY_PREFIX)
# Fallback to BASE_GROQ_KEY if no specific numbered keys are found
GENERATE_API_KEYS = GENERATION_KEYS_LIST or [BASE_GROQ_KEY] 

# Task Scheduler Key (Synchronous service)
GROQ_API_KEY_STELLE_MODEL = os.getenv("GROQ_API_KEY_STELLE_MODEL", BASE_GROQ_KEY)

# Non-Groq Key (OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- Rate Limiting Configuration ---
CALLS_PER_MINUTE = 50
PERIOD = 60 # seconds

# Other Global Configurations
FAISS_EMBEDDING_DIM = 768