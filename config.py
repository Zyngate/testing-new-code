# stelle_backend/config.py
import os
import random
import logging
from dotenv import load_dotenv
import cloudinary

# --- Load Environment Variables ---
load_dotenv()  # Load .env file in project root

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("stelle_backend")

# --- Database ---
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("MONGO_URI environment variable is missing.")

GROK_REASONING = os.getenv("GROK_REASONING")
if not GROK_REASONING:
    logger.warning("GROK_REASONING key missing")


# --- VAPID Keys (WebPush) ---
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")

raw_from_email = os.getenv("FROM_EMAIL", "mailto:info@stelle.world")
if not raw_from_email.startswith("mailto:"):
    raw_from_email = f"mailto:{raw_from_email}"
FROM_EMAIL = raw_from_email
VAPID_CLAIMS = {"sub": FROM_EMAIL}

# --- SMTP Configuration ---
SMTP_CONFIG = {
    "server": "smtpout.secureserver.net",  # Hostinger SMTP inferred
    "port": int(os.getenv("SMTP_PORT", 465)),
    "username": os.getenv("SMTP_USERNAME"),
    "password": os.getenv("SMTP_PASSWORD"),
    "from_email": FROM_EMAIL,
}
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.hostinger.com")
EMAIL_PORT = int(os.getenv("SMTP_PORT", 465))
EMAIL_ADDRESS = os.getenv("SMTP_USERNAME")
EMAIL_PASSWORD = os.getenv("SMTP_PASSWORD")

# --- Pexels API ---
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# --- Cloudinary (Post Scheduler Media Storage) ---
POST_CLOUDINARY_CLOUD_NAME = os.getenv("POST_CLOUDINARY_CLOUD_NAME")
POST_CLOUDINARY_API_KEY = os.getenv("POST_CLOUDINARY_API_KEY")
POST_CLOUDINARY_API_SECRET = os.getenv("POST_CLOUDINARY_API_SECRET")

if not all([
    POST_CLOUDINARY_CLOUD_NAME,
    POST_CLOUDINARY_API_KEY,
    POST_CLOUDINARY_API_SECRET
]):
    logger.warning("⚠️ Cloudinary credentials for post scheduler are missing")

cloudinary.config(
    cloud_name=POST_CLOUDINARY_CLOUD_NAME,
    api_key=POST_CLOUDINARY_API_KEY,
    api_secret=POST_CLOUDINARY_API_SECRET,
    secure=True
)


# --- Redis Configuration ---
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# --- Groq API Key Management ---
def get_groq_keys(prefix: str) -> list[str]:
    """Get all Groq API keys from environment starting with prefix."""
    keys = [v for k, v in os.environ.items() if k.startswith(prefix) and v]
    # Sort numerically if key ends with number
    keys.sort(key=lambda k: int(''.join(filter(str.isdigit, k)) or 0))
    return keys

BASE_GROQ_KEY = os.getenv("BASE_GROQ_KEY") or os.getenv("GROQ_API_KEY")

# Specialized Keys
CONTENT_CLIENT_KEY        = os.getenv("GROQ_API_KEY_CONTENT", BASE_GROQ_KEY)
EXPLANATION_CLIENT_KEY    = os.getenv("GROQ_API_KEY_EXPLANATION", BASE_GROQ_KEY)
CLASSIFY_CLIENT_KEY       = os.getenv("GROQ_API_KEY_CLASSIFY", BASE_GROQ_KEY)
GROQ_API_KEY_CAPTION      = os.getenv("GROQ_API_KEY_CAPTION", BASE_GROQ_KEY)
GROQ_API_KEY_CODE         = os.getenv("GROQ_API_KEY_CODE", BASE_GROQ_KEY)
GROQ_API_KEY_RESEARCHAGENT= os.getenv("GROQ_API_KEY_RESEARCHAGENT", BASE_GROQ_KEY)
GROQ_API_KEY_STELLE_MODEL = os.getenv("GROQ_API_KEY_STELLE_MODEL", BASE_GROQ_KEY)
PLANNING_KEY              = os.getenv("GROQ_API_KEY_PLANNING", BASE_GROQ_KEY)
GOAL_SETTING_KEY          = os.getenv("GROQ_API_KEY_GOAL_SETTING", BASE_GROQ_KEY)
MEMORY_SUMMARY_KEY        = os.getenv("GROQ_API_KEY_MEMORY_SUMMARY", BASE_GROQ_KEY)
GROQ_API_KEY_RECOMMENDATION = os.getenv("GROQ_API_KEY_RECOMMENDATION")
# --- NEW: Video Captioning Key (STRICT — no fallback) ---
GROQ_API_KEY_VIDEO_CAPTION = os.getenv("GROQ_API_KEY_VIDEO_CAPTION")
if not GROQ_API_KEY_VIDEO_CAPTION:
    logger.warning("GROQ_API_KEY_VIDEO_CAPTION is missing! Video caption endpoints will fail.")

GROQ_API_KEY_VISUALIZE = os.getenv("GROQ_API_KEY_VISUALIZE")
if not GROQ_API_KEY_VISUALIZE:
    raise RuntimeError("GROQ_API_KEY_VISUALIZE is not set")

# Browsing / DeepSearch
INTERNET_CLIENT_KEY = os.getenv("GROQ_API_KEY_BROWSE", BASE_GROQ_KEY)
DEEPSEARCH_CLIENT_KEY = os.getenv("GROQ_API_KEY_DEEPSEARCH", BASE_GROQ_KEY)
BROWSE_ENDPOINT_KEY = os.getenv("GROQ_API_KEY_BROWSE_ENDPOINT", BASE_GROQ_KEY)
ASYNC_CLIENT_KEY = BASE_GROQ_KEY

# Randomized Generation Keys
GENERATION_KEY_PREFIX = "GROQ_API_KEY_GENERATE_"
GENERATE_API_KEYS = get_groq_keys(GENERATION_KEY_PREFIX) or [BASE_GROQ_KEY]

# --- OpenAI API Key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Rate Limiting ---
CALLS_PER_MINUTE = 50
PERIOD = 60  # seconds

# --- Other Global Configs ---
FAISS_EMBEDDING_DIM = 768

# --- Quick sanity check (optional) ---
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is missing!")
if not BASE_GROQ_KEY:
    logger.warning("BASE_GROQ_KEY / GROQ_API_KEY is missing!")
if not MONGO_URI:
    logger.warning("MONGO_URI is missing!")
