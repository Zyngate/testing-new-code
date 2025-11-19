# stelle_backend/config.py

import os
import logging
from dotenv import load_dotenv

# -------------------------------------------------------------------
# LOAD .env
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("stelle_backend")

# -------------------------------------------------------------------
# SAFE ENV GETTERS
# -------------------------------------------------------------------
def require_env(name: str) -> str:
    """Require an environment variable; log error if missing."""
    value = os.getenv(name)
    if not value or value.strip() == "":
        logger.error(f"❌ Missing environment variable: {name}")
    return value

def optional_env(name: str, default=None):
    """Optional env var; returns default if missing."""
    return os.getenv(name, default)

# -------------------------------------------------------------------
# DATABASE
# -------------------------------------------------------------------
MONGO_URI = require_env("MONGO_URI")

# -------------------------------------------------------------------
# SMTP / EMAIL
# -------------------------------------------------------------------
SMTP_USERNAME = optional_env("SMTP_USERNAME")
SMTP_PASSWORD = optional_env("SMTP_PASSWORD")
SMTP_PORT     = optional_env("SMTP_PORT", 465)

EMAIL_HOST = optional_env("EMAIL_HOST", "smtp.hostinger.com")
EMAIL_ADDRESS = SMTP_USERNAME
EMAIL_PASSWORD = SMTP_PASSWORD

raw_from_email = optional_env("FROM_EMAIL", "mailto:info@stelle.world")
if not raw_from_email.startswith("mailto:"):
    raw_from_email = f"mailto:{raw_from_email}"
FROM_EMAIL = raw_from_email
VAPID_PUBLIC_KEY = optional_env("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = optional_env("VAPID_PRIVATE_KEY")
VAPID_CLAIMS = {"sub": FROM_EMAIL}

# -------------------------------------------------------------------
# REDIS
# -------------------------------------------------------------------
USE_REDIS = optional_env("USE_REDIS", "false").lower() == "true"
REDIS_HOST = optional_env("REDIS_HOST", "localhost")

# -------------------------------------------------------------------
# GROQ API KEYS (NO FALLBACKS MIXED)
# -------------------------------------------------------------------
# These three MUST be different
GROQ_API_KEY = require_env("GROQ_API_KEY")                     # main general key
GROQ_API_KEY_CAPTION = require_env("GROQ_API_KEY_CAPTION")     # caption generation key
BASE_GROQ_KEY = require_env("BASE_GROQ_KEY")                   # async client key

# Specialized Keys (Optional)
CONTENT_CLIENT_KEY        = optional_env("GROQ_API_KEY_CONTENT", GROQ_API_KEY)
EXPLANATION_CLIENT_KEY    = optional_env("GROQ_API_KEY_EXPLANATION", GROQ_API_KEY)
CLASSIFY_CLIENT_KEY       = optional_env("GROQ_API_KEY_CLASSIFY", GROQ_API_KEY)
GROQ_API_KEY_CODE         = optional_env("GROQ_API_KEY_CODE", GROQ_API_KEY)
GROQ_API_KEY_RESEARCHAGENT= optional_env("GROQ_API_KEY_RESEARCHAGENT", GROQ_API_KEY)
GROQ_API_KEY_STELLE_MODEL = optional_env("GROQ_API_KEY_STELLE_MODEL", GROQ_API_KEY)
PLANNING_KEY              = optional_env("GROQ_API_KEY_PLANNING", GROQ_API_KEY)
GOAL_SETTING_KEY          = optional_env("GROQ_API_KEY_GOAL_SETTING", GROQ_API_KEY)
MEMORY_SUMMARY_KEY        = optional_env("GROQ_API_KEY_MEMORY_SUMMARY", GROQ_API_KEY)

# Browsing Keys
INTERNET_CLIENT_KEY = optional_env("GROQ_API_KEY_BROWSE", GROQ_API_KEY)
DEEPSEARCH_CLIENT_KEY = optional_env("GROQ_API_KEY_DEEPSEARCH", GROQ_API_KEY)
BROWSE_ENDPOINT_KEY = optional_env("GROQ_API_KEY_BROWSE_ENDPOINT", GROQ_API_KEY)

# Async operations use BASE_GROQ_KEY
ASYNC_CLIENT_KEY = BASE_GROQ_KEY

# -------------------------------------------------------------------
# MULTIPLE GENERATION KEYS
# -------------------------------------------------------------------
def collect_keys(prefix: str) -> list[str]:
    return [v for k, v in os.environ.items() if k.startswith(prefix) and v.strip()]

GENERATE_API_KEYS = collect_keys("GROQ_API_KEY_GENERATE_")
if not GENERATE_API_KEYS:
    GENERATE_API_KEYS = [GROQ_API_KEY]   # fallback pool

# -------------------------------------------------------------------
# OPENAI (Optional)
# -------------------------------------------------------------------
OPENAI_API_KEY = optional_env("OPENAI_API_KEY")

# -------------------------------------------------------------------
# GLOBAL CONFIGS
# -------------------------------------------------------------------
CALLS_PER_MINUTE = 50
PERIOD = 60
FAISS_EMBEDDING_DIM = 768

# -------------------------------------------------------------------
# WARNINGS
# -------------------------------------------------------------------
if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY missing!")
if not GROQ_API_KEY_CAPTION:
    logger.error("❌ GROQ_API_KEY_CAPTION missing!")
if not BASE_GROQ_KEY:
    logger.error("❌ BASE_GROQ_KEY missing!")
if not MONGO_URI:
    logger.error("❌ MONGO_URI missing!")
