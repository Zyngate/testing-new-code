# stelle_backend/database.py
import os
import faiss
import numpy as np
# Import time and datetime for the OTP index handling consistency
from datetime import datetime, timezone 
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from config import MONGO_URI, logger
import asyncio

# --- Async MongoDB Setup (for FastAPI background tasks and async endpoints) ---
def get_async_database():
    """Initializes and returns the asynchronous MongoDB client database."""
    if not MONGO_URI:
        logger.error("Cannot connect to MongoDB: MONGO_URI is missing.")
        raise ConnectionError("MONGO_URI environment variable is not set.")
    try:
        client = AsyncIOMotorClient(MONGO_URI)
        db = client["stelle_db"] 
        logger.info("Async MongoDB connection established.")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to Async MongoDB: {e}")
        raise

# --- Database Collections (Async, initialized immediately on import) ---
db = get_async_database()
chats_collection = db["chats"]
memory_collection = db["long_term_memory"]
uploads_collection = db["uploads"]
goals_collection = db["goals"]
users_collection = db["users"] 
notifications_collection = db["notifications"]
otp_collection = db["user_otps"]
weekly_plans_collection = db["weekly_plans"]
overview_week_plans_collection = db["overview_week_plans"]
calendar_events_collection = db["calendar_events"]
user_profiles_collection = db["user_profiles"]
user_post_analytics_collection = db["user_post_analytics"]  # For caching user post analytics
video_analysis_cache_collection = db["video_analysis_cache"]  # Cache for video vision/STT analysis

# --- Autonomous Engagement Collections ---
user_tone_profiles_collection = db["user_tone_profiles"]          # Tone DNA profiles per user
post_content_cache_collection = db["post_content_cache"]          # AI analysis of each published post
comment_replies_log_collection = db["comment_replies_log"]        # Audit trail of all auto-replies
engagement_settings_collection = db["engagement_settings"]        # Per-user engagement config

# ----------------------------------------------------------------------
# --- Synchronous MongoDB Setup (Lazy & Robust Initialization) ---
# ----------------------------------------------------------------------

_sync_db = None
_sync_db_initialized = False  # <-- NEW GLOBAL STATUS FLAG

def get_or_init_sync_collections():
    """
    Initializes and returns synchronous task collections only when first called.
    Uses a boolean flag to track state safely, avoiding the pymongo truth-value bug.
    """
    global _sync_db, _sync_db_initialized
    
    # CRITICAL FIX: Only attempt initialization once
    if not _sync_db_initialized: 
        _sync_db_initialized = True  # Mark as attempted
        
        if not MONGO_URI:
            logger.error("Synchronous DB initialization failed: MONGO_URI is missing.")
            return None, None
        try:
            # Initialize the synchronous connection
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # Added timeout
            # Force connection attempt immediately
            client.admin.command('ping') 
            _sync_db = client["stelle_db"]
            logger.info("Synchronous MongoDB client initialized successfully.")
        except Exception as e:
            logger.error(f"FATAL: Failed to initialize synchronous MongoDB client: {e}")
            _sync_db = None  # Ensure it is None on failure
            return None, None 
            
    # Safely return collections if initialization succeeded
    if _sync_db is not None:  # <-- CRITICAL FIX: Checks against None (not bool())
        return _sync_db["tasks"], _sync_db["blogs"]
        
    return None, None

# --- FAISS/Vector Store Setup ---
FAISS_DIM = 768
doc_index = faiss.IndexFlatL2(FAISS_DIM)
code_index = faiss.IndexFlatL2(FAISS_DIM)

user_memory_map = {}
file_doc_memory_map = {}
code_memory_map = {}

# --- Safe OTP TTL index creation (converted into an async function) ---
async def create_otp_ttl_index():
    """Creates a TTL index on the OTP collection asynchronously."""
    try:
        existing_indexes = await otp_collection.index_information()
        if "created_at_1" not in existing_indexes:
            await otp_collection.create_index("created_at", expireAfterSeconds=300)
            logger.info("OTP TTL index created.")
    except Exception as e:
        logger.warning(f"Failed to create OTP TTL index: {e}")

# Run OTP TTL index creation only when an event loop is already active.
# This avoids binding Motor internals to a short-lived loop during module import.
try:
    running_loop = asyncio.get_running_loop()
    running_loop.create_task(create_otp_ttl_index())
except RuntimeError:
    # No running loop at import time (common in scripts/tests).
    # The index can be created later when app startup initializes async context.
    pass

# --- FAISS Index Loader ---
async def load_faiss_indices():
    """Loads user memory vectors into FAISS on startup."""
    try:
        async for mem in memory_collection.find():
            if mem.get("vector"):
                vector = np.array(mem["vector"], dtype="float32").reshape(1, -1)
                idx = doc_index.ntotal
                doc_index.add(vector)
                user_memory_map[mem["user_id"]] = idx
        logger.info(f"FAISS doc_index initialized with {doc_index.ntotal} vectors from long_term_memory.")
    except Exception as e:
        logger.error(f"Error loading FAISS index from MongoDB: {e}")
