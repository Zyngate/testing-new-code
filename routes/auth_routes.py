# stelle_backend/routes/auth_routes.py
import asyncio
import datetime # <-- CRITICAL FIX: Missing 'datetime' import from the original code
from typing import Dict, Any # <-- ADDED for Dict[str, Any] response model
import uuid # <-- ADDED for uuid.uuid4()
import bcrypt # <-- ADDED for password hashing placeholder

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request # Removed unused Request import, added APIRouter
from fastapi.responses import JSONResponse

# CRITICAL FIX: Ensure all imports from services and models are present
from models.common_models import OTPRequest, VerifyOTPRequest, RegisterUserRequest # <-- ADDED RegisterUserRequest
from services.otp_service import generate_otp, send_email, store_otp, verify_otp_and_delete
from database import users_collection # <-- ADDED for MongoDB access (saving new user)
from config import logger

router = APIRouter()

# --- 1. Send OTP Endpoint ---
# --- 1. Send OTP Endpoint ---
@router.post("/send-otp")
async def send_otp_endpoint(request: OTPRequest, background_tasks: BackgroundTasks):
    email = request.email.strip().lower()

    user_doc = await users_collection.find_one({"email": email})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    otp = generate_otp()
    await store_otp(email, otp, "otp")  # purpose is internal only

    background_tasks.add_task(send_email, email, otp)

    return {
        "message": "OTP sent",
        "success": True
    }



# --- 2. Verify OTP Endpoint ---
@router.post("/verify-otp")
async def verify_otp_endpoint(request: VerifyOTPRequest):
    email = request.email.strip().lower()

    await verify_otp_and_delete(
        email,
        request.otp,
        "otp"
    )

    return {
        "message": "OTP verified",
        "success": True
    }


# --- 3. Register User Endpoint (New Feature) ---

@router.post("/register_user", response_model=Dict[str, Any])
async def register_user_endpoint(request: RegisterUserRequest):

    email = request.email.strip().lower()
    existing_user = await users_collection.find_one({"email": email})

    if existing_user:
        raise HTTPException(status_code=400, detail="User already registered.")

    hashed_password_bytes = bcrypt.hashpw(
        request.password.encode('utf-8'),
        bcrypt.gensalt()
    )

    new_user_id = str(uuid.uuid4())
    user_doc = {
        "user_id": new_user_id,
        "email": email,  # ✅ FIXED
        "username": request.username,
        "password": hashed_password_bytes.decode('utf-8'),
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }

    await users_collection.insert_one(user_doc)

    logger.info(f"User registered: {email} (ID: {new_user_id})")

    return {
        "status": "success",
        "message": "User registered successfully.",
        "user_id": new_user_id
    }


# --- 4. Save OAuth Credentials Endpoint (for Meta/Instagram/Facebook) ---

@router.post("/save-oauth", response_model=Dict[str, Any])
async def save_oauth_credentials(request: Dict[str, Any]):
    """
    Save OAuth credentials for a user's connected platform.
    
    This is called AFTER the Meta OAuth flow completes.
    Frontend sends the access token and account ID obtained from Meta.
    
    Required fields:
    - userId: User's internal ID
    - platform: "instagram" or "facebook" (lowercase)
    - accessToken: Meta Graph API access token
    - accountId: Instagram Business Account ID or Facebook Page ID
    """
    from database import db
    
    user_id = request.get("userId")
    platform = request.get("platform", "").lower().strip()
    access_token = request.get("accessToken")
    account_id = request.get("accountId")
    
    # Validation
    if not all([user_id, platform, access_token, account_id]):
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: userId, platform, accessToken, accountId"
        )
    
    if platform not in ["instagram", "facebook", "youtube", "tiktok", "linkedin", "threads", "pinterest"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid platform: {platform}"
        )
    
    oauth_collection = db["oauthcredentials"]
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Upsert (update if exists, insert if not)
    result = await oauth_collection.update_one(
        {"userId": user_id, "platform": platform},
        {
            "$set": {
                "accessToken": access_token,
                "accountId": account_id,
                "updatedAt": now
            },
            "$setOnInsert": {
                "userId": user_id,
                "platform": platform,
                "createdAt": now
            }
        },
        upsert=True
    )
    
    if result.upserted_id:
        logger.info(f"✅ New OAuth credentials saved: {user_id} on {platform}")
        message = "OAuth credentials saved successfully"
    else:
        logger.info(f"✅ OAuth credentials updated: {user_id} on {platform}")
        message = "OAuth credentials updated successfully"
    
    return {
        "success": True,
        "message": message,
        "platform": platform
    }


# --- 5. Get Connected Platforms for a User ---

@router.get("/connected-platforms/{user_id}", response_model=Dict[str, Any])
async def get_connected_platforms(user_id: str):
    """
    Get list of platforms a user has connected via OAuth.
    
    Returns which platforms have valid credentials stored.
    """
    from database import db
    
    oauth_collection = db["oauthcredentials"]
    
    # Find all connected platforms for this user
    credentials = await oauth_collection.find(
        {"userId": user_id},
        {"platform": 1, "accountId": 1, "updatedAt": 1, "_id": 0}  # Don't expose accessToken!
    ).to_list(length=20)
    
    platforms = []
    for cred in credentials:
        platforms.append({
            "platform": cred.get("platform"),
            "accountId": cred.get("accountId"),
            "connectedAt": cred.get("updatedAt", cred.get("createdAt"))
        })
    
    return {
        "success": True,
        "userId": user_id,
        "connectedPlatforms": platforms,
        "count": len(platforms)
    }


# --- 6. DEBUG: Check OAuth credentials in database ---

@router.get("/debug-oauth/{user_id}", response_model=Dict[str, Any])
async def debug_oauth_credentials(user_id: str):
    """
    DEBUG ENDPOINT: Check what OAuth credentials exist for a user.
    
    This helps diagnose why auto-refresh might not find credentials.
    Shows the actual field names and values stored in database.
    
    REMOVE THIS IN PRODUCTION!
    """
    from database import db
    
    oauth_collection = db["oauthcredentials"]
    
    # Try to find with exact userId
    exact_match = await oauth_collection.find(
        {"userId": user_id}
    ).to_list(length=20)
    
    # Also try to find with user_id (different field name)
    alt_match = await oauth_collection.find(
        {"user_id": user_id}
    ).to_list(length=20)
    
    # Get a sample document to see field structure
    sample_doc = await oauth_collection.find_one({})
    
    # Sanitize - don't expose tokens
    def sanitize(doc):
        if not doc:
            return None
        return {
            "_id": str(doc.get("_id", "")),
            "userId": doc.get("userId"),
            "user_id": doc.get("user_id"),  # Check both formats
            "platform": doc.get("platform"),
            "accountId": doc.get("accountId"),
            "has_accessToken": bool(doc.get("accessToken")),
            "all_keys": list(doc.keys())  # Show all field names
        }
    
    return {
        "query_userId": user_id,
        "exact_match_count": len(exact_match),
        "exact_matches": [sanitize(d) for d in exact_match],
        "alt_match_count": len(alt_match),
        "alt_matches": [sanitize(d) for d in alt_match],
        "sample_document_structure": sanitize(sample_doc) if sample_doc else "No documents in collection"
    }


# --- 7. DEBUG: Test Meta API Token Directly ---

@router.post("/test-meta-token", response_model=Dict[str, Any])
async def test_meta_token(request: Dict[str, Any]):
    """
    DEBUG ENDPOINT: Test if a user's Meta token is valid.
    
    This directly calls Meta API to verify the token works.
    
    REMOVE THIS IN PRODUCTION!
    """
    import httpx
    from database import db
    
    user_id = request.get("userId")
    platform = request.get("platform", "instagram").lower()
    
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    
    oauth_collection = db["oauthcredentials"]
    
    # Find OAuth credentials (case-insensitive platform match)
    auth = await oauth_collection.find_one({
        "userId": user_id,
        "platform": {"$regex": f"^{platform}$", "$options": "i"}
    })
    
    if not auth:
        return {
            "success": False,
            "error": "No OAuth credentials found",
            "userId": user_id,
            "platform": platform
        }
    
    access_token = auth.get("accessToken", "")
    account_id = auth.get("accountId", "")
    
    # Debug info about the token
    token_info = {
        "token_length": len(access_token),
        "token_first_20": access_token[:20] if access_token else None,
        "token_last_20": access_token[-20:] if access_token else None,
        "token_has_spaces": " " in access_token if access_token else False,
        "token_has_newlines": "\n" in access_token or "\r" in access_token if access_token else False,
        "account_id": account_id,
    }
    
    # Try to call Meta API
    try:
        base_url = "https://graph.facebook.com/v18.0"
        
        # First, test with /me endpoint (simpler)
        test_url = f"{base_url}/me"
        params = {
            "access_token": access_token,
            "fields": "id,name"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(test_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": "Token is VALID!",
                    "meta_response": data,
                    "token_info": token_info
                }
            else:
                error_data = response.json()
                return {
                    "success": False,
                    "message": "Token is INVALID",
                    "http_status": response.status_code,
                    "meta_error": error_data,
                    "token_info": token_info,
                    "debug": {
                        "url_used": test_url,
                        "suggestion": "Token may be expired, revoked, or malformed"
                    }
                }
                
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "token_info": token_info
        }
