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
@router.post("/send-otp")
async def send_otp_endpoint(request: OTPRequest, background_tasks: BackgroundTasks):
    email = request.email
    otp = generate_otp()
    
    # Store OTP asynchronously
    await store_otp(email, otp)
    
    # Send email in a background task (recommended for speed)
    background_tasks.add_task(send_email, email, otp)
    
    logger.info(f"OTP generated and scheduled for email: {email}")
    return JSONResponse(content={"message": "OTP sent", "success": True}, status_code=200)

# --- 2. Verify OTP Endpoint ---
@router.post("/verify-otp")
async def verify_otp_endpoint(request: VerifyOTPRequest):
    email = request.email
    otp = request.otp
    
    # Verification logic handles HTTPException if failed
    await verify_otp_and_delete(email, otp)
    
    logger.info(f"OTP verified successfully for email: {email}")
    return JSONResponse(content={"message": "OTP verified", "success": True}, status_code=200)

# --- 3. Register User Endpoint (New Feature) ---

@router.post("/register_user", response_model=Dict[str, Any])
async def register_user_endpoint(request: RegisterUserRequest):
    """Handles new user registration and stores credentials (simulated)."""
    
    # 1. Check for existing user
    existing_user = await users_collection.find_one({"email": request.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already registered.")

    # 2. Generate secure password hash
    # Ensure bcrypt is installed (pip install bcrypt)
    hashed_password_bytes = bcrypt.hashpw(
        request.password.encode('utf-8'), bcrypt.gensalt()
    )

    # 3. Create the user document and save to MongoDB
    new_user_id = str(uuid.uuid4())
    user_doc = {
        "user_id": new_user_id,
        "email": request.email,
        "username": request.username,
        "password_hash": hashed_password_bytes.decode('utf-8'),
        "created_at": datetime.datetime.now(datetime.timezone.utc),
    }
    
    # Save user to the collection (users_collection points to the WebPush/Users collection)
    await users_collection.insert_one(user_doc)
    
    logger.info(f"User registered: {request.email} (ID: {new_user_id})")
    
    return {
        "status": "success", 
        "message": "User registered successfully.", 
        "user_id": new_user_id
    }