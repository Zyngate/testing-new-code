# stelle_backend/services/otp_service.py
import secrets
import smtplib
from email.message import EmailMessage
from fastapi import HTTPException

from config import SMTP_CONFIG, logger
from database import otp_collection
from datetime import datetime, timezone

def generate_otp() -> str:
    """Generates a random 6-digit OTP."""
    return str(secrets.randbelow(900000) + 100000)

def send_email(to_email: str, otp: str):
    """Sends the OTP via email using Hostinger SMTP."""
    msg = EmailMessage()
    msg.set_content(f"Your OTP is {otp}. It is valid for 5 minutes.")
    msg["Subject"] = "Your OTP for Password Reset"
    msg["From"] = "info@stelle.world"
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtpout.secureserver.net", 465) as server:
            server.login("info@stelle.world", "zyngate123")
            server.send_message(msg)
            print(f"✅ OTP email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send OTP email: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

async def store_otp(email: str, otp: str):
    """Stores the OTP in MongoDB with a TTL index."""
    await otp_collection.insert_one(
        {
            "email": email,
            "otp": otp,
            "created_at": datetime.now(timezone.utc),
        }
    )

async def verify_otp_and_delete(email: str, otp: str) -> bool:
    """Verifies the OTP against the database and deletes it if correct."""
    stored_otp = await otp_collection.find_one({"email": email})
    
    if stored_otp is None:
        raise HTTPException(
            status_code=400, detail="No OTP found for this email or it has expired"
        )
        
    if stored_otp["otp"] != otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
        
    await otp_collection.delete_one({"email": email})
    return True