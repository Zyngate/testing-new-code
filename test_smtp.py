import smtplib
from email.message import EmailMessage

server = "smtpout.secureserver.net"
port = 465
username = "info@stelle.world"
password = "zyngate123"
...
msg = EmailMessage()
msg.set_content("Testing OTP mail from Stelle backend")
msg["Subject"] = "SMTP Test - Stelle"
msg["From"] = username
msg["To"] = "keerthiadapa70@gmail.com"

try:
    print(f"Connecting to {server}:{port} ...")
    with smtplib.SMTP_SSL(server, port) as smtp:
        smtp.login(username, password)
        print("✅ SMTP login successful!")
        smtp.send_message(msg)
        print("✅ Test email sent to keerthiadapa70@gmail.com")
except Exception as e:
    print("❌ Error sending email:", e)
