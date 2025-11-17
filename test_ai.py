import requests
import json

BASE_URL = "https://your-app.onrender.com"
ENDPOINT = "/aiassist"

payload = {"query": "Hello"}
headers = {"Content-Type": "application/json"}

response = requests.post(f"{BASE_URL}{ENDPOINT}", headers=headers, data=json.dumps(payload))
print(response.json())
