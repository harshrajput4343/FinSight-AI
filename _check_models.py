import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ API Key not found!")
    exit(1)

print(f"🔑 Using API Key: {api_key[:5]}...")

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

if response.status_code == 200:
    models = response.json().get("data", [])
    free_models = [m['id'] for m in models if ":free" in m['id']]
    print("\n✅ AVAILABLE FREE MODELS:")
    for model in free_models:
        print(f" - {model}")
else:
    print(f"❌ Error fetching models: {response.status_code} - {response.text}")
