import os
from dotenv import load_dotenv

load_dotenv()

print("SITE_URL:", os.getenv("SITE_URL"))
print("BRAND_NAME:", os.getenv("BRAND_NAME"))
print("WHATSAPP:", os.getenv("HUMAN_WHATSAPP_URL"))
print("GEMINI KEY existe?:", "SI 👌" if os.getenv("GEMINI_API_KEY") else "NO ❌")
