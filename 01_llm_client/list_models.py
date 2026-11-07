"""Anahtarınla erişebildiğin ve metin üretebilen Gemini modellerini listeler."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()  # .env dosyasını bulana kadar üst klasörlere doğru arar
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for model in client.models.list():
    # Yalnızca metin üretebilen (generateContent destekleyen) modelleri göster
    if "generateContent" in (model.supported_actions or []):
        print(model.name)