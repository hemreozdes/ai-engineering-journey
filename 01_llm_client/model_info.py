"""Seçili Gemini modelinin token limitlerini API'den sorgular."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model_name = os.getenv("GEMINI_MODEL")
info = client.models.get(model=model_name)   # Tek bir modelin meta bilgisini getirir

print(f"Model              : {info.name}")
print(f"Girdi token limiti : {info.input_token_limit}")    # Context window: modele tek seferde gönderebileceğin en fazla token
print(f"Çıktı token limiti : {info.output_token_limit}")   # Modelin tek cevapta üretebileceği en fazla token

"""Gözlem: Girdi limiti, çıktı limitinden 16 kat büyük. Modeller çok okuyup az yazacak şekilde tasarlanmış. Bu asimetri, ileride göreceğimiz RAG (doküman verip soru sorma) mimarisinin temel dayanaklarından biri."""