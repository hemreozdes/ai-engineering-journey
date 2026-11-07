"""Groq hesabınla erişebildiğin modelleri listeler."""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# models.list() -> .data içinde model nesneleri döner; her birinin .id alanı model adıdır
for model in client.models.list().data:
    print(model.id)