"""
Factory: .env'deki LLM_PROVIDER değerine göre doğru adapter'ı üretir.
Yeni sağlayıcı eklemek = buraya bir dal eklemek. Başka hiçbir dosya değişmez.
"""

import os

from .base import LLMClient


def _require_env(name: str) -> str:
    """Zorunlu ortam değişkenini okur; yoksa açık bir hata verir (sessizce None dönmez)."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} .env dosyasında tanımlı değil.")
    return value


def create_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()

    if provider == "gemini":
        # Lazy import: Gemini SDK'sı yalnızca Gemini seçilirse yüklenir.
        # Groq kullanan biri google-genai kurmak zorunda kalmaz (bağımlılık izolasyonu).
        from .gemini_client import GeminiClient
        return GeminiClient(model=model or _require_env("GEMINI_MODEL"))

    if provider == "groq":
        # Lazy import: groq paketi yalnızca Groq seçilirse yüklenir.
        from .groq_client import GroqClient
        return GroqClient(model=model or _require_env("GROQ_MODEL"))
    raise ValueError(f"Desteklenmeyen sağlayıcı: '{provider}'")