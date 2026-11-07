"""
Groq adapter'ı: Ortak arayüzü Groq SDK'sının (OpenAI uyumlu format) diline çevirir.
groq paketi SADECE bu dosyada import edilir.
"""

import os

import groq
from groq import Groq

from .base import LLMClient, LLMError, LLMResponse, Message, TokenUsage


class GroqClient(LLMClient):
    provider_name = "groq"
    temperature_range = (0.0, 2.0)   # OpenAI uyumlu API'ler 0-2 aralığını kullanır

    def __init__(self, model: str, api_key: str | None = None):
        super().__init__(model)
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY bulunamadı. .env dosyasını kontrol et.")
        # DİKKAT: Groq SDK'sında timeout SANİYE cinsindendir (Gemini'de milisaniyeydi).
        self._client = Groq(api_key=key, timeout=60.0)

    def _send(self, messages: list[Message], temperature: float,
              max_tokens: int) -> LLMResponse:
        # 1) Çeviri: OpenAI formatında system mesajı listenin İÇİNDE kalır,
        #    rol adları da bizimkilerle birebir aynıdır. Bu yüzden _split_system'e gerek yok;
        #    dataclass'ı sözlüğe çevirmek yeterli.
        payload = [{"role": m.role, "content": m.content} for m in messages]

        # 2) API çağrısı + hata çevirisi
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temperature,
                max_completion_tokens=max_tokens,   # Düşünme token'ları da bu limite dahil
            )
        except groq.APIError as exc:
            # groq.APIError: SDK'nın belgelenmiş ana hata sınıfı.
            # Zaman aşımı (APITimeoutError), bağlantı (APIConnectionError) ve
            # HTTP durum hataları (RateLimitError, AuthenticationError...) bunun alt sınıfıdır.
            raise LLMError(f"Groq API hatası: {type(exc).__name__}: {exc}") from exc

        choice = response.choices[0]   # Tek cevap istedik; ilk (ve tek) seçeneği al

        # 3) Token bilgisini ortak formata çevir.
        #    MANTIKSAL TUTARLILIK: Gemini'de "çıktı" yalnızca görünen cevaptı, düşünme ayrıydı.
        #    OpenAI formatında completion_tokens düşünmeyi de İÇERİR.
        #    İki sağlayıcının "çıktı" anlamı aynı olsun diye düşünmeyi çıkarıyoruz.
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        completion = usage.completion_tokens or 0

        token_usage = TokenUsage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=completion - reasoning,
            reasoning_tokens=reasoning,
            total_tokens=usage.total_tokens or 0,
        )

        return LLMResponse(
            text=choice.message.content or "",
            provider=self.provider_name,
            model=self.model,
            usage=token_usage,
            finish_reason=choice.finish_reason,   # "stop" veya "length" (limit doldu)
        )