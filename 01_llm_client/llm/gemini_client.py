"""
Gemini adapter'ı: Ortak arayüzü Google Gen AI SDK'sının diline çevirir.
google.genai SADECE bu dosyada import edilir.
"""

import os

from google import genai
from google.genai import errors, types

from .base import LLMClient, LLMError, LLMResponse, Message, TokenUsage


class GeminiClient(LLMClient):
    provider_name = "gemini"
    temperature_range = (0.0, 2.0)   # Gemini 0-2 aralığını kabul eder

    def __init__(self, model: str, api_key: str | None = None):
        super().__init__(model)
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY bulunamadı. .env dosyasını kontrol et.")
        # SDK istemcisi bir kez oluşturulur ve her istekte tekrar kullanılır.
        # timeout milisaniye cinsinden: 60 saniyede cevap gelmezse istek iptal edilir.
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=60_000),
        )

    def _send(self, messages: list[Message], temperature: float,
              max_tokens: int) -> LLMResponse:
        # 1) System mesajını ayır: Gemini onu konuşmanın içinde değil, config'de ister.
        system_text, conversation = self._split_system(messages)

        # 2) Rol çevirisi: Bizim "assistant" -> Gemini'nin "model" rolü.
        #    Her mesaj, içinde "parts" listesi olan bir Content nesnesine dönüşür
        #    (Gemini bir mesajda metin + resim gibi birden fazla parça taşıyabildiği için).
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in conversation
        ]

        # 3) İstek ayarları
        config = types.GenerateContentConfig(
            system_instruction=system_text,   # None ise system mesajı gönderilmez
            temperature=temperature,
            max_output_tokens=max_tokens,     # Düşünen modellerde düşünme token'ları da buna dahildir
        )

        # 4) API çağrısı: Sağlayıcıya özgü hatayı ortak LLMError'a çeviriyoruz.
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        # APIError: Google'ın döndürdüğü hatalar (kota, geçersiz anahtar...)
        # httpx.HTTPError: Ağ seviyesindeki hatalar (zaman aşımı, bağlantı kopması...)
        except errors.APIError as exc:
            # Google'ın bildirdiği hatalar: kota doldu, geçersiz anahtar, model bulunamadı...
            raise LLMError(f"Gemini API hatası: {exc}") from exc
        except Exception as exc:
            # Ağ seviyesindeki hatalar: zaman aşımı, bağlantı kopması...
            # SDK'nın hangi HTTP kütüphanesini kullandığından bağımsız olarak hepsini yakalar.
            # Bu blok YALNIZCA SDK çağrısını sarıyor; bizim kodumuzdaki bug'ları gizlemez.
            raise LLMError(f"Gemini bağlantı hatası: {type(exc).__name__}: {exc}") from exc

        # 5) Token bilgisini ortak formata çevir. Alan None gelebilir; "or 0" ile güvenceye al.
        meta = response.usage_metadata
        usage = TokenUsage(
            input_tokens=(meta.prompt_token_count or 0) if meta else 0,
            output_tokens=(meta.candidates_token_count or 0) if meta else 0,
            reasoning_tokens=(meta.thoughts_token_count or 0) if meta else 0,
            total_tokens=(meta.total_token_count or 0) if meta else 0,
        )

        # 6) Bitiş sebebi (STOP, MAX_TOKENS, SAFETY ...). Enum geliyorsa adını alıyoruz.
        finish_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            reason = response.candidates[0].finish_reason
            finish_reason = getattr(reason, "name", str(reason))

        return LLMResponse(
            text=response.text or "",      # Güvenlik filtresine takılırsa text None olabilir
            provider=self.provider_name,
            model=self.model,
            usage=usage,
            finish_reason=finish_reason,
        )