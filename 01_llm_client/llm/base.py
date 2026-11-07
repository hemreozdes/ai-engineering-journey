"""
Sağlayıcıdan bağımsız LLM katmanının sözleşmesi.

Bu dosyada HİÇBİR sağlayıcı SDK'sı import edilmez.
Gemini, Groq veya ileride eklenecek her sağlayıcı bu sınıflara uymak zorundadır.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal

# Rol tipini kısıtlıyoruz: yalnızca bu üç değer geçerli.
# Sağlayıcılar farklı adlar kullanabilir (Gemini "assistant" yerine "model" der);
# bu çeviriyi adapter yapar, dış dünya yalnızca bu üç ismi bilir.
Role = Literal["system", "user", "assistant"]


# frozen=True -> nesne oluşturulduktan sonra değiştirilemez (immutable).
# Mesaj geçmişinin yanlışlıkla değiştirilmesini engeller.
@dataclass(frozen=True)
class Message:
    """Konuşmadaki tek bir mesaj (DTO)."""
    role: Role
    content: str


@dataclass(frozen=True)
class TokenUsage:
    """Bir isteğin token tüketimi. Her sağlayıcı farklı alan adı kullanır; burada ortaklaşır."""
    input_tokens: int          # Modele gönderilen token sayısı (system + tüm geçmiş + yeni mesaj)
    output_tokens: int         # Modelin ürettiği cevap token'ları
    total_tokens: int          # Sağlayıcının raporladığı toplam (faturalandırma buna göre)
    reasoning_tokens: int = 0  # "Düşünen" modellerde cevaptan önce harcanan iç düşünme token'ları


@dataclass(frozen=True)
class LLMResponse:
    """Her sağlayıcının döndürdüğü ortak cevap formatı (DTO)."""
    text: str
    provider: str
    model: str
    usage: TokenUsage
    finish_reason: str | None  # Neden durdu? (STOP = doğal bitiş, MAX_TOKENS = limit doldu...)
    latency_ms: float = 0.0    # İsteğin kaç milisaniye sürdüğü


class LLMError(Exception):
    """
    Sağlayıcıdan gelen tüm hataların ortak tipi.
    main.py, Google veya Groq'un kendi hata sınıflarını bilmek zorunda kalmaz;
    yalnızca LLMError yakalar. (Spring'deki custom exception + @ControllerAdvice mantığı.)
    """


class LLMClient(ABC):
    """
    Tüm LLM sağlayıcılarının uyması gereken arayüz.
    ABC (Abstract Base Class) -> Bu sınıftan doğrudan nesne üretilemez; Java'daki abstract class.
    """

    provider_name: str = "base"                         # Alt sınıf kendi adını yazar
    temperature_range: tuple[float, float] = (0.0, 1.0)  # Alt sınıf kendi geçerli aralığını yazar

    def __init__(self, model: str):
        self.model = model

    def chat(self, messages: list[Message], temperature: float = 0.7,
             max_tokens: int = 2048) -> LLMResponse:
        """
        Dış dünyanın çağırdığı TEK metot. (Template Method deseni)
        Ortak işler burada yapılır: doğrulama ve süre ölçümü.
        Sağlayıcıya özel kısım _send() içindedir.
        """
        self._validate(messages, temperature)

        start = time.perf_counter()                      # Yüksek hassasiyetli zamanlayıcı
        response = self._send(messages, temperature, max_tokens)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Dataclass frozen olduğu için değiştiremeyiz; replace() yeni bir kopya üretir.
        return replace(response, latency_ms=elapsed_ms)

    @abstractmethod
    def _send(self, messages: list[Message], temperature: float,
              max_tokens: int) -> LLMResponse:
        """Her sağlayıcı bu metodu kendi SDK'sıyla uygulamak ZORUNDADIR."""

    def _validate(self, messages: list[Message], temperature: float) -> None:
        """
        İstek API'ye gitmeden önce mantıksal tutarlılığı kontrol eder.
        Hatalı isteği para ve kota harcamadan, yerelde yakalarız.
        """
        if not messages:
            raise ValueError("Mesaj listesi boş olamaz.")

        for index, message in enumerate(messages):
            if message.role == "system" and index != 0:
                raise ValueError("System mesajı yalnızca listenin en başında olabilir.")
            if not message.content.strip():
                raise ValueError(f"{index}. mesajın içeriği boş.")

        if messages[-1].role != "user":
            raise ValueError("Son mesaj 'user' rolünde olmalı; model bir soruya cevap verir.")

        low, high = self.temperature_range
        if not low <= temperature <= high:
            raise ValueError(
                f"{self.provider_name} için temperature {low}-{high} aralığında olmalı, "
                f"gelen değer: {temperature}"
            )

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """
        System mesajını konuşmadan ayırır.
        Bazı sağlayıcılar (Gemini) system'i ayrı bir parametre olarak ister,
        bazıları (Groq) listenin içinde ister. Bu yardımcı ilk gruba hizmet eder.
        """
        if messages[0].role == "system":
            return messages[0].content, messages[1:]
        return None, messages