"""
Sağlayıcıdan bağımsız sohbet döngüsü.
Bu dosya Gemini'yi de Groq'u da TANIMAZ; yalnızca llm.base ve llm.factory'yi bilir.
"""

from dotenv import load_dotenv

from llm.base import LLMError, Message
from llm.factory import create_client

# System mesajı: Modelin kimliğini ve kurallarını belirler.
# Kullanıcı mesajlarından AYRI tutulur; konuşma boyunca sabit kalır.
SYSTEM_PROMPT = (
    "Sen Türkçe cevap veren bir yazılım mentorüsün. "
    "Cevapların kısa, net ve örnekli olsun."
)


def main() -> None:
    load_dotenv()
    client = create_client()
    print(f"Sağlayıcı: {client.provider_name} | Model: {client.model}")
    print("Çıkmak için 'exit', geçmişi silmek için '/reset' yaz.")

    # Konuşma geçmişi. İlk eleman her zaman system mesajı.
    history: list[Message] = [Message("system", SYSTEM_PROMPT)]

    while True:
        user_input = input("\nSen: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input == "/reset":
            history = history[:1]          # System mesajını koru, gerisini sil
            print("Geçmiş temizlendi.")
            continue

        history.append(Message("user", user_input))

        try:
            response = client.chat(history, temperature=0.7)
        except LLMError as error:
            history.pop()                  # Cevapsız kalan kullanıcı mesajını geri al
            print(f"[HATA] {error}")
            continue

        # Modelin cevabını geçmişe ekle -> bir sonraki turda model önceki konuşmayı "hatırlar"
        history.append(Message("assistant", response.text))

        print(f"\nModel: {response.text}")
        u = response.usage
        print(f"[token] girdi={u.input_tokens} çıktı={u.output_tokens} "
              f"düşünme={u.reasoning_tokens} toplam={u.total_tokens} | "
              f"{response.latency_ms:.0f} ms | bitiş={response.finish_reason}")


if __name__ == "__main__":   # Dosya doğrudan çalıştırılırsa main() başlar; import edilirse başlamaz
    main()