"""
Başlangıç: 181 token
Tur başına artış: ≈ 252 token
Limit: 1.048.576 token

a) 100. turda girdi kaç olur?
b) Kaçıncı turda limite ulaşırsın?
c) 100 tur boyunca gönderilen toplam girdi token’ı ne kadar?

Ardından şu üç soruyu cevapla:

Limit aşılınca ne olur?
Limitten önce de bir sorun var mı?
Bunu çözmek için aklına gelen bir yöntem ne olurdu?
"""

first = 181          # 1. turdaki girdi
step = 252           # Her turda eklenen miktar (önceki cevap + yeni soru)
limit = 1_048_576    # Context window

# a) 100. turdaki girdi: başlangıç + 99 kez eklenen artış
turn_100 = first + (100 - 1) * step
print("a)", turn_100)
"""25.129"""

# b) Limite kadar kaç tur: (limit - başlangıç) / artış, aşağı yuvarla, +1 (1. tur da sayılır)
max_turn = (limit - first) // step + 1
print("b)", max_turn)
"""4.161	Limite ulaşmak için ~4 bin tur gerekiyor. Normal bir sohbette bu sınıra pek varılmaz"""

# c) 100 tur boyunca gönderilen TOPLAM token: her turun girdisini tek tek topla
total = sum(first + (n - 1) * step for n in range(1, 101))
print("c)", total)
"""1.265.500""" """Hiçbir istek 25 bin token’ı geçmedi,
ama 100 turda toplam 1,26 milyon token harcandı.
Limitin kendisinden bile fazla. Sebep, ilk mesajların her turda tekrar tekrar gönderilmesi
: 1. mesajın token’larını 100 kez ödüyorsun."""

"""1. Limit aşılınca ne olur?"""
"""Context window (1.048.576) -- Tek bir isteğin boyutunu-- İstek reddedilir, API hata döndürür. O istek sadece gönderilemez"""

"""2. Limitten önce sorun var mı?"""
"""Evet, üç tane:
Maliyet: c şıkkında gördüğün gibi toplam token birikerek artıyor. Ücretsiz katmanda kotan beklediğinden çok daha hızlı dolar, ücretli katmanda faturan artar.
Gecikme: Model her turda daha fazla token okuyor, bu yüzden cevap süresi uzuyor.
Kalite: Teorik eğitimdeki Attention mekanizmasıyla doğrudan ilişkili. Her token diğer tüm token’larla ilişkilendiriliyor. Bağlam büyüdükçe modelin “dikkati” dağılıyor ve konuşmanın ortasındaki bilgileri gözden kaçırma eğilimi artıyor.
"""

"""
3. Çözüm yöntemleri
Sliding window: Sadece son N mesajı gönder, eskileri at (system mesajı her zaman kalır).
Özetleme: Eski mesajları modele özetlet, özeti tek bir mesaj olarak tut.
Token bütçesi: Geçmişi mesaj sayısına göre değil, token sayısına göre kırp.
"""