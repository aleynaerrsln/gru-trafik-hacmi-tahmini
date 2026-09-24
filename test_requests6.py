import requests  # http isteği göndermek için
import random  # örnek veri oluşturmak için

# fast api endpoint url
API_URL = "http://127.0.0.1:8000/predict"  # gerekirse port u değiştir

# örnek giriş verisi (24 saatlik dizi)
# her bir zaman adımı için 7 özellik: [temp, rain_1h, snow_1h, clouds_all, hour, dayofweek, month]

# rastgele örnek veri üret
sample_sequence = []
for i in range(24):
    temp = random.uniform(280, 300)  # kelvin cinsinden sıcaklık
    rain = 0  # yağış miktarı
    snow = 0  # kar miktarı
    clouds = random.randint(0, 100)  # bulutluluk oranı %
    hour = i  # 0 dan 23 e kadar
    dayofweek = random.randint(0,6)  # 0 = pazartesi, 1 = salı, ..., 6 = pazar
    month = 10  # ekim ayı
    sample_sequence.append([temp, rain, snow, clouds, hour, dayofweek, month])

print(sample_sequence)

# post isteği gönder
# json formatında
response = requests.post(API_URL, json = {"sequence": sample_sequence})

if response.status_code == 200:
    result = response.json()
    print(f"Tahmin başarılı \n Tahmin edilen trafik hacmi: {result['tahmin_edilen_trafik_hacmi']}")
else:
    print("hata oluştu")
    print(f"Status code: {response.status_code}")
    print(f"Tanı: {response.text}")