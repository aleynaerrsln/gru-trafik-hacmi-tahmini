# veri ön işleme , veriyi modelin anlayacağı biçime getirme
import pandas as pd  # veri yükleme ve ön işleme
import numpy as np  # sayısal işlemler için
from sklearn.preprocessing import MinMaxScaler  # veriyi normalize etmek
import joblib  # scaler objesini kaydetmek

# csv dosyasını oku
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# date_time string i datetime objesine çevir
df["date_time"] = pd.to_datetime(df["date_time"])

# datetime i index yap
df.set_index("date_time", inplace=True)

# zaman temelli öznitelikler
df["hour"] = df.index.hour  # saat bilgisi ,  0-23
df["dayofweek"] = df.index.dayofweek  # 0 = pazartesi, 1 = salı, ..., 6 = pazar
df["month"] = df.index.month  # ay bilgisi [1-12]

# modelde kullanilacak olan giriş be hedef degerleri
# features: Modele verilecek 7 bilgi: sıcaklık, yağmur, kar, bulutluluk, saat, gün ve ay
features = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "dayofweek", "month"]  # input features (girdi öznitelikleri)
target = "traffic_volume"  # hedef değişken , Tahmin edilecek değer

#df[...]: Sadece bu 8 sütunu tutar. holiday ve weather_main gibi yazı sütunları atılıyor, çünkü model yazı değil sayı ister.
df = df[features + [target]].dropna()  # gerekli sütunları al ve eksik değer içeren satırları sil
print(df.head())

# normalizasyon (0-1) arasına ölçekle - Farklı büyüklükteki sayıları aynı aralığa getirmek.
#Scaler-ölçekleyici -Normalizasyonu yapan araç.
#Scaled -Ölçeklenmiş, yani dönüştürülmüş veri
scaler_X = MinMaxScaler()  # giriş özellikleri için scaler oluştur
scaler_y = MinMaxScaler()  # hedef değişken için scaler oluştur

#fit: Her sütunun en küçük ve en büyük değerini öğrenir.
#transform: Bu değerlere göre dönüştürür. 
#fit_transform ikisini birden yapar.
#Target (hedef)-Modelin tahmin etmeye çalıştığı şey.-traffic_volume

#- X_scaled: 7 girdi sütununun 0-1 aralığındaki hâli.
#- y_scaled: Trafiğin 0-1 aralığındaki hâli.

X_scaled = scaler_X.fit_transform(df[features])  # giriş verini normalize et
y_scaled = scaler_y.fit_transform(df[[target]])  # hedef değişkeni normalize et

# scaler objelerini daha sonra model tahmininde kullanmak üzere kaydet
joblib.dump(scaler_X, "scaler_X.save")
joblib.dump(scaler_y, "scaler_y.save")

# zaman serisi için sqeuence (dizi) oluşturma
#(X, y, seq_length): Fonksiyonun dışarıdan aldığı 3 bilgi:
  # X: Girdiler (7 sütun, 0-1 arası)
  # y: Hedef (trafik, 0-1 arası)
  # seq_length: Pencere uzunluğu (24)
def create_sequences(X, y, seq_length):  # pencereleme
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):  
        X_seq.append(X[i:i+seq_length])  # giriş olarak ardışık seq_length kadar veriyi aldık ,X[i:i+24]: i. satırdan başlayıp 24 satır alır. Buna dilimleme (slicing) denir.
        y_seq.append(y[i+seq_length])  # bu pencerenin hemen sonrasındaki değeri hedef olarak ekliyoruz
    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 24  # sequence uzunluğu yani 24 saat
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

# eğitim ve test split
split_idx = int(0.8 * len(X_seq))

# eğitim verisi
X_train = X_seq[:split_idx] # baştan 38.544'e kadar  → %80
y_train = y_seq[:split_idx]  # 38.544'ten sonuna kadar → %20

# test verisi
X_test = X_seq[split_idx:]
y_test = y_seq[split_idx:]

# save
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Kayıt Tamamlandı")