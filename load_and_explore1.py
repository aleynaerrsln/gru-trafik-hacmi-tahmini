"""
GRU (Gated Recurrent Unit) ile Trafik Hacmi Tahmini
yükle ve keşfet bölümü
Problem tanımı: geçmiş trafik verilerine bakarak gelecek saatlerdeki trafik hacmini tahmin etmek

Data: https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    - 2012 - 2018
    - saatlik ölçüm
    - 48000 sample
    - hedef değişken: traffic volume
        - Features (öznitelikler):
        - date_time (zaman), holiday (resmi tatil), temp (sıcaklık), rain-snow (yağmur ve kar), clouds_all (bulutluluk oranı)
        - weather_main (genel hava durumu)

Teknolojiler/Araçlar:
    - Pytorch: GRU tabanlı zaman serisi modeli
    - FastAPI: web sunucu, modelimizi rest api olarak servis etmemizi sağlar
    - Streamlit: Web tabanlı kullanıcı arayüzü oluşturma

Plan/Program:
    - veri analizi (load_and_explore1.py)
    - ön işleme (preprocessing2.py)
    - model eğitimi (train3.py)
    - Test ve değerlendirme (test4.py)
    - FastAPI servisleştirme (main_api5.py)
    - FastAPI testi (test_requests6.py)
    - Streamlit (app_streamlit7.py)
    
install libraries: freeze - pip freeze > requirements.txt
pip install pandas numpy matplotlib seaborn scikit-learn torch fastapi uvicorn streamlit

"""
import pandas as pd  # veri işleme ve analizi , Tablo (Excel benzeri) verilerle çalışmak
import numpy as np  # matematiksel işlemler
import matplotlib.pyplot as plt  # görselleştirme ,Grafik çizmek
import seaborn as sns  # gelişmiş görselleştirme

#veriyi yükleme
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv") #csv dosyasını oku
print(df.head()) #ilk 5 satırı konsola yazdır

#veri çerçevesi hakkında gelen bilgi (ka. satır , veri türleri, eksik değer durumu ...)
print(df.info())

#sütunlarda eksik değerler
#Her sütunda kaç boş (eksik) değer var? Eksik veri modeli bozabileceği için baştan bilmek gerekir.
print(df.isnull().sum())

#sayısal değişkenler için temel istatiksel özet
print(df.describe())

#zaman sütunu düzenle
#Sorun: CSV'den okunan "2012-10-02 09:00:00" Python için sadece bir yazıdır. Python bunun bir tarih olduğunu bilmez.
df["date_time"] = pd.to_datetime(df["date_time"]) #string olarak bulunan date_time sütununu datetime objesine çeviriyoruz
df.set_index("date_time", inplace=True) # date_time index olur
# set_index: Tarih sütununu tablonun satır etiketi yapar. Artık her satır bir sayıyla değil, bir zamanla anılır. Zaman serilerinde standart yöntem budur.
print(df.head())

#zaman serisi görselleştirme
#trafik hacminin zamana göre çizdirilmesi
plt.figure()  # yeni boş bir çizim alanı aç
plt.plot(df["traffic_volume"], label = "Trafik Hacmi", color = "steelblue") # trafik hacmini zamana göre çiz
plt.title("Trafik Hacmi Zaman Serisi")
plt.xlabel("Tarih")  # eksen isimleri
plt.ylabel("Trafik Hacmi")  # eksen isimleri
plt.legend() # "Trafik Hacmi" etiketini göster
plt.tight_layout() # yazılar taşmasın diye yerleşimi düzelt
plt.show() # pencereyi aç

# saatlik ortalama trafik hacmi
#df.index.hour: Her satırın tarihinden sadece saati (0-23) alır ve hour adında yeni bir sütuna yazar.
df["hour"] = df.index.hour  # saat bilgisi sütunu oluştur

# saatlere göre ortalama trafik hacmini grupla
# groupby("hour"): Satırları saatlerine göre gruplar. Saat 0'dakileri bir grupta, saat 1'dekileri başka bir grupta toplar ve böyle devam eder.
hourly_avg = df.groupby("hour")["traffic_volume"].mean()

# bar plot ile saatlik ortalamaları görselleştir (gün içi yoğunluk ortaya çıkar)
plt.figure()
sns.barplot(x = hourly_avg.index, y = hourly_avg.values, palette="viridis")
plt.title("Gün içi saatlik ortalama trafik hacmi")
plt.xlabel("Saat")
plt.ylabel("Ortalama Trafik")
plt.show()