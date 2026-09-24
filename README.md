# GRU ile Trafik Hacmi Tahmini

Geçmiş 24 saatin hava durumu ve zaman bilgisine bakarak bir sonraki saatin trafik hacmini (araç/saat) GRU ağıyla tahmin eden proje. Eğitilen model FastAPI ile REST servisi olarak sunulur, Streamlit arayüzünden kullanılır.

**Veri seti:** [UCI – Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume). ABD'de I-94 otoyolunun 2012-2018 arasındaki saatlik ölçümleri, yaklaşık 48.000 satır. `Metro_Interstate_Traffic_Volume.csv` proje klasöründe bulunur, ayrıca indirilmesi gerekmez.

**Model girdisi:** Her saat için 7 öznitelik: `temp` (sıcaklık, Kelvin), `rain_1h`, `snow_1h` (mm), `clouds_all` (bulutluluk, %), `hour` (0-23), `dayofweek` (0 = pazartesi … 6 = pazar), `month` (1-12). Hedef değişken `traffic_volume`.

## Dosyalar ve çalıştırma sırası

Dosya adlarının sonundaki numara çalıştırma sırasını gösterir.

| # | Dosya | Ne yapar | Ürettiği dosya |
|---|---|---|---|
| 1 | `load_and_explore1.py` | Veriyi okur, özet istatistikleri yazdırır, trafik hacminin zaman serisi grafiğini ve gün içi saatlik ortalama grafiğini çizer. | – |
| 2 | `preprocessing2.py` | Saat/gün/ay özniteliklerini çıkarır, veriyi 0-1 aralığına ölçekler (MinMaxScaler) ve **24 saatlik kayan pencereler** oluşturur. Verinin %80'i eğitime, %20'si teste ayrılır. | `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, `scaler_X.save`, `scaler_y.save` |
| 3 | `train3.py` | 2 katmanlı, 64 gizli birimli GRU ve 1 çıkışlı Linear katmandan oluşan modeli eğitir (PyTorch, Adam, MSE, 5 epoch). Kayıp grafiğini çizer. | `gru_model.pth` |
| 4 | `test4.py` | Modeli test verisinde çalıştırır, **RMSE** ve **MAE** değerlerini yazdırır, ilk 200 saat için gerçek ve tahmin grafiğini çizer. | – |
| 5 | `main_api5.py` | Modeli **FastAPI** servisi olarak açar. `POST /predict` uç noktası 24×7'lik diziyi alır, tahmini döndürür. | – |
| 6 | `test_requests6.py` | Rastgele 24 saatlik örnek veri üretip API'ye istek atar, sonucu yazdırır. API açıkken çalıştırılır. | – |
| 7 | `app_streamlit7.py` | **Streamlit** web arayüzü. 24 saatin değerleri formdan girilir, **Tahmin Et** butonu API'ye istek atar ve sonucu gösterir. API açıkken çalıştırılır. | – |

`sample.txt` API'ye gönderilebilecek örnek bir 24×7 girdi içerir.

Eğitilmiş model (`gru_model.pth`) ve ölçekleyiciler (`scaler_X.save`, `scaler_y.save`) klasörde hazır bulunur. Bu yüzden API ve Streamlit arayüzü (5-7. adımlar) modeli yeniden eğitmeden doğrudan çalışır.

> **Not:** `.npy` dosyaları zip'e boyutları nedeniyle eklenmemiştir. 3. ve 4. adımı çalıştırmadan önce `python preprocessing2.py` bir kez çalıştırılarak birkaç saniyede yeniden üretilir.

## Kurulum (gereklilikler)

**Gerekenler:** Python 3.13 (proje 3.13.9 ile geliştirildi) ve yaklaşık 1.5 GB boş disk alanı (PyTorch büyük bir paket).

`venv` klasörü repoya ve zip'e dahil edilmez. Proje klasöründe şu adımlarla yeniden oluşturulur:

```bash
# 1) Sanal ortam oluştur
python -m venv venv

# 2) Sanal ortamı etkinleştir
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3) Kütüphaneleri kur
pip install -r requirements.txt
```

Asıl kütüphaneler şunlardır: `torch`, `fastapi`, `uvicorn`, `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `requests`. `requirements.txt` dosyasında bunların ve bağımlılıklarının tam sürümleri bulunur.

> Windows PowerShell `activate` komutunda "running scripts is disabled" hatası verirse önce şunu çalıştırın:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

## Çalıştırma

### Model hazırlığı (isteğe bağlı)

```bash
python load_and_explore1.py
python preprocessing2.py
python train3.py
python test4.py
```

### API ve web arayüzü

İki ayrı terminal gerekir. İkisinde de sanal ortam etkin olmalıdır.

**Terminal 1 – API:**
```bash
uvicorn main_api5:app --reload
```
API `http://127.0.0.1:8000` adresinde açılır. `http://127.0.0.1:8000/docs` adresinden tarayıcıda denenebilir.

**Terminal 2 – Arayüz:**
```bash
streamlit run app_streamlit7.py
```
Tarayıcıda `http://localhost:8501` açılır.

> Streamlit ilk çalıştırmada terminalde `Email:` sorar. Boş bırakıp **Enter**'a basmak yeterlidir, ardından adres görünür.

İsteğe bağlı olarak API, arayüz olmadan da test edilebilir:
```bash
python test_requests6.py
```
