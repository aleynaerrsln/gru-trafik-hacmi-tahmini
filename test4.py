import numpy as np
import torch
import torch.nn as nn  # pytorch sinir agi modelleri
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error # değerlendirme metrikleri

# GRU model sınıfı tanımlama

class GRUNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,  output_size)

    def forward(self, x):
        # gru ya giriş (x) (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out

# hiperparametreler
SEQ_LEN = 24  # son 24 saatlik veri uzerinde tahmin yapma
INPUT_SIZE = 7  # girdi vektöründeki öznitelik sayısı
HIDDEN_SIZE = 64  # GRU hücresindeki gizli katman boyutu
NUM_LAYERS = 2  # GRU katman sayisi
OUTPUT_SIZE = 1  # çıktı: tek bir trafik hacmi değeri

#test verisini yükle
X_test = np.load("X_test.npy")  # girdi: (örnek_sayisi, 24, 7)
y_test = np.load("y_test.npy")  # hedef: (örnek_sayisi, 1) 1 -> traffic volume

#numpy tp torch
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# modeli oluştur ve ağırlıkları yükle
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# modelin içine  eğitilmiş ağırlıkları .pth dan yükle
model.load_state_dict(torch.load("gru_model.pth"))
model.eval() #modeli test moduna ya da değerlendirme moduna al

#model tahmini
with torch.no_grad(): #tahmin sırasında gradyan hesabı yapılmaz ve daha hızlı çalışacaktır
   predictions = model(X_test_tensor) #tahmin 

predictions = predictions.numpy()  #pytorch tensörünü numpy arraye çevir

#normalizasyon yapmıştık o yüzden 0-1 arası değer alıcak onu geri alıyoruz
# normalizasyonu geri al (inverse transform)
scaler_y = joblib.load("scaler_y.save")

# normalize edilmiş tahminleri orijinal skalaya geri döndür
predictions_orig = scaler_y.inverse_transform(predictions)

# aynı şekilde gerçek y test değerlerini de geri döndürelim
y_test_orig = scaler_y.inverse_transform(y_test)

#değerlendirme metrikleri
rmse = root_mean_squared_error(y_test_orig, predictions_orig) #rmse (root mean squared error) ortalama hata
mae = mean_absolute_error(y_test_orig, predictions_orig)

print(f"rmse: {rmse}")
print(f"mae: {mae}")

# tahmin grafiği
plt.figure()
plt.plot(y_test_orig[:200], label = "Gerçek Değerler", color = "blue")
plt.plot(predictions_orig[:200], label = "Tahminler", color = "orange")
plt.title("GRU Modeli - Trafik Hacmi Tahmini")
plt.xlabel("Zaman")
plt.ylabel("Trafik Hacmi")
plt.legend()
plt.show()