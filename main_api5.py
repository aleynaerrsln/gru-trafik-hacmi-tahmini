from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import joblib
import random

# fast api uygulamasini baslat
app = FastAPI(title = "GRU Trafik Tahmin API")

class InputData(BaseModel):  # 24 saatlik veriden olusan giriş dizisi
    # her zaman adımı bir liste: [temp, rain_1h, snow_1h, clouds_all, hout, dayofweek, month]

    sequence: list  # iç içe list 24 x 7

# gru model sınıfı:
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out

 # modeli ve scaler ı yukle
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model.load_state_dict(torch.load("gru_model.pth"))
model.eval()  # değerlendirme modunda çalıştır

scaler_X = joblib.load("scaler_X.save")  # giriş özellikleri scaler (model eğitimde 0-1 arası girdi gördü)
scaler_y = joblib.load("scaler_y.save")  # trafic volume scaler

# API Endpoint: /predict
# POST isteği alır, tahmin sonucunu dönen
@app.post("/predict")
def predict(data: InputData):
    """
    24 saatlik giriş disizi üzerinden trafik hacmi tahmini yapar
    """
    # girdi dizisini numpy array'e çevir
    input_seq = np.array(data.sequence).astype(np.float32)

    # girdi boyutunu kontrol et
    if input_seq.shape != (24, 7):
      return {"error": "Veri boyutu hatalı, beklenen boyut (24,7)"}

    # eğitimdeki gibi 0-1 aralığına ölçekle (preprocessing2.py'deki scaler_X ile aynı dönüşüm)
    input_seq = scaler_X.transform(input_seq).astype(np.float32)

    # pytorch tensör dönüşümü
    input_tensor = torch.tensor(input_seq).unsqueeze(0)  # shape (1, 24, 7)

    # tahmin uret
    with torch.no_grad():  # gradyanların hesaplaması yapılmasın
      prediction = model(input_tensor)
      prediction = prediction.numpy()  # numpy array e çevir

    # tahmini normalize skaladan orijinal skalaya dönüştür
    prediction_orig = scaler_y.inverse_transform(prediction)

    predicted_value = float(prediction_orig[0][0])

    return {"tahmin_edilen_trafik_hacmi": predicted_value}

#test generator
sample_sequence = []
for i in range(24):
  temp = random.uniform(280,300)  #kelvin cinsinden sıcaklık
  rain = 0
  snow = 0
  clouds = random.randint(0, 100)
  hour = i
  dayofweek = random.randint(0,6)
  month = 10
  sample_sequence.append((temp, rain, snow, clouds, hour, dayofweek, month))

print(sample_sequence) 