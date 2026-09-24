import numpy as np
import torch     # PyTorch: yapay sinir ağı kütüphanesi
import torch.nn as nn   # nn = neural network, katmanlar burada
import torch.optim as optim  # optimizer'lar (düzeltme yöntemleri)
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# hiperparameters - Modelin kendi öğrenmediği, senin baştan belirlediğin ayarlar
INPUT_SIZE = 7 # feature sayısı , Her saatte kaç bilgi var
HIDDEN_SIZE = 64 # GRU'nun hafıza genişliği. Her adımda aklında 64 sayı tutuyor
NUM_LAYERS = 2 # gruda ki katman sayısı
OUTPUT_SIZE = 1 # çıkış boyutu sadece traffic hacmi
BATCH_SIZE = 64 # mini batch boyutu 
LEARNING_RATE = 0.001 # öğrenme oranı
NUM_EPOCHS = 5  #Tüm eğitim verisi baştan sona kaç kez dolaşılacak

# verileri yukle (train test)
X_train = np.load("X_train.npy")  # shape: (örnek sayısı, 24, 7)
y_train = np.load("y_train.npy")  # shape: (örnek sayısı, 1) 1 = target value (traffic_volume)

# numpy dan tensor formatına çevir
X_train = torch.tensor(X_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)

# data loader
#TensorDataset: Her soruyu (X) cevabıyla (y) eşleştirir, yani "soru-cevap kartları" oluşturur.
#DataLoader: Bu kartları 64'lük paketlere böler ve modele sırayla verir.
#Batch (paket): Model 38.544 pencerenin hepsini aynı anda görmez. 64'erli paketler hâlinde görür: 38.544 / 64 ≈ 603 paket
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle = True)

#GRU modeli tanımla
class GRUNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()

        # gru katmani
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # output katmani
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x shape (batch_size, seq_len, input_size)

        out, _ = self.gru(x)  # out = (batch_size, seq_len, hidden_size)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out
#forward: "Veri geldiğinde sırayla ne yapılacak?" sorusunun cevabı. model(X) yazdığında PyTorch arka planda bu fonksiyonu çalıştırır.
#out, _ =: GRU iki şey döndürür. İkincisine ihtiyaç olmadığı için _ ile atılıyor. _ Python'da "bunu kullanmayacağım" anlamında bir alışkanlık.
#out[:, -1, :]: GRU her saat için bir çıktı üretir, yani 24 çıktı. Bize sadece sonuncusu lazım, çünkü o 24 saatin hepsini okumuş hâli.


# gru model
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# kayıp fonksiyonu ve optimizasyon
criterion = nn.MSELoss()  # ortalama kare hata
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

# eğitim döngüsü
lost_list = []  # her epoch için kayıp değerlerini tutar

for epoch in range(NUM_EPOCHS):
    model.train()  # modeli eğitim moduna al
    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        # tahmin üret
        # 1. tahmin et
        outputs = model(X_batch)

        # kayıp hesapla - 2. hatayı ölç
        loss = criterion(outputs, y_batch)

        # geri yayilim (backpropagation) - 3a. eski hesabı sil
        optimizer.zero_grad()
        loss.backward()  # gradyanları hesapla - 3b. suçluyu bul
        optimizer.step()  # parametreleri güncelle - 4. düzelt

        # toplam kaybı topla
        #Epoch: Eğitim verisinin tamamının bir kez dolaşılması. 5 epoch × 603 paket = yaklaşık 3.000 düzeltme
        epoch_loss += loss.item()

    avg_loss = epoch_loss/len(train_loader)
    lost_list.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

# kayıp grafiği
plt.figure()
plt.plot(lost_list, marker = "o")
plt.title("Eğitim Kayıp Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.grid(True)
plt.tight_layout()
plt.show()

# modeli kaydet
torch.save(model.state_dict(), "gru_model.pth")
print("Model başarıyla kaydedildi.")