import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Veriyi yükle ---
df = pd.read_csv("silver_ml_dataset.csv")

X = df[['Price_Lag_5', 'Price_Lag_4', 'Price_Lag_3', 'Price_Lag_2', 'Price_Lag_1']].values
y = df['Target'].values.reshape(-1, 1)

# --- Veriyi normalize et ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM için reshape: (batch_size, sequence_length, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Torch tensorlara çevir
X_train_tensor = torch.FloatTensor(X_train_reshaped)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_reshaped)
y_test_tensor = torch.FloatTensor(y_test)

# --- LSTM Modeli ---
class SilverPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(SilverPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM katmanı
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Fully connected katmanlar
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Son zaman adımını al
        out = out[:, -1, :]
        
        # FC katmanlar
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# Model oluştur
model = SilverPriceLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Eğitim ---
epochs = 2000
losses = []

print("LSTM Eğitimi başlıyor...")
for epoch in range(epochs):
    model.train()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- Test ---
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    test_loss = criterion(y_pred_scaled, y_test_tensor)
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())
    
    mae = np.mean(np.abs(y_pred - y_test_actual))
    
    print(f'\nTest Loss (MSE): {test_loss.item():.4f}')
    print(f'Mean Absolute Error: ${mae:.2f}')
    
    print("\nİlk 5 Tahmin:")
    for i in range(min(5, len(y_test_actual))):
        print(f"Gerçek: ${y_test_actual[i][0]:.2f} | Tahmin: ${y_pred[i][0]:.2f}")

# --- TAHMİN FONKSİYONU (LSTM için düzeltilmiş) ---
def tahmin_yap(fiyat1, fiyat2, fiyat3, fiyat4, fiyat5):
    model.eval()
    with torch.no_grad():
        # Normalize et
        scaled = scaler_X.transform([[fiyat1, fiyat2, fiyat3, fiyat4, fiyat5]])
        
        # LSTM için 3D'ye çevir: (batch_size=1, sequence_length=5, features=1)
        scaled_reshaped = scaled.reshape(1, 5, 1)
        
        # Tahmin yap
        pred = model(torch.FloatTensor(scaled_reshaped))
        result = scaler_y.inverse_transform(pred.numpy())[0][0]
        
        print(f"Tahmin: ${result:.2f}")
        return result

# Tahmin yap
print("\n" + "="*50)
print("MANUEL TAHMİN")
print("="*50)

variba = tahmin_yap(36.58300, 40.59200, 47.38725, 48.65540, 56.92940)
print(f"Tahmin edilen değer: ${variba:.2f}")
print(f"Gerçek değer: $58.35")
print(f"Mutlak hata: ${np.abs(variba - 58.35):.2f}")
print("="*50)


variba = tahmin_yap(32.391998291015625,32.03200149536133,34.15800094604492,32.18899917602539,34.5629997253418)
print(f"Tahmin edilen değer: ${variba:.2f}")
print(f"Gerçek değer: $36.78")
print(f"Mutlak hata: ${np.abs(variba - 36.7869987487793):.2f}")
print("="*50)

variba = tahmin_yap(32.03200149536133,34.15800094604492,32.18899917602539,34.5629997253418,36.082000732421875)
print(f"Tahmin edilen değer: ${variba:.2f}")
print(f"Gerçek değer: $41.07")
print(f"Mutlak hata: ${np.abs(variba - 41.07099914550781):.2f}")
print("="*50)

variba = tahmin_yap(34.15800094604492,32.18899917602539,34.5629997253418,36.082000732421875,36.7869987487793)
print(f"Tahmin edilen değer: ${variba:.2f}")
print(f"Gerçek değer: $47.29")
print(f"Mutlak hata: ${np.abs(variba - 47.290000915527344):.2f}")
print("="*50)

variba = tahmin_yap(32.18899917602539,34.5629997253418,36.082000732421875,36.7869987487793,41.07099914550781)
print(f"Tahmin edilen değer: ${variba:.2f}")
print(f"Gerçek değer: $ 47.88")
print(f"Mutlak hata: ${np.abs(variba - 47.88800048828125):.2f}")
print("="*50)

variba = tahmin_yap(17.85, 16.67, 13.97, 14.96, 17.87)
print(f"Tahmin edilen değer: ${variba:.2f}")
print(f"Gerçek değer: $18.18)")
print(f"Mutlak hata: ${np.abs(variba - 18.18):.2f}")
print("="*50)


torch.save(model.state_dict(), 'silver_lstm_model.pth')
print("\nLSTM Model kaydedildi!")