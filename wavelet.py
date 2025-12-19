import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt  # Wavelet için
import matplotlib.pyplot as plt

# --- Wavelet Ayrıştırma Fonksiyonu ---
def wavelet_decompose(data, wavelet='db4', level=2):
    """
    Veriyi wavelet ile ayrıştırır
    level=2: sadece 2 seviye (a2, d2, d1) - 60 aylık veri için uygun
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs  # [a2, d2, d1] formatında döner

def wavelet_reconstruct(coeffs, wavelet='db4'):
    """
    Wavelet bileşenlerini tekrar birleştirir
    """
    return pywt.waverec(coeffs, wavelet)

def denoise_signal(data, wavelet='db4', level=2):
    """
    Gürültüyü temizler: d1'i sıfırlayarak
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # d1 (en yüksek frekanslı gürültü) sıfırla
    coeffs[-1] = np.zeros_like(coeffs[-1])
    # Geri birleştir
    denoised = pywt.waverec(coeffs, wavelet)
    return denoised[:len(data)]  # Orijinal uzunlukta döndür

def plot_wavelet_decomposition(prices, wavelet='db4', level=2):
    """
    Wavelet ayrıştırmasını görselleştirir (a1, a2, d1, d2 grafikleri)
    """
    # Wavelet ayrıştırma
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    
    # Grafik sayısını hesapla
    n_plots = 1 + len(coeffs)  # Orijinal + tüm bileşenler
    
    # Büyük bir figure oluştur
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots))
    fig.suptitle('Wavelet Ayrıştırma Analizi (Daubechies-4)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Orijinal sinyal
    axes[0].plot(prices, 'b-', linewidth=2, label='Orijinal Sinyal')
    axes[0].set_title('Orijinal Gümüş Fiyatları', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Fiyat ($)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    
    # Approximation ve Detail bileşenleri
    labels = [f'a{level}'] + [f'd{level-i}' for i in range(level)]
    colors = ['green'] + ['red'] * level
    
    for idx, (coeff, label, color) in enumerate(zip(coeffs, labels, colors), 1):
        axes[idx].plot(coeff, color=color, linewidth=2, label=label)
        
        if label.startswith('a'):
            axes[idx].set_title(f'{label} - Approximation (Yaklaşım/Trend)', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Değer', fontsize=10)
        else:
            axes[idx].set_title(f'{label} - Detail (Detay/Dalgalanma)', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Değer', fontsize=10)
            
            # d1 için özel not
            if label == 'd1':
                axes[idx].text(0.02, 0.95, '⚠️ Bu bileşen atılıyor (gürültü)', 
                             transform=axes[idx].transAxes, fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                             verticalalignment='top')
        
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc='upper left')
        axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('İndeks', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('wavelet_decomposition.png', dpi=150, bbox_inches='tight')
    print("\n✓ Wavelet ayrıştırma grafiği kaydedildi: wavelet_decomposition.png")
    plt.show()
    
    return coeffs

# --- Veriyi yükle ---
df = pd.read_csv("silver_last_60_months_fixed.csv")
prices = df['Price'].values

print("="*60)
print("WAVELET-LSTM HİBRİT MODEL")
print("="*60)
print(f"Toplam veri: {len(prices)} ay")

# --- Wavelet grafiklerini çiz ---
print("\n" + "="*60)
print("WAVELET AYRIŞTIRMA GRAFİKLERİ")
print("="*60)
wavelet_coeffs = plot_wavelet_decomposition(prices, wavelet='db4', level=2)

# --- Wavelet ile gürültü temizleme ---
print("\nWavelet ile gürültü temizleniyor...")
denoised_prices = denoise_signal(prices, wavelet='db4', level=2)

print(f"Orijinal fiyat ortalaması: ${prices.mean():.2f}")
print(f"Temizlenmiş fiyat ortalaması: ${denoised_prices.mean():.2f}")

# --- Lag features oluştur (temizlenmiş veri ile) ---
def create_lag_features(data, n_lags=5):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_lag_features(denoised_prices, n_lags=5)
y = y.reshape(-1, 1)

print(f"\nLag features oluşturuldu: {X.shape[0]} örneklem")

# --- Veriyi normalize et ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
)

# LSTM için reshape: (batch_size, sequence_length, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Torch tensorlara çevir
X_train_tensor = torch.FloatTensor(X_train_reshaped)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_reshaped)
y_test_tensor = torch.FloatTensor(y_test)

print(f"Train set: {X_train.shape[0]} örneklem")
print(f"Test set: {X_test.shape[0]} örneklem")

# --- LSTM Modeli (aynı mimari) ---
class WaveletLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(WaveletLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# Model oluştur
model = WaveletLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Eğitim ---
epochs = 2000
losses = []

print("\n" + "="*60)
print("WAVELET-LSTM EĞİTİMİ BAŞLIYOR...")
print("="*60)

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
print("\n" + "="*60)
print("TEST SONUÇLARI")
print("="*60)

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    test_loss = criterion(y_pred_scaled, y_test_tensor)
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())
    
    mae = np.mean(np.abs(y_pred - y_test_actual))
    rmse = np.sqrt(np.mean((y_pred - y_test_actual)**2))
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    
    print(f'Test Loss (MSE): {test_loss.item():.4f}')
    print(f'Mean Absolute Error (MAE): ${mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): ${rmse:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    
    print("\nİlk 10 Tahmin:")
    print("-" * 60)
    for i in range(min(10, len(y_test_actual))):
        error = y_pred[i][0] - y_test_actual[i][0]
        error_pct = (error / y_test_actual[i][0]) * 100
        print(f"Gerçek: ${y_test_actual[i][0]:6.2f} | Tahmin: ${y_pred[i][0]:6.2f} | "
              f"Hata: ${error:+6.2f} ({error_pct:+5.2f}%)")

# --- TAHMİN FONKSİYONU ---
def tahmin_yap_wavelet(fiyat1, fiyat2, fiyat3, fiyat4, fiyat5):
    """
    5 aylık fiyat verisi ile tahmin yapar
    NOT: Bu fiyatlar temizlenmiş veriden gelmeli!
    """
    model.eval()
    with torch.no_grad():
        # Normalize et
        scaled = scaler_X.transform([[fiyat1, fiyat2, fiyat3, fiyat4, fiyat5]])
        
        # LSTM için 3D'ye çevir
        scaled_reshaped = scaled.reshape(1, 5, 1)
        
        # Tahmin yap
        pred = model(torch.FloatTensor(scaled_reshaped))
        result = scaler_y.inverse_transform(pred.numpy())[0][0]
        
        return result

# --- Model kaydet ---
torch.save(model.state_dict(), 'silver_wavelet_lstm_model.pth')
print("\n" + "="*60)
print("WAVELET-LSTM MODEL KAYDEDİLDİ!")
print("="*60)

# --- Orijinal vs Temizlenmiş veri karşılaştırması ---
def plot_original_vs_denoised(original, denoised):
    """
    Orijinal ve temizlenmiş veriyi karşılaştırır
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # İki veri birlikte
    axes[0].plot(original, 'b-', label='Orijinal Veri', linewidth=2, alpha=0.7)
    axes[0].plot(denoised, 'r-', label='Temizlenmiş Veri (d1 atıldı)', linewidth=2)
    axes[0].set_title('Orijinal vs Temizlenmiş Gümüş Fiyatları', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Fiyat ($)', fontsize=11)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Fark (gürültü)
    noise = original - denoised[:len(original)]
    axes[1].plot(noise, 'purple', linewidth=1.5, label='Atılan Gürültü (d1)')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Temizlenen Gürültü Bileşeni', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Ay', fontsize=11)
    axes[1].set_ylabel('Gürültü ($)', fontsize=11)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wavelet_denoising.png', dpi=150, bbox_inches='tight')
    print("\n✓ Gürültü temizleme grafiği kaydedildi: wavelet_denoising.png")
    plt.show()

print("\n" + "="*60)
print("ORİJİNAL vs TEMİZLENMİŞ VERİ KARŞILAŞTIRMASI")
print("="*60)
plot_original_vs_denoised(prices, denoised_prices)

# --- Karşılaştırma bilgisi ---
print("\n" + "="*60)
print("WAVELET AVANTAJLARI")
print("="*60)
print("✓ Gürültü temizlendi (d1 bileşeni kaldırıldı)")
print("✓ Trend daha net görünüyor")
print("✓ Model daha stabil öğreniyor")
print("✓ Overfitting riski azaldı")
print("\nNOT: Bu modeli eski modelinizle (main.py) karşılaştırın!")
print("     Test MAE değerlerini karşılaştırarak hangisi daha iyi görebilirsiniz.")
print("="*60)