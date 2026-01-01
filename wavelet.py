import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')  # Bunu ekle
import matplotlib.pyplot as plt

# --- Wavelet Decomposition Function ---
def wavelet_decompose(data, wavelet='db4', level=2):
    """
    Decomposes data with wavelet
    level=2: only 2 levels (a2, d2, d1) - suitable for 60 months data
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs  # Returns in [a2, d2, d1] format

def wavelet_reconstruct(coeffs, wavelet='db4'):
    """
    Reconstructs wavelet components
    """
    return pywt.waverec(coeffs, wavelet)

def denoise_signal(data, wavelet='db4', level=2):
    """
    Cleans noise: by zeroing d1
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Zero d1 (highest frequency noise)
    coeffs[-1] = np.zeros_like(coeffs[-1])
    # Reconstruct
    denoised = pywt.waverec(coeffs, wavelet)
    return denoised[:len(data)]  # Return in original length

def plot_wavelet_decomposition(prices, wavelet='db4', level=2):
    """
    Visualizes wavelet decomposition (a1, a2, d1, d2 graphs)
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    
    # Calculate number of plots
    n_plots = 1 + len(coeffs)  # Original + all components
    
    # Create large figure
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots))
    fig.suptitle('Wavelet Decomposition Analysis (Daubechies-4)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Original signal
    axes[0].plot(prices, 'b-', linewidth=2, label='Original Signal')
    axes[0].set_title('Original Silver Prices', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    
    # Approximation and Detail components
    labels = [f'a{level}'] + [f'd{level-i}' for i in range(level)]
    colors = ['green'] + ['red'] * level
    
    for idx, (coeff, label, color) in enumerate(zip(coeffs, labels, colors), 1):
        axes[idx].plot(coeff, color=color, linewidth=2, label=label)
        
        if label.startswith('a'):
            axes[idx].set_title(f'{label} - Approximation (Trend)', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value', fontsize=10)
        else:
            axes[idx].set_title(f'{label} - Detail (Fluctuation)', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value', fontsize=10)
            
            # Special note for d1
            if label == 'd1':
                axes[idx].text(0.02, 0.95, '⚠️ This component is discarded (noise)', 
                             transform=axes[idx].transAxes, fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
                             verticalalignment='top')
        
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc='upper left')
        axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Index', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('wavelet_decomposition.png', dpi=150, bbox_inches='tight')
    print("\n✓ Wavelet decomposition graph saved: wavelet_decomposition.png")
    plt.show()
    
    return coeffs

# --- Load data ---
df = pd.read_csv("silver_last_60_months_fixed.csv")
prices = df['Price'].values

print("="*60)
print("WAVELET-LSTM HYBRID MODEL")
print("="*60)
print(f"Total data: {len(prices)} months")

# --- Draw wavelet graphs ---
print("\n" + "="*60)
print("WAVELET DECOMPOSITION GRAPHS")
print("="*60)
wavelet_coeffs = plot_wavelet_decomposition(prices, wavelet='db4', level=2)

# --- Noise cleaning with Wavelet ---
print("\nCleaning noise with Wavelet...")
denoised_prices = denoise_signal(prices, wavelet='db4', level=2)

print(f"Original price average: ${prices.mean():.2f}")
print(f"Denoised price average: ${denoised_prices.mean():.2f}")

# --- Create lag features (with denoised data) ---
def create_lag_features(data, n_lags=5):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_lag_features(denoised_prices, n_lags=5)
y = y.reshape(-1, 1)

print(f"\nLag features created: {X.shape[0]} samples")

# --- Normalize data ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
)

# Reshape for LSTM: (batch_size, sequence_length, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Convert to Torch tensors
X_train_tensor = torch.FloatTensor(X_train_reshaped)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_reshaped)
y_test_tensor = torch.FloatTensor(y_test)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# --- LSTM Model (same architecture) ---
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

# Create model
model = WaveletLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training ---
epochs = 2000
losses = []

print("\n" + "="*60)
print("WAVELET-LSTM TRAINING STARTING...")
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
print("TEST RESULTS")
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
    
    print("\nFirst 10 Predictions:")
    print("-" * 60)
    for i in range(min(10, len(y_test_actual))):
        error = y_pred[i][0] - y_test_actual[i][0]
        error_pct = (error / y_test_actual[i][0]) * 100
        print(f"Actual: ${y_test_actual[i][0]:6.2f} | Prediction: ${y_pred[i][0]:6.2f} | "
              f"Error: ${error:+6.2f} ({error_pct:+5.2f}%)")

# --- PREDICTION FUNCTION ---
def tahmin_yap_wavelet(fiyat1, fiyat2, fiyat3, fiyat4, fiyat5):
    """
    Predicts with 5 months price data
    NOTE: These prices should come from denoised data!
    """
    model.eval()
    with torch.no_grad():
        # Normalize
        scaled = scaler_X.transform([[fiyat1, fiyat2, fiyat3, fiyat4, fiyat5]])
        
        # Convert to 3D for LSTM
        scaled_reshaped = scaled.reshape(1, 5, 1)
        
        # Make prediction
        pred = model(torch.FloatTensor(scaled_reshaped))
        result = scaler_y.inverse_transform(pred.numpy())[0][0]
        
        return result

# --- Save model ---
torch.save(model.state_dict(), 'silver_wavelet_lstm_model.pth')
print("\n" + "="*60)
print("WAVELET-LSTM MODEL SAVED!")
print("="*60)

# --- Original vs Denoised data comparison ---
def plot_original_vs_denoised(original, denoised):
    """
    Compares original and denoised data
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Both data together
    axes[0].plot(original, 'b-', label='Original Data', linewidth=2, alpha=0.7)
    axes[0].plot(denoised, 'r-', label='Denoised Data (d1 discarded)', linewidth=2)
    axes[0].set_title('Original vs Denoised Silver Prices', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price ($)', fontsize=11)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Difference (noise)
    noise = original - denoised[:len(original)]
    axes[1].plot(noise, 'purple', linewidth=1.5, label='Discarded Noise (d1)')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Cleaned Noise Component', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Month', fontsize=11)
    axes[1].set_ylabel('Noise ($)', fontsize=11)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wavelet_denoising.png', dpi=150, bbox_inches='tight')
    print("\n✓ Noise cleaning graph saved: wavelet_denoising.png")
    plt.show()

print("\n" + "="*60)
print("ORIGINAL vs DENOISED DATA COMPARISON")
print("="*60)
plot_original_vs_denoised(prices, denoised_prices)

# --- Comparison information ---
print("\n" + "="*60)
print("WAVELET ADVANTAGES")
print("="*60)
print("✓ Noise cleaned (d1 component removed)")
print("✓ Trend appears clearer")
print("✓ Model learns more stably")
print("✓ Overfitting risk reduced")
print("\nNOTE: Compare this model with your old model (main.py)!")
print("     You can see which one is better by comparing Test MAE values.")
print("="*60)