import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Son 60 ay tarih aralığı ---
end_date = datetime.today()
start_date = end_date - timedelta(days=60*30)

# --- Veri çek (SI=F önerilir) ---
symbol = "SI=F"
data = yf.download(symbol, start=start_date, end=end_date)
data.index = pd.to_datetime(data.index)

# --- Ay ay grupla ve her ayın ilk veri bulunan gününü seç ---
monthly_data = data.resample('MS').first()  # Month Start
filled_months = []

for date in monthly_data.index:
    # O ayın tüm verilerini al
    month_end = (date + pd.DateOffset(months=1)) - timedelta(days=1)
    subset = data[(data.index >= date) & (data.index <= month_end)]
    
    if len(subset) > 0:
        # İlk bulunan veriyi al
        first_date = subset.index[0].strftime('%d.%m.%Y')
        first_price = float(subset["Close"].iloc[0])
        filled_months.append([first_date, first_price])

# --- ML için veri hazırlama ---
df = pd.DataFrame(filled_months, columns=["Date", "Price"])

# Önceki 5 ayın fiyatlarını sütunlar olarak ekle
for i in range(1, 6):
    df[f'Price_Lag_{i}'] = df['Price'].shift(i)

# Hedef değişken: Sonraki ayın fiyatı
df['Target'] = df['Price'].shift(-1)

# İlk 5 satır (yeterli geçmiş yok) ve son satırı (hedef yok) temizle
df_ml = df.dropna()

# Sadece ML için gerekli sütunları seç
df_ml = df_ml[['Date', 'Price_Lag_5', 'Price_Lag_4', 'Price_Lag_3', 'Price_Lag_2', 'Price_Lag_1', 'Target']]

# --- CSV'ye yaz ---
csv_filename = "silver_ml_dataset.csv"
df_ml.to_csv(csv_filename, index=False)

print("ML Dataset:")
print(df_ml.head(10))
print(f"\nToplam {len(df_ml)} satır veri")
print("\nCSV oluşturuldu:", csv_filename)