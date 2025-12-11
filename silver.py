import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- 1. Son 60 ayın tarih aralığını hesapla ---
end_date = datetime.today()
start_date = end_date - timedelta(days=60*30)  # 60 ay ≈ 30 gün * 60

# --- 2. Gümüş ons verisini çek (XAGUSD=X) ---
symbol = "XAGUSD=X"

data = yf.download(symbol, start=start_date, end=end_date)

# --- 3. Tarihi datetime formatına çevir ---
data.index = pd.to_datetime(data.index)

# --- 4. Her ayın 1’ine ait verileri filtrele ---
monthly_first = data[data.index.day == 1]

# --- 5. Sadece Close fiyatını al ve tabloyu düzenle ---
result = monthly_first[['Close']]
result = result.rename(columns={"Close": "Silver_Oz_USD"})

# --- 6. CSV dosyası oluştur ---
csv_filename = "silver_last_60_months.csv"
result.to_csv(csv_filename)

print("CSV oluşturuldu:", csv_filename)
print(result)
