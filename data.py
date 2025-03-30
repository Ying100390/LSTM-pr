import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 設定亂數種子
np.random.seed(42)

# --- 1️⃣ 產生虛擬氣象資料 ---
days = 365
temperature = 30 + 5 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 1, days)
humidity = 50 + 10 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 2, days)
wind_speed = 10 + 3 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 1, days)

data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed
})

# 顯示前幾筆資料
print("📋 原始資料（前 5 筆）：")
print(data.head())

# --- 2️⃣ 標準化（Normalization 0~1） ---
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

print("\n✅ 資料已標準化（0~1）")
print("📋 前 5 筆正規化後的資料：")
print(data_scaled[:5])

# --- 3️⃣ 切分訓練 / 測試資料 + 建立序列 ---
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 7  # 用前 7 天預測第 8 天
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# --- 顯示結果資訊 ---
print(f"\n✅ 序列資料準備完成：")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")

# --- 畫圖看原始資料 ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

data.plot(figsize=(12, 5), title="一年氣象資料變化")
plt.grid(True)
plt.show()
