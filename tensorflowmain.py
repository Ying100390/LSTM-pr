import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ========= 1️⃣ 資料準備 =========
np.random.seed(42)
days = 365
temperature = 30 + 5 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 1, days)
humidity = 50 + 10 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 2, days)
wind_speed = 10 + 3 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 1, days)

data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed
})

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 建立序列資料
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_len = 7
X, y = create_sequences(data_scaled, seq_len)

# 分割訓練 / 測試
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# ========= 2️⃣ 建立 Keras LSTM 模型 =========
model = Sequential()
model.add(LSTM(units=64, input_shape=(seq_len, 3)))  # input: 7天 x 3特徵
model.add(Dense(3))  # 預測 3 個輸出（溫度、濕度、風速）

model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# ========= 3️⃣ 訓練模型 =========
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# ========= 4️⃣ 預測與反標準化 =========
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_pred_rescaled = scaler.inverse_transform(train_pred)
y_train_rescaled = scaler.inverse_transform(y_train)
test_pred_rescaled = scaler.inverse_transform(test_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# ========= 5️⃣ 畫圖 =========
predicted_df = pd.DataFrame(test_pred_rescaled, columns=['temperature', 'humidity', 'wind_speed'])
actual_df = pd.DataFrame(y_test_rescaled, columns=['temperature', 'humidity', 'wind_speed'])

colors = ['#1f77b4', '#ff7f0e']
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Weather Prediction Using LSTM (TensorFlow)', fontsize=16, weight='bold')

labels = ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']
cols = ['temperature', 'humidity', 'wind_speed']

for i in range(3):
    axes[i].plot(actual_df[cols[i]], color=colors[0], label='Actual')
    axes[i].plot(predicted_df[cols[i]], color=colors[1], label='Predicted')
    axes[i].set_title(f'{cols[i].capitalize()} Prediction', fontsize=14, weight='bold')
    axes[i].set_ylabel(labels[i], fontsize=12)
    axes[i].legend(fontsize=10)
    axes[i].grid(alpha=0.3)

axes[2].set_xlabel('Days', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# ========= 6️⃣ 評估模型 =========

# 三個欄位對應
labels = ['temperature', 'humidity', 'wind_speed']
icons = ['🌡️', '💧', '🍃']

print("\n🔍 模型預測指標評估（每個變數）:")
for i, label in enumerate(labels):
    y_true = y_test_rescaled[:, i]
    y_pred = test_pred_rescaled[:, i]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"{icons[i]} {label.capitalize()}:")
    print(f"   ✅ RMSE  : {rmse:.3f}")
    print(f"   ✅ MAE   : {mae:.3f}")
    print(f"   ✅ R²    : {r2:.4f}")
    print(f"   ✅ MAPE  : {mape:.2f}%\n")
