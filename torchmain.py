import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# ========= 1️⃣ 讀取 Excel 資料 =========
data = pd.read_excel("weather_data.xlsx")  # ⬅️ 確保檔名正確
print("✅ 已成功讀取資料：")
print(data.head())

# ========= 2️⃣ 正規化與序列建構 =========
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

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

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ========= 3️⃣ 模型定義 =========
class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(WeatherLSTM, self).__init__()
        ### 🔧 修改：將多層 LSTM 支援的參數加進來（num_layers）
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # ❗改進建議：如果你要加 Dropout，可寫成：
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.2, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最後一個時間步的輸出作為預測結果

# -------- 建立模型與訓練元件 --------
input_size = 3
hidden_size = 64
output_size = 3
### 🔧 修改：改為多層 LSTM 的層數
num_layers = 2

model = WeatherLSTM(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ========= 4️⃣ 訓練模型 =========
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ========= 5️⃣ 預測與反標準化 =========
model.eval()
with torch.no_grad():
    train_pred = model(X_train).numpy()
    test_pred = model(X_test).numpy()

train_pred_rescaled = scaler.inverse_transform(train_pred)
y_train_rescaled = scaler.inverse_transform(y_train.numpy())
test_pred_rescaled = scaler.inverse_transform(test_pred)
y_test_rescaled = scaler.inverse_transform(y_test.numpy())

# ========= 6️⃣ 畫圖 =========
predicted_df = pd.DataFrame(test_pred_rescaled, columns=['temperature', 'humidity', 'wind_speed'])
actual_df = pd.DataFrame(y_test_rescaled, columns=['temperature', 'humidity', 'wind_speed'])

colors = ['#1f77b4', '#ff7f0e']
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Weather Prediction Using LSTM', fontsize=16, weight='bold')

labels = ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']
cols = ['temperature', 'humidity', 'wind_speed']

for i in range(3):
    axes[i].plot(actual_df[cols[i]], color=colors[0], label='Actual')  # 實際值
    axes[i].plot(predicted_df[cols[i]], color=colors[1], label='Predicted')  # 改成實線預測值
    axes[i].set_title(f'{cols[i].capitalize()} Prediction', fontsize=14, weight='bold')
    axes[i].set_ylabel(labels[i], fontsize=12)
    axes[i].legend(fontsize=10)
    axes[i].grid(alpha=0.3)


axes[2].set_xlabel('Days', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ========= 7️⃣ 評估指標 =========
icons = ['🌡️', '💧', '🍃']
print("\n🔍 模型預測指標評估（每個變數）:")
for i, label in enumerate(cols):
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
# 這段程式碼是用來訓練一個 LSTM 模型來預測氣象資料，並且將結果可視化和評估模型的表現。
# 你可以根據需要調整模型的參數、訓練的 epochs 數量等。