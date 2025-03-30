# save_data_to_excel.py
import numpy as np
import pandas as pd

np.random.seed(42)  # 固定亂數種子，保證每次都一樣

days = 365
temperature = 30 + 5 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 1, days)
humidity = 50 + 10 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 2, days)
wind_speed = 10 + 3 * np.sin(np.linspace(0, 2 * np.pi, days)) + np.random.normal(0, 1, days)

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed
})

df.to_excel("weather_data.xlsx", index=False)
print("✅ 已將隨機氣象資料儲存到 weather_data.xlsx")
