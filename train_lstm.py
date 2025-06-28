import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# === Configuration ===
READ_API_KEY = '60SQCX95B7XKZN2E'
READ_CHANNEL_ID = '2692605'
WRITE_API_KEY = 'IXTWUKUQDPJ5KIOH'
SEQ_LENGTH = 15

# === Step 1: Fetch last 40 temperature readings ===
url = f"https://api.thingspeak.com/channels/{READ_CHANNEL_ID}/fields/1.json?api_key={READ_API_KEY}&results=40"
response = requests.get(url)
data = response.json()
temps = [float(entry['field1']) for entry in data['feeds'] if entry['field1'] is not None]

if len(temps) < 25:
    raise ValueError("❌ Not enough data to train.")

# === Step 2: Basic Cleaning ===
temps = np.array(temps)
temps = temps[temps > 0]               # Remove 0s
temps = temps[~np.isnan(temps)]        # Remove NaN

# === Step 3: Remove Outliers (Z-score filtering) ===
z = np.abs((temps - np.mean(temps)) / np.std(temps))
temps_clean = temps[z < 2]             # Keep within ±2 std

if len(temps_clean) < SEQ_LENGTH + 1:
    raise ValueError("❌ Not enough clean data after removing outliers.")

# === Step 4: Normalize ===
scaler = MinMaxScaler()
temps_scaled = scaler.fit_transform(temps_clean.reshape(-1, 1))

# === Step 5: Prepare Sequences ===
X = []
for i in range(len(temps_scaled) - SEQ_LENGTH):
    X.append(temps_scaled[i:i+SEQ_LENGTH])
X = np.array(X)

y = temps_scaled[SEQ_LENGTH:]

# === Step 6: Define and Train Model ===
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# === Step 7: Predict the next time step ===
last_seq = temps_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
pred_scaled = model.predict(last_seq)
pred = scaler.inverse_transform(pred_scaled)[0][0]

# === Step 8: Anomaly Detection ===
last_actual = scaler.inverse_transform([temps_scaled[-1]])[0][0]
residual = abs(pred - last_actual)
res_mean = np.mean(np.abs(scaler.inverse_transform(y) - scaler.inverse_transform(model.predict(X)).flatten()))
res_std = np.std(np.abs(scaler.inverse_transform(y) - scaler.inverse_transform(model.predict(X)).flatten()))
z_score = (residual - res_mean) / res_std
is_anomaly = int(abs(z_score) > 2)

# === Step 9: Send Prediction and Anomaly Flag to ThingSpeak ===
url = f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}&field1={pred:.2f}&field2={is_anomaly}"
resp = requests.get(url)
print("✅ LSTM Prediction: {:.2f} | Anomaly: {} | HTTP Status: {}".format(pred, is_anomaly, resp.status_code))
