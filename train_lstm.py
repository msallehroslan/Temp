import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Config
READ_API_KEY = '60SQCX95B7XKZN2E'
READ_CHANNEL_ID = '2692605'
WRITE_API_KEY = 'IXTWUKUQDPJ5KIOH'
PREDICT_LENGTH = 5
SEQ_LENGTH = 15

# Fetch last 20 temperature readings
url = f"https://api.thingspeak.com/channels/{READ_CHANNEL_ID}/fields/1.json?api_key={READ_API_KEY}&results=20"
response = requests.get(url)
data = response.json()
temps = [float(entry['field1']) for entry in data['feeds'] if entry['field1'] is not None]

if len(temps) < 20:
    raise ValueError("Not enough data to train.")

# Normalize
scaler = MinMaxScaler()
temps_scaled = scaler.fit_transform(np.array(temps).reshape(-1, 1))

# Prepare data
X, y = [], []
for i in range(len(temps_scaled) - SEQ_LENGTH - PREDICT_LENGTH + 1):
    X.append(temps_scaled[i:i+SEQ_LENGTH])
    y.append(temps_scaled[i+SEQ_LENGTH:i+SEQ_LENGTH+PREDICT_LENGTH].reshape(-1))
X, y = np.array(X), np.array(y)

# Define model
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, 1)),
    Dense(PREDICT_LENGTH)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Predict next 5 steps
last_seq = temps_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
pred_scaled = model.predict(last_seq)
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# Send to ThingSpeak
base_url = f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}"
fields = ''.join([f"&field{i+1}={pred[i]:.2f}" for i in range(PREDICT_LENGTH)])
send_url = base_url + fields
resp = requests.get(send_url)
print("Updated ThingSpeak with prediction:", resp.status_code)
