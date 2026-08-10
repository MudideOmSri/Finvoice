import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

company_files = [
    "TCS.csv", "ITC.csv", "RELIANCE.csv", 
    "INFOSYS.csv", "HDFC_Bank.csv", "HINDUSTAN.csv"
]

if not os.path.exists("models"):
    os.makedirs("models")

for file in company_files:
    print(f"🔄 Training model for {file}...")

    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()  # Remove column name spaces
        df['close'] = df['close'].astype(str).str.replace(",", "").astype(float)
        data = df['close'].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X_train = []
        y_train = []
        for i in range(60, len(data_scaled)):
            X_train.append(data_scaled[i-60:i])
            y_train.append(data_scaled[i])
        X_train, y_train = np.array(X_train), np.array(y_train)

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

        model_name = os.path.splitext(file)[0]
        model.save(f"models/{model_name}_model.h5")
        joblib.dump(scaler, f"models/{model_name}_scaler.save")

        print(f"✅ Model for {file} trained and saved.")
    
    except Exception as e:
        print(f"❌ Error training model for {file}: {e}")
