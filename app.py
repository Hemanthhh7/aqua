import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------------------------
# ALL 28 STATES
# -------------------------------------------------
STATES = {
    "Andhra Pradesh (Amaravati)": (16.5730, 80.3575),
    "Arunachal Pradesh (Itanagar)": (27.0844, 93.6053),
    "Assam (Dispur)": (26.1408, 91.7900),
    "Bihar (Patna)": (25.5941, 85.1376),
    "Chhattisgarh (Raipur)": (21.2514, 81.6296),
    "Goa (Panaji)": (15.4909, 73.8278),
    "Gujarat (Gandhinagar)": (23.2156, 72.6369),
    "Haryana (Chandigarh)": (30.7333, 76.7794),
    "Himachal Pradesh (Shimla)": (31.1048, 77.1734),
    "Jharkhand (Ranchi)": (23.3441, 85.3096),
    "Karnataka (Bengaluru)": (12.9716, 77.5946),
    "Kerala (Thiruvananthapuram)": (8.5241, 76.9366),
    "Madhya Pradesh (Bhopal)": (23.2599, 77.4126),
    "Maharashtra (Mumbai)": (19.0760, 72.8777),
    "Manipur (Imphal)": (24.8170, 93.9368),
    "Meghalaya (Shillong)": (25.5788, 91.8933),
    "Mizoram (Aizawl)": (23.7271, 92.7176),
    "Nagaland (Kohima)": (25.6751, 94.1086),
    "Odisha (Bhubaneswar)": (20.2961, 85.8245),
    "Punjab (Chandigarh)": (30.7333, 76.7794),
    "Rajasthan (Jaipur)": (26.9124, 75.7873),
    "Sikkim (Gangtok)": (27.3389, 88.6065),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tripura (Agartala)": (23.8315, 91.2868),
    "Uttar Pradesh (Lucknow)": (26.8467, 80.9462),
    "Uttarakhand (Dehradun)": (30.3165, 78.0322),
    "West Bengal (Kolkata)": (22.5726, 88.3639)
}

# -------------------------------------------------
# FETCH DATA
# -------------------------------------------------
def fetch_weather(lat, lon, start, end):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }
    d = requests.get(url, params=params).json()

    df = pd.DataFrame({
        "time": pd.to_datetime(d["hourly"]["time"]),
        "temperature": d["hourly"]["temperature_2m"],
        "humidity": d["hourly"]["relative_humidity_2m"],
        "dew_point": d["hourly"]["dewpoint_2m"],
        "pressure": d["hourly"]["surface_pressure"]
    })

    return df.dropna()

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def preprocess(df):
    df["water_yield"] = (
        (df["humidity"]/100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# TRAIN ALL STATES
# -------------------------------------------------
@st.cache_resource
def train_models():

    all_data = []
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=365)

    for state, (lat, lon) in STATES.items():
        try:
            df = fetch_weather(lat, lon, start, end)
            df = preprocess(df)
            all_data.append(df)
        except:
            continue

    full_df = pd.concat(all_data)

    X = full_df[["temperature","humidity","dew_point","pressure"]]
    y = full_df["water_yield"]

    # ---------------- XGBOOST ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
    xgb.fit(X_train, y_train)

    # ---------------- LSTM ----------------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df[["water_yield"]])

    X_lstm = []
    y_lstm = []

    window = 24
    for i in range(window, len(scaled)):
        X_lstm.append(scaled[i-window:i])
        y_lstm.append(scaled[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm = Sequential()
    lstm.add(LSTM(50, return_sequences=False, input_shape=(window,1)))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')

    lstm.fit(X_lstm, y_lstm, epochs=5, batch_size=64, verbose=0)

    return xgb, lstm, scaler

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🌊 AquaGenesis – XGBoost + LSTM Hybrid (All India Model)")

state = st.selectbox("Select State", list(STATES.keys()))

if st.button("Run Hybrid Prediction"):

    xgb, lstm, scaler = train_models()

    lat, lon = STATES[state]

    # Future forecast
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }

    d = requests.get(forecast_url, params=params).json()

    future_df = pd.DataFrame({
        "temperature": d["hourly"]["temperature_2m"],
        "humidity": d["hourly"]["relative_humidity_2m"],
        "dew_point": d["hourly"]["dewpoint_2m"],
        "pressure": d["hourly"]["surface_pressure"]
    }).head(24)

    # XGBoost Prediction
    xgb_pred = xgb.predict(future_df)

    # LSTM Prediction (using last 24 predicted values)
    scaled_input = scaler.transform(xgb_pred.reshape(-1,1))
    lstm_input = scaled_input.reshape(1,24,1)
    lstm_pred_scaled = lstm.predict(lstm_input)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)[0][0]

    # Hybrid (average)
    hybrid_pred = (np.mean(xgb_pred) + lstm_pred) / 2

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.subheader("🔮 XGBoost Prediction (Next 24 hrs)")
    st.line_chart(xgb_pred)

    st.metric("LSTM Next Hour Prediction", round(lstm_pred,3))
    st.metric("Hybrid Final Water Yield", round(hybrid_pred,3))
