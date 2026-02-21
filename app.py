import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- CUSTOM CSS ----------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
body {
    background-color: #F4F7FB;
}

.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #2E2E2E;
}

.subtitle {
    font-size: 18px;
    color: #5A5A5A;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(135deg, #3A7BD5, #00C9A7);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- STATES ----------------
STATES = {
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Maharashtra (Mumbai)": (19.0760, 72.8777),
    "Karnataka (Bengaluru)": (12.9716, 77.5946)
}

# ---------------- FETCH ----------------
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
    }).dropna()

    df["water_yield"] = (
        (df["humidity"]/100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )

    return df

# ---------------- TRAIN FAST MODEL ----------------
@st.cache_resource
def train_models():

    all_data = []
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=60)

    for state, (lat, lon) in STATES.items():
        df = fetch_weather(lat, lon, start, end)
        all_data.append(df)

    full_df = pd.concat(all_data)

    X = full_df[["temperature","humidity","dew_point","pressure"]]
    y = full_df["water_yield"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    xgb = XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=5)
    xgb.fit(X_train, y_train)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df[["water_yield"]])

    window = 24
    X_lstm, y_lstm = [], []

    for i in range(window, len(scaled)):
        X_lstm.append(scaled[i-window:i])
        y_lstm.append(scaled[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(window,1)))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_lstm, y_lstm, epochs=2, batch_size=128, verbose=0)

    return xgb, lstm, scaler

xgb, lstm, scaler = train_models()

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🌊 AquaGenesis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Atmospheric Water Decision Support System</div>', unsafe_allow_html=True)

st.markdown("---")

state = st.selectbox("Select State", list(STATES.keys()))

if st.button("🚀 Run Smart Analysis"):

    lat, lon = STATES[state]

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)

    past_df = fetch_weather(lat, lon, start, end)

    # ----------- PAST GRAPH CARD -----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Last 7 Days Water Availability")

    fig1, ax1 = plt.subplots()
    ax1.plot(past_df["time"], past_df["water_yield"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Water Yield (L/m²/day)")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------- FUTURE PREDICTION -----------
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

    xgb_pred = xgb.predict(future_df)

    scaled_input = scaler.transform(xgb_pred.reshape(-1,1))
    lstm_input = scaled_input.reshape(1,24,1)
    lstm_pred_scaled = lstm.predict(lstm_input, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)[0][0]

    hybrid_pred = (np.mean(xgb_pred) + lstm_pred) / 2

    # ----------- METRIC CARDS -----------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f'<div class="metric-card">💧 Hybrid Water Estimate<br><br>{round(hybrid_pred,3)} L/m²/day</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-card">⏰ Next Hour Prediction<br><br>{round(lstm_pred,3)} L/m²/day</div>',
            unsafe_allow_html=True
        )

    # ----------- FUTURE GRAPH CARD -----------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔮 Next 24 Hours Prediction")

    fig2, ax2 = plt.subplots()
    ax2.plot(range(1,25), xgb_pred)
    ax2.set_xlabel("Hours from Now")
    ax2.set_ylabel("Predicted Water Yield (L/m²/day)")
    st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)
