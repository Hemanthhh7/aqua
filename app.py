import streamlit as st
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AquaGenesis Intelligence", layout="wide")

# ================= SEASON CONFIG =================
SEASON_COLORS = {
    "Winter (Dec–Feb)": "#3B82F6",
    "Summer (Mar–May)": "#F97316",
    "Monsoon (Jun–Sep)": "#10B981",
    "Post-Monsoon (Oct–Nov)": "#8B5CF6"
}

SEASON_ORDER = [
    "Winter (Dec–Feb)",
    "Summer (Mar–May)",
    "Monsoon (Jun–Sep)",
    "Post-Monsoon (Oct–Nov)"
]

# ================= SIDEBAR =================
st.sidebar.title("🌊 AquaGenesis")
st.sidebar.markdown("Hybrid AI Atmospheric Water Intelligence")

STATES = {
    "Andhra Pradesh (Amaravati)": (16.5730, 80.3575),
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707)
}

state = st.sidebar.selectbox("Select State", list(STATES.keys()))
run = st.sidebar.button("Run Full Analysis")

# ================= SAFE API FUNCTION =================
def safe_api_call(url, params):
    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            st.error(f"API Error {response.status_code}")
            return None

        try:
            return response.json()
        except:
            st.error("Invalid JSON from API")
            st.text(response.text[:300])
            return None

    except Exception as e:
        st.error(f"Request failed: {e}")
        return None

# ================= FETCH WEATHER =================
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

    r = safe_api_call(url, params)

    if r is None or "hourly" not in r:
        return pd.DataFrame()

    df = pd.DataFrame({
        "time": pd.to_datetime(r["hourly"]["time"]),
        "temperature": r["hourly"]["temperature_2m"],
        "humidity": r["hourly"]["relative_humidity_2m"],
        "dew_point": r["hourly"]["dewpoint_2m"],
        "pressure": r["hourly"]["surface_pressure"]
    }).dropna()

    if df.empty:
        return df

    df["water_yield"] = (df["humidity"]/100)*(df["temperature"]-df["dew_point"])*0.1
    df["month"] = df["time"].dt.month

    df["season"] = df["month"].apply(
        lambda m: "Winter (Dec–Feb)" if m in [12,1,2] else
        "Summer (Mar–May)" if m in [3,4,5] else
        "Monsoon (Jun–Sep)" if m in [6,7,8,9] else
        "Post-Monsoon (Oct–Nov)"
    )

    return df

# ================= TRAIN =================
@st.cache_resource
def train_models():
    all_data = []
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)

    for lat, lon in STATES.values():
        df = fetch_weather(lat, lon, start, end)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        st.error("No training data available")
        st.stop()

    full_df = pd.concat(all_data)

    X = full_df[["temperature","humidity","dew_point","pressure"]]
    y = full_df["water_yield"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    xgb = XGBRegressor(n_estimators=50)
    xgb.fit(X_train, y_train)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df[["water_yield"]])

    window = 12
    X_lstm, y_lstm = [], []

    for i in range(window, len(scaled)):
        X_lstm.append(scaled[i-window:i])
        y_lstm.append(scaled[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm = Sequential()
    lstm.add(LSTM(16, input_shape=(window,1)))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_lstm, y_lstm, epochs=1, batch_size=64, verbose=0)

    return xgb, lstm, scaler

xgb, lstm, scaler = train_models()

# ================= MAIN =================
st.title("Atmospheric Water Intelligence Dashboard")

if run:

    lat, lon = STATES[state]

    # ===== Past =====
    past = fetch_weather(lat, lon, date.today()-timedelta(days=7), date.today()-timedelta(days=1))

    if past.empty:
        st.error("No past data")
        st.stop()

    present_yield = past["water_yield"].iloc[-1]
    st.metric("Current Water Yield", round(present_yield,3))

    # ===== Forecast =====
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }

    f = safe_api_call(forecast_url, params)

    if f is None or "hourly" not in f:
        st.error("Forecast API failed")
        st.stop()

    future_df = pd.DataFrame({
        "temperature": f["hourly"]["temperature_2m"],
        "humidity": f["hourly"]["relative_humidity_2m"],
        "dew_point": f["hourly"]["dewpoint_2m"],
        "pressure": f["hourly"]["surface_pressure"]
    }).head(12)

    if future_df.empty:
        st.error("No forecast data")
        st.stop()

    xgb_pred = xgb.predict(future_df)

    scaled_input = scaler.transform(xgb_pred.reshape(-1,1))
    lstm_input = scaled_input.reshape(1,12,1)
    lstm_pred = scaler.inverse_transform(lstm.predict(lstm_input))[0][0]

    hybrid_yield = (np.mean(xgb_pred)+lstm_pred)/2

    st.metric("Hybrid Prediction", round(hybrid_yield,3))
