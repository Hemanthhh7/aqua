import streamlit as st
import requests
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------------------------
# LOCATIONS (INDIA – CAPITAL CITIES)
# -------------------------------------------------
LOCATIONS = {
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Maharashtra (Mumbai)": (19.0760, 72.8777),
    "Kerala (Thiruvananthapuram)": (8.5241, 76.9366),
    "Rajasthan (Jaipur)": (26.9124, 75.7873),
}

# -------------------------------------------------
# FETCH LIVE WEATHER DATA
# -------------------------------------------------
def fetch_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }
    data = requests.get(url, params=params).json()

    df = pd.DataFrame({
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "dew_point": data["hourly"]["dewpoint_2m"],
        "pressure": data["hourly"]["surface_pressure"]
    })
    return df.dropna()

# -------------------------------------------------
# WATER YIELD ESTIMATION (PHYSICAL ASSUMPTION)
# -------------------------------------------------
def add_water_yield(df):
    df["water_yield"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# LSTM SEQUENCE CREATION
# -------------------------------------------------
def create_sequences(X, y, steps=24):
    Xs, ys = [], []
    for i in range(len(X) - steps):
        Xs.append(X[i:i+steps])
        ys.append(y[i+steps])
    return np.array(Xs), np.array(ys)

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
@st.cache_resource
def train_models():
    lat, lon = LOCATIONS["Telangana (Hyderabad)"]
    df = add_water_yield(fetch_weather(lat, lon))

    X = df[["temperature", "humidity", "dew_point", "pressure"]]
    y = df["water_yield"]

    # ----- XGBOOST -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    mae_xgb = mean_absolute_error(y_test, xgb.predict(X_test))

    # ----- LSTM -----
    X_seq, y_seq = create_sequences(X.values, y.values)

    lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        LSTM(32),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mae")
    lstm.fit(X_seq, y_seq, epochs=15, batch_size=16, verbose=0)

    return xgb, lstm, mae_xgb

xgb_model, lstm_model, mae_xgb = train_models()

# -------------------------------------------------
# FEASIBILITY LOGIC
# -------------------------------------------------
def feasibility(yield_value):
    if yield_value > 1.2:
        return "High Feasibility"
    elif yield_value > 0.6:
        return "Moderate Feasibility"
    else:
        return "Low Feasibility"

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config("AquaGenesis", layout="centered")

st.title("🌊 AquaGenesis")
st.subheader("AI-Based Atmospheric Water Harvesting Feasibility System")

state = st.selectbox("Select Indian State", list(LOCATIONS.keys()))

if st.button("Analyze Feasibility"):
    lat, lon = LOCATIONS[state]
    df_live = add_water_yield(fetch_weather(lat, lon))

    X_live = df_live[["temperature", "humidity", "dew_point", "pressure"]]
    xgb_pred = xgb_model.predict(X_live)[-1]

    seq = X_live.values[-24:].reshape(1, 24, 4)
    lstm_pred = lstm_model.predict(seq)[0][0]

    final_pred = (xgb_pred + lstm_pred) / 2
    status = feasibility(final_pred)

    st.success(f"Location: {state}")
    st.metric("Predicted Water Yield (L/m²/day)", round(final_pred, 3))
    st.info(f"Feasibility Status: {status}")
    st.caption(f"XGBoost MAE: {round(mae_xgb, 4)}")

    st.subheader("Hourly Water Yield Trend")
    st.line_chart(df_live["water_yield"])
