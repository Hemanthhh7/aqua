import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------------------------------
# STATE COORDINATES
# -------------------------------------------------
LOCATIONS = {
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Karnataka (Bengaluru)": (12.9716, 77.5946),
    "Maharashtra (Mumbai)": (19.0760, 72.8777),
    "Uttar Pradesh (Lucknow)": (26.8467, 80.9462),
    "West Bengal (Kolkata)": (22.5726, 88.3639)
}

# -------------------------------------------------
# FETCH LIVE WEATHER DATA (HOURLY)
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
# SIMPLE WATER YIELD FORMULA (INTERPRETABLE)
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
# TRAIN MODELS (ONCE)
# -------------------------------------------------
@st.cache_resource
def train_models():
    lat, lon = LOCATIONS["Telangana (Hyderabad)"]
    df = add_water_yield(fetch_weather(lat, lon))

    X = df[["temperature", "humidity", "dew_point", "pressure"]]
    y = df["water_yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, xgb.predict(X_test))

    # LSTM
    X_seq, y_seq = create_sequences(X.values, y.values)

    lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        LSTM(32),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mae")
    lstm.fit(X_seq, y_seq, epochs=15, batch_size=16, verbose=0)

    return xgb, lstm, mae

xgb_model, lstm_model, mae_xgb = train_models()

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="AquaGenesis", layout="centered")

st.title("🌊 AquaGenesis")
st.subheader("Yesterday – Today – Tomorrow Water from Air")

state = st.selectbox("Select State", list(LOCATIONS.keys()))

if st.button("Analyze Water Availability"):

    lat, lon = LOCATIONS[state]
    df_live = add_water_yield(fetch_weather(lat, lon))
    X_live = df_live[["temperature", "humidity", "dew_point", "pressure"]]

    # ---------------- PRESENT PREDICTION ----------------
    pred_xgb = xgb_model.predict(X_live)[-1]
    seq = X_live.values[-24:].reshape(1, 24, 4)
    pred_lstm = lstm_model.predict(seq)[0][0]
    present_pred = (pred_xgb + pred_lstm) / 2

    # ---------------- FUTURE (NEXT 24 HOURS) ----------------
    future_preds = []
    last_input = X_live.iloc[-1:].values

    for _ in range(24):
        p = xgb_model.predict(last_input)[0]
        future_preds.append(p)

    # =====================================================
    # GRAPH 1: PAST
    # =====================================================
    st.subheader("📊 Past Water Availability")

    fig1, ax1 = plt.subplots()
    ax1.plot(
        range(len(df_live)),
        df_live["water_yield"],
        color="black"
    )
    ax1.set_title("Past Water Collected from Air")
    ax1.set_xlabel("Time (Past Hours)")
    ax1.set_ylabel("Water Collected (Litres per m² per day)")
    ax1.grid(True)
    st.pyplot(fig1)

    st.caption("This graph shows how water availability from air changed in past hours.")

    # =====================================================
    # GRAPH 2: PRESENT
    # =====================================================
    st.subheader("📍 Water Availability Today")

    fig2, ax2 = plt.subplots()
    ax2.bar(
        ["Today"],
        [present_pred],
        color="green"
    )
    ax2.set_title("Water Available from Air Today")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Water Collected (Litres per m² per day)")
    ax2.grid(axis="y")
    st.pyplot(fig2)

    st.caption("This bar shows how much water can be collected from air right now.")

    # =====================================================
    # GRAPH 3: FUTURE
    # =====================================================
    st.subheader("🔮 Future Water Availability (Next 24 Hours)")

    fig3, ax3 = plt.subplots()
    ax3.plot(
        range(1, 25),
        future_preds,
        linestyle="--",
        color="blue"
    )
    ax3.set_title("Expected Water from Air in Next 24 Hours")
    ax3.set_xlabel("Future Time (Hours)")
    ax3.set_ylabel("Expected Water (Litres per m² per day)")
    ax3.grid(True)
    st.pyplot(fig3)

    st.caption("This graph predicts how water availability may change in coming hours.")

    # ---------------- SUMMARY ----------------
    st.metric("Current Water Yield (L/m²/day)", round(present_pred, 3))
    st.caption(f"Model Accuracy (XGBoost MAE): {round(mae_xgb, 4)}")
