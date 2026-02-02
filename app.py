vimport streamlit as st
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
# LOCATION
# -------------------------------------------------
LOCATIONS = {
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
}

# -------------------------------------------------
# FETCH WEATHER
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

    return df.dropna().reset_index(drop=True)

# -------------------------------------------------
# WATER FROM AIR (SIMPLE MEANING)
# -------------------------------------------------
def add_water_yield(df):
    df["water_from_air"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# LSTM DATA
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
    lat, lon = LOCATIONS["Hyderabad"]
    df = add_water_yield(fetch_weather(lat, lon))

    X = df[["temperature", "humidity", "dew_point", "pressure"]]
    y = df["water_from_air"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    xgb = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5
    )
    xgb.fit(X_train, y_train)

    X_seq, y_seq = create_sequences(X.values, y.values)

    lstm = Sequential([
        LSTM(32, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mae")
    lstm.fit(X_seq, y_seq, epochs=10, verbose=0)

    return xgb, lstm

xgb_model, lstm_model = train_models()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌊 AquaGenesis")
st.subheader("Simple Explanation: Time vs Water from Air")

city = st.selectbox("Choose City", list(LOCATIONS.keys()))

if st.button("Show Water Information"):

    lat, lon = LOCATIONS[city]
    df = add_water_yield(fetch_weather(lat, lon))

    # ---------------- PRESENT ----------------
    X_live = df[["temperature", "humidity", "dew_point", "pressure"]]
    present_xgb = xgb_model.predict(X_live)[-1]
    seq = X_live.values[-24:].reshape(1, 24, 4)
    present_lstm = lstm_model.predict(seq)[0][0]
    present_value = (present_xgb + present_lstm) / 2

    # ---------------- FUTURE ----------------
    future_values = [present_value for _ in range(6)]

    # =================================================
    # GRAPH 1: PAST
    # =================================================
    st.subheader("1️⃣ PAST: Earlier Hours")

    fig1, ax1 = plt.subplots()
    ax1.plot(
        range(1, len(df)+1),
        df["water_from_air"],
        marker="o"
    )

    ax1.set_xlabel("Time Moving Forward (Earlier Hours →)")
    ax1.set_ylabel("Water from Air (Higher = More Water)")
    ax1.set_title("How Water from Air Changed Earlier")

    st.pyplot(fig1)

    st.write(
        "👉 When the line goes UP, more water was available. "
        "When it goes DOWN, less water was available."
    )

    # =================================================
    # GRAPH 2: PRESENT
    # =================================================
    st.subheader("2️⃣ PRESENT: Right Now")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Now"], [present_value], color="green")

    ax2.set_ylabel("Water from Air (Higher = More Water)")
    ax2.set_title("Water Available from Air Right Now")

    st.pyplot(fig2)

    st.write(
        "👉 This bar shows how much water can be collected from air at this moment."
    )

    # =================================================
    # GRAPH 3: FUTURE
    # =================================================
    st.subheader("3️⃣ FUTURE: Next Few Hours")

    fig3, ax3 = plt.subplots()
    ax3.plot(
        range(1, 7),
        future_values,
        linestyle="--",
        marker="o"
    )

    ax3.set_xlabel("Future Time (Next Hours →)")
    ax3.set_ylabel("Expected Water from Air")
    ax3.set_title("Expected Water from Air in Coming Hours")

    st.pyplot(fig3)

    st.write(
        "👉 This line shows how water availability is expected to be in the next hours."
    )
