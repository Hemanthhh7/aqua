import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# INDIAN STATES
# -------------------------------------------------
STATES = {
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Karnataka (Bengaluru)": (12.9716, 77.5946),
    "Maharashtra (Mumbai)": (19.0760, 72.8777)
}

# -------------------------------------------------
# FETCH WEATHER WITH TIME
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
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "dew_point": data["hourly"]["dewpoint_2m"],
        "pressure": data["hourly"]["surface_pressure"]
    })

    return df.dropna()

# -------------------------------------------------
# WATER YIELD (Litres / m² / day)
# -------------------------------------------------
def calculate_water_yield(df):
    df["water_yield"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
@st.cache_resource
def train_model():
    lat, lon = STATES["Telangana (Hyderabad)"]
    df = calculate_water_yield(fetch_weather(lat, lon))

    X = df[["temperature", "humidity", "dew_point", "pressure"]]
    y = df["water_yield"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🌊 AquaGenesis")
st.subheader("Past – Present – Future Water Availability")

state = st.selectbox("Select State", list(STATES.keys()))

if st.button("Analyze"):

    lat, lon = STATES[state]
    df = calculate_water_yield(fetch_weather(lat, lon))

    # ---------------- PRESENT ----------------
    X_live = df[["temperature", "humidity", "dew_point", "pressure"]]
    present_value = model.predict(X_live)[-1]

    # ---------------- FUTURE (NEXT 6 HOURS) ----------------
    future_hours = pd.date_range(
        start=df["time"].iloc[-1] + pd.Timedelta(hours=1),
        periods=6,
        freq="H"
    )
    future_values = [present_value] * 6

    # =====================================================
    # PAST GRAPH
    # =====================================================
    st.subheader("📊 Past Water Availability")

    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(df["time"], df["water_yield"], marker="o")

    ax1.set_title(f"Past Water Availability – {state}")
    ax1.set_xlabel("Time (Date & Hour)")
    ax1.set_ylabel("Water Yield (Litres per m² per day)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)

    st.pyplot(fig1)

    st.caption(
        "X-axis shows actual date and hour. "
        "Y-axis shows how much water could be collected from air."
    )

    # =====================================================
    # PRESENT GRAPH
    # =====================================================
    st.subheader("📍 Current Water Availability")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Now"], [present_value], color="green")

    ax2.set_ylabel("Water Yield (Litres per m² per day)")
    ax2.set_title("Water Available Right Now")

    st.pyplot(fig2)

    # =====================================================
    # FUTURE GRAPH
    # =====================================================
    st.subheader("🔮 Future Water Availability (Next 6 Hours)")

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(future_hours, future_values, linestyle="--", marker="o")

    ax3.set_xlabel("Future Time (Date & Hour)")
    ax3.set_ylabel("Expected Water Yield (Litres per m² per day)")
    ax3.set_title("Expected Water Availability")
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True)

    st.pyplot(fig3)

    st.metric(
        "Current Water Yield",
        f"{round(present_value,3)} Litres / m² / day"
    )
