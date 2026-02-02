import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import date, timedelta

# -------------------------------------------------
# STATE COORDINATES
# -------------------------------------------------
STATES = {
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Karnataka (Bengaluru)": (12.9716, 77.5946),
    "Maharashtra (Mumbai)": (19.0760, 72.8777)
}

# -------------------------------------------------
# REAL PAST DATA (ARCHIVE API)
# -------------------------------------------------
def fetch_past_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }
    data = requests.get(url, params=params).json()

    return pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "dew_point": data["hourly"]["dewpoint_2m"],
        "pressure": data["hourly"]["surface_pressure"]
    })

# -------------------------------------------------
# FUTURE DATA (FORECAST API)
# -------------------------------------------------
def fetch_future_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }
    data = requests.get(url, params=params).json()

    return pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "dew_point": data["hourly"]["dewpoint_2m"],
        "pressure": data["hourly"]["surface_pressure"]
    })

# -------------------------------------------------
# WATER YIELD FORMULA
# UNIT: Litres / m² / day
# -------------------------------------------------
def add_water_yield(df):
    df["water_yield"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# TRAIN MODEL ON REAL PAST DATA
# -------------------------------------------------
@st.cache_resource
def train_model():
    lat, lon = STATES["Telangana (Hyderabad)"]
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)

    df = add_water_yield(fetch_past_weather(lat, lon, start, end))

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
# UI
# -------------------------------------------------
st.title("🌊 AquaGenesis – REAL DATA VERSION")
st.subheader("Past – Present – Future Atmospheric Water Yield")

state = st.selectbox("Select State", list(STATES.keys()))

if st.button("Analyze with Real Data"):

    lat, lon = STATES[state]

    # REAL PAST (LAST 7 DAYS)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    past_df = add_water_yield(fetch_past_weather(lat, lon, start, end))

    # PRESENT = LAST HOUR OF PAST
    present_value = past_df["water_yield"].iloc[-1]

    # REAL FUTURE
    future_df = add_water_yield(fetch_future_weather(lat, lon).head(24))
    future_value = model.predict(
        future_df[["temperature", "humidity", "dew_point", "pressure"]]
    )

    # ---------------- PAST GRAPH ----------------
    st.subheader("📊 Past (Last 7 Days – REAL Historical Data)")
    fig1, ax1 = plt.subplots()
    ax1.plot(past_df["time"], past_df["water_yield"])
    ax1.set_xlabel("Date & Time (Past)")
    ax1.set_ylabel("Water Yield (Litres per m² per day)")
    ax1.set_title("REAL Past Water Availability")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    st.pyplot(fig1)

    # ---------------- PRESENT ----------------
    st.metric(
        "Present Water Yield (Latest Hour)",
        f"{round(present_value,3)} Litres / m² / day"
    )

    # ---------------- FUTURE GRAPH ----------------
    st.subheader("🔮 Future (Next 24 Hours – Forecast)")
    fig2, ax2 = plt.subplots()
    ax2.plot(future_df["time"], future_value, linestyle="--")
    ax2.set_xlabel("Date & Time (Future)")
    ax2.set_ylabel("Expected Water Yield (Litres per m² per day)")
    ax2.set_title("Future Water Availability (Forecast)")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True)
    st.pyplot(fig2)
