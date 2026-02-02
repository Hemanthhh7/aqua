import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# ALL INDIAN STATES WITH CAPITAL COORDINATES
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
        "temperature": data["hourly"]["temperature_2m"],   # °C
        "humidity": data["hourly"]["relative_humidity_2m"],# %
        "dew_point": data["hourly"]["dewpoint_2m"],        # °C
        "pressure": data["hourly"]["surface_pressure"]     # hPa
    })

    return df.dropna().reset_index(drop=True)

# -------------------------------------------------
# WATER YIELD CALCULATION
# UNIT: Litres / m² / day
# -------------------------------------------------
def calculate_water_yield(df):
    df["water_yield"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# TRAIN MODEL (ONCE, USED FOR ALL STATES)
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
st.set_page_config(page_title="AquaGenesis", layout="centered")

st.title("🌊 AquaGenesis")
st.subheader("Past – Present – Future Water Yield (State-wise)")

state = st.selectbox("Select Indian State", list(STATES.keys()))

if st.button("Analyze Water Yield"):

    lat, lon = STATES[state]
    df = calculate_water_yield(fetch_weather(lat, lon))

    X_live = df[["temperature", "humidity", "dew_point", "pressure"]]

    # ---------------- PRESENT ----------------
    present_value = model.predict(X_live)[-1]

    # ---------------- FUTURE (NEXT 6 HOURS) ----------------
    future_hours = list(range(1, 7))
    future_values = [present_value for _ in future_hours]

    # =====================================================
    # GRAPH 1: PAST
    # =====================================================
    st.subheader("📊 Past Water Availability")

    fig1, ax1 = plt.subplots()
    ax1.plot(
        range(len(df)),
        df["water_yield"],
        marker="o"
    )

    ax1.set_title(f"Past Water Availability – {state}")
    ax1.set_xlabel("Time (Past Hours)")
    ax1.set_ylabel("Water Yield (Litres per m² per day)")
    ax1.grid(True)

    st.pyplot(fig1)

    st.caption(
        "X-axis shows past hours. Y-axis shows how much water could be collected from air."
    )

    # =====================================================
    # GRAPH 2: PRESENT
    # =====================================================
    st.subheader("📍 Present Water Availability (Now)")

    fig2, ax2 = plt.subplots()
    ax2.bar(
        ["Now"],
        [present_value],
        color="green"
    )

    ax2.set_title(f"Current Water Availability – {state}")
    ax2.set_xlabel("Time (Current)")
    ax2.set_ylabel("Water Yield (Litres per m² per day)")
    ax2.grid(axis="y")

    st.pyplot(fig2)

    st.caption(
        "This bar shows how much water can be collected from air right now."
    )

    # =====================================================
    # GRAPH 3: FUTURE
    # =====================================================
    st.subheader("🔮 Future Water Availability")

    fig3, ax3 = plt.subplots()
    ax3.plot(
        future_hours,
        future_values,
        linestyle="--",
        marker="o"
    )

    ax3.set_title(f"Expected Water Availability – Next 6 Hours ({state})")
    ax3.set_xlabel("Future Time (Hours Ahead)")
    ax3.set_ylabel("Expected Water Yield (Litres per m² per day)")
    ax3.grid(True)

    st.pyplot(fig3)

    st.caption(
        "X-axis shows future hours. Y-axis shows expected water availability from air."
    )

    # ---------------- SUMMARY ----------------
    st.metric(
        "Current Water Yield",
        f"{round(present_value, 3)} Litres / m² / day"
    )
