import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# 28 INDIAN STATES WITH CAPITAL COORDINATES
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
# FETCH REAL PAST WEATHER (ARCHIVE API)
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
# FETCH FUTURE WEATHER (FORECAST API)
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
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "dew_point": data["hourly"]["dewpoint_2m"],
        "pressure": data["hourly"]["surface_pressure"]
    })

# -------------------------------------------------
# WATER YIELD FORMULA
# Unit: Litres / m² / day
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
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="AquaGenesis", layout="centered")

st.title("🌊 AquaGenesis")
st.subheader("State-wise Atmospheric Water Availability")

state = st.selectbox("Select Indian State", list(STATES.keys()))

if st.button("Analyze Water Availability"):

    lat, lon = STATES[state]

    # ---------- REAL PAST (LAST 7 DAYS) ----------
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    past_df = add_water_yield(fetch_past_weather(lat, lon, start, end))

    # ---------- PRESENT ----------
    present_value = past_df["water_yield"].iloc[-1]

    # ---------- FUTURE (NEXT 12 HOURS) ----------
    future_df = fetch_future_weather(lat, lon).head(12)
    future_df = add_water_yield(future_df)
    future_hours = list(range(1, 13))

    # ---------------- PAST GRAPH ----------------
    st.subheader("📊 Past Water Availability (Last 7 Days – REAL)")

    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(past_df["time"], past_df["water_yield"], color="black")
    ax1.set_xlabel("Date & Time (Past)")
    ax1.set_ylabel("Water Yield (Litres per m² per day)")
    ax1.set_title(f"Past Water Availability – {state}")
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    st.pyplot(fig1)

    # ---------------- PRESENT ----------------
    st.metric(
        "Present Water Availability (Latest Hour)",
        f"{round(present_value,3)} Litres / m² / day"
    )

    # ---------------- FUTURE GRAPH ----------------
    st.subheader("🔮 Future Water Availability (Simplified View)")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(
        future_hours,
        future_df["water_yield"],
        linestyle="--",
        marker="o",
        color="blue"
    )
    ax2.set_xlabel("Future Time (Hours Ahead)")
    ax2.set_ylabel("Expected Water Yield (Litres per m² per day)")
    ax2.set_title("Expected Water Availability in Coming Hours")
    ax2.grid(True)
    st.pyplot(fig2)

    st.caption(
        "X-axis shows how many hours ahead from now. "
        "Y-axis shows expected water that can be collected from air."
    )
