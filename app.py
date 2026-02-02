import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# ALL INDIAN STATES (CAPITAL COORDINATES)
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
# FETCH LIVE WEATHER (HOURLY)
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
    return df.dropna()

# -------------------------------------------------
# WATER YIELD FORMULA
# UNIT: Litres / m² / day
# -------------------------------------------------
def calculate_water_yield(df):
    df["water_yield"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# TRAIN MODEL (ONCE)
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
st.subheader("State-wise Atmospheric Water Yield Prediction")

st.write(
    "This system predicts **how much water can be collected from air** "
    "in **each Indian state**, using live weather data."
)

if st.button("Predict Water Yield for All States"):

    results = []

    for state, (lat, lon) in STATES.items():
        df = calculate_water_yield(fetch_weather(lat, lon))
        X_live = df[["temperature", "humidity", "dew_point", "pressure"]]

        prediction = model.predict(X_live)[-1]

        results.append({
            "State": state,
            "Water Yield (Litres / m² / day)": round(prediction, 3)
        })

    result_df = pd.DataFrame(results)

    # ---------------- TABLE ----------------
    st.subheader("📋 State-wise Water Yield (Numerical)")
    st.dataframe(result_df, use_container_width=True)

    # ---------------- GRAPH ----------------
    st.subheader("📊 State-wise Water Yield Comparison")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        result_df["State"],
        result_df["Water Yield (Litres / m² / day)"],
        color="skyblue"
    )

    ax.set_xlabel("Water Yield (Litres per m² per day)")
    ax.set_ylabel("Indian States")
    ax.set_title("Comparison of Atmospheric Water Availability Across Indian States")

    st.pyplot(fig)

    st.caption(
        "X-axis shows water that can be collected from air "
        "(Litres per square meter per day). "
        "Each bar represents one Indian state."
    )
