import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------------------
# MULTI-LOCATION LIST (DROPDOWN)
# -----------------------------------
LOCATIONS = {
    "Hyderabad": (17.3850, 78.4867),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946)
}

# -----------------------------------
# FETCH LIVE WEATHER DATA
# -----------------------------------
def fetch_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame({
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "dew_point": data["hourly"]["dewpoint_2m"],
        "pressure": data["hourly"]["surface_pressure"]
    })

    return df.dropna()

# -----------------------------------
# WATER YIELD CALCULATION
# -----------------------------------
def add_water_yield(df):
    df["water_yield"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -----------------------------------
# TRAIN AI MODEL (RUNS ONCE)
# -----------------------------------
@st.cache_resource
def train_model():
    # Use one city to build base model
    lat, lon = LOCATIONS["Hyderabad"]

    df = fetch_weather(lat, lon)
    df = add_water_yield(df)

    X = df[["temperature", "humidity", "dew_point", "pressure"]]
    y = df["water_yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae

model, mae = train_model()

# -----------------------------------
# STREAMLIT UI
# -----------------------------------
st.set_page_config(page_title="AquaGenesis", layout="centered")

st.title("🌊 AquaGenesis")
st.subheader("AI-Based Atmospheric Water Harvesting Predictor")

st.markdown("Predict how much water can be extracted from air using **live weather data**.")

# DROPDOWN INPUT
city = st.selectbox("📍 Select Location", list(LOCATIONS.keys()))

# BUTTON
if st.button("Predict Water Yield"):
    lat, lon = LOCATIONS[city]

    df_live = fetch_weather(lat, lon)
    df_live = add_water_yield(df_live)

    X_live = df_live[["temperature", "humidity", "dew_point", "pressure"]]
    predictions = model.predict(X_live)

    latest_yield = predictions[-1]

    st.success(f"Location: {city}")
    st.metric(
        label="Predicted Water Yield (Liters / m² / day)",
        value=round(latest_yield, 3)
    )

    st.caption(f"Model MAE: {round(mae, 4)}")

    # GRAPH
    st.subheader("📈 Water Yield Trend")
    st.line_chart(df_live["water_yield"])
