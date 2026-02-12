import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------------------------------
# STATES (28)
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
# FETCH WEATHER
# -------------------------------------------------
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
    d = requests.get(url, params=params).json()

    df = pd.DataFrame({
        "time": pd.to_datetime(d["hourly"]["time"]),
        "temperature": d["hourly"]["temperature_2m"],
        "humidity": d["hourly"]["relative_humidity_2m"],
        "dew_point": d["hourly"]["dewpoint_2m"],
        "pressure": d["hourly"]["surface_pressure"]
    })
    return df

# -------------------------------------------------
# WATER YIELD FORMULA (BASE)
# -------------------------------------------------
def add_water_yield(df):
    df["water_yield"] = (df["humidity"]/100) * (df["temperature"] - df["dew_point"]) * 0.1
    return df

# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
def create_features(df):
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month

    for lag in range(1, 7):
        df[f"lag_{lag}"] = df["water_yield"].shift(lag)

    df = df.dropna()
    return df

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("🌊 AquaGenesis – Real ML Forecasting System")

state = st.selectbox("Select Indian State", list(STATES.keys()))

if st.button("Run Real ML Forecast"):

    lat, lon = STATES[state]

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=90)

    df = fetch_weather(lat, lon, start, end)
    df = add_water_yield(df)
    df = create_features(df)

    # Predict 6 hours ahead
    forecast_horizon = 6
    df["target"] = df["water_yield"].shift(-forecast_horizon)
    df = df.dropna()

    features = [
        "temperature","humidity","dew_point","pressure",
        "hour","day","month",
        "lag_1","lag_2","lag_3","lag_4","lag_5","lag_6"
    ]

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.metric("R² Score (6h Forecast)", round(r2,4))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.set_title("Actual vs Predicted Water Yield (6h Ahead)")
    ax.legend()
    st.pyplot(fig)

    # ----------- TRUE FUTURE FORECAST -----------
    latest_data = df.iloc[-1:][features]
    future_prediction = model.predict(latest_data)[0]

    st.subheader("🔮 Next 6 Hour Forecast")
    st.metric("Predicted Water Yield After 6 Hours (L/m²)", round(future_prediction,3))
