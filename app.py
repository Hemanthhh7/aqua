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
# ALL 28 INDIAN STATES (CAPITAL COORDINATES)
# -------------------------------------------------
LOCATIONS = {
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
# FETCH WEATHER DATA
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
# WATER YIELD ESTIMATION
# -------------------------------------------------
def add_water_yield(df):
    df["water_yield"] = (
        (df["humidity"] / 100) *
        (df["temperature"] - df["dew_point"]) * 0.1
    )
    return df

# -------------------------------------------------
# CREATE LSTM SEQUENCES
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, xgb.predict(X_test))

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
st.subheader("Past – Present – Future Water Yield Prediction")

state = st.selectbox("Select Indian State", list(LOCATIONS.keys()))

if st.button("Analyze"):
    lat, lon = LOCATIONS[state]
    df_live = add_water_yield(fetch_weather(lat, lon))
    X_live = df_live[["temperature", "humidity", "dew_point", "pressure"]]

    # Present prediction
    pred_xgb = xgb_model.predict(X_live)[-1]
    seq = X_live.values[-24:].reshape(1, 24, 4)
    pred_lstm = lstm_model.predict(seq)[0][0]
    present_pred = (pred_xgb + pred_lstm) / 2

    # Future prediction (next 24 hours)
    future_preds = []
    last_input = X_live.iloc[-1:].values

    for _ in range(24):
        p = xgb_model.predict(last_input)[0]
        future_preds.append(p)

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(df_live["water_yield"], label="Past (Actual)", color="black")
    ax.axhline(present_pred, color="blue", linestyle="--", label="Present Prediction")
    ax.plot(
        range(len(df_live), len(df_live) + 24),
        future_preds,
        linestyle="--",
        color="red",
        label="Future Prediction"
    )

    ax.set_xlabel("Time (Hourly)")
    ax.set_ylabel("Water Yield (L/m²/day)")
    ax.set_title("Past – Present – Future Water Yield Trend")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -------------------------------------------------
    # OUTPUT METRICS
    # -------------------------------------------------
    st.metric("Predicted Water Yield (L/m²/day)", round(present_pred, 3))
    st.caption(f"XGBoost MAE: {round(mae_xgb, 4)}")
