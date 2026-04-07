import streamlit as st
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AquaGenesis Intelligence", layout="wide")

# ================= SIDEBAR =================
st.sidebar.title("🌊 AquaGenesis")
st.sidebar.markdown("Hybrid AI Atmospheric Water Intelligence")

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

state = st.sidebar.selectbox("Select State", list(STATES.keys()))
run = st.sidebar.button("Run Full Analysis")

# ================= SAFE API =================
def safe_api_call(url, params):
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

# ================= SYNTHETIC DATA =================
def generate_synthetic_data(hours=24):
    return pd.DataFrame({
        "temperature": np.random.uniform(25, 35, hours),
        "humidity": np.random.uniform(60, 90, hours),
        "dew_point": np.random.uniform(20, 25, hours),
        "pressure": np.random.uniform(1000, 1015, hours)
    })

# ================= FETCH WEATHER =================
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

    r = safe_api_call(url, params)

    if r is None or "hourly" not in r:
        df = generate_synthetic_data(168)  # 7 days
        df["time"] = pd.date_range(end=pd.Timestamp.now(), periods=168, freq="H")
    else:
        df = pd.DataFrame({
            "time": pd.to_datetime(r["hourly"]["time"]),
            "temperature": r["hourly"]["temperature_2m"],
            "humidity": r["hourly"]["relative_humidity_2m"],
            "dew_point": r["hourly"]["dewpoint_2m"],
            "pressure": r["hourly"]["surface_pressure"]
        })

    df["water_yield"] = (df["humidity"]/100)*(df["temperature"]-df["dew_point"])*0.1
    return df

# ================= TRAIN =================
@st.cache_resource
def train_models():
    all_data = []

    for lat, lon in STATES.values():
        df = generate_synthetic_data(200)
        df["water_yield"] = (df["humidity"]/100)*(df["temperature"]-df["dew_point"])*0.1
        all_data.append(df)

    full_df = pd.concat(all_data)

    X = full_df[["temperature","humidity","dew_point","pressure"]]
    y = full_df["water_yield"]

    xgb = XGBRegressor(n_estimators=50)
    xgb.fit(X, y)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df[["water_yield"]])

    window = 12
    X_lstm, y_lstm = [], []

    for i in range(window, len(scaled)):
        X_lstm.append(scaled[i-window:i])
        y_lstm.append(scaled[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm = Sequential()
    lstm.add(LSTM(16, input_shape=(window,1)))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_lstm, y_lstm, epochs=1, batch_size=64, verbose=0)

    return xgb, lstm, scaler

xgb, lstm, scaler = train_models()

# ================= MAIN =================
st.title("Atmospheric Water Intelligence Dashboard")

if run:

    lat, lon = STATES[state]

    past = fetch_weather(lat, lon, date.today()-timedelta(days=7), date.today())

    present_yield = past["water_yield"].iloc[-1]
    st.metric("Current Water Yield (L/m²/day)", round(present_yield,3))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=past["time"], y=past["water_yield"], mode="lines"))
    st.plotly_chart(fig1, use_container_width=True)

    # ===== Forecast =====
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }

    f = safe_api_call(forecast_url, params)

    if f is None or "hourly" not in f:
        future_df = generate_synthetic_data(12)
    else:
        future_df = pd.DataFrame({
            "temperature": f["hourly"]["temperature_2m"],
            "humidity": f["hourly"]["relative_humidity_2m"],
            "dew_point": f["hourly"]["dewpoint_2m"],
            "pressure": f["hourly"]["surface_pressure"]
        }).head(12)

    xgb_pred = xgb.predict(future_df)

    scaled_input = scaler.transform(xgb_pred.reshape(-1,1))
    lstm_input = scaled_input.reshape(1,12,1)
    lstm_pred = scaler.inverse_transform(lstm.predict(lstm_input))[0][0]

    hybrid_yield = (np.mean(xgb_pred)+lstm_pred)/2

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(1,13)), y=xgb_pred, mode="lines"))
    st.plotly_chart(fig2, use_container_width=True)

    st.metric("Hybrid Predicted Yield (Next 12h Avg)", round(hybrid_yield,3))

    if hybrid_yield > 0.5:
        st.success("🟢 HIGH – Suitable")
    elif hybrid_yield > 0.3:
        st.warning("🟡 MODERATE – Seasonal Use")
    else:
        st.error("🔴 LOW – Not Recommended")
