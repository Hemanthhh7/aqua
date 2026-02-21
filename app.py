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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AquaGenesis Pro", layout="wide")

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F8FAFC, #EEF2F7);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B, #0F172A);
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main Title */
.main-title {
    font-size: 48px;
    font-weight: 700;
    color: #0F172A;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #475569;
}

/* Card */
.card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.05);
    margin-bottom: 30px;
}

/* Metric Card */
.metric-card {
    background: linear-gradient(135deg, #2563EB, #14B8A6);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
}

/* Section Title */
.section-title {
    font-size: 26px;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌊 AquaGenesis Pro")
st.sidebar.write("AI Water Intelligence Dashboard")

# 28 States
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

selected_state = st.sidebar.selectbox("Select State", list(STATES.keys()))
run = st.sidebar.button("🚀 Run Analysis")

# ---------------- DATA FUNCTION ----------------
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
    r = requests.get(url, params=params).json()

    df = pd.DataFrame({
        "time": pd.to_datetime(r["hourly"]["time"]),
        "temperature": r["hourly"]["temperature_2m"],
        "humidity": r["hourly"]["relative_humidity_2m"],
        "dew_point": r["hourly"]["dewpoint_2m"],
        "pressure": r["hourly"]["surface_pressure"]
    }).dropna()

    df["water_yield"] = (df["humidity"]/100)*(df["temperature"]-df["dew_point"])*0.1
    df["month"] = df["time"].dt.month
    df["season"] = df["month"].apply(
        lambda m: "Winter" if m in [12,1,2] else
        "Summer" if m in [3,4,5] else
        "Monsoon" if m in [6,7,8,9] else
        "Post-Monsoon"
    )
    return df

# ---------------- TRAIN MODEL (FAST) ----------------
@st.cache_resource
def train_models():
    all_data = []
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=90)

    for lat, lon in STATES.values():
        try:
            df = fetch_weather(lat, lon, start, end)
            all_data.append(df)
        except:
            continue

    full_df = pd.concat(all_data)

    X = full_df[["temperature","humidity","dew_point","pressure"]]
    y = full_df["water_yield"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5)
    xgb.fit(X_train, y_train)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full_df[["water_yield"]])

    window = 24
    X_lstm, y_lstm = [], []
    for i in range(window, len(scaled)):
        X_lstm.append(scaled[i-window:i])
        y_lstm.append(scaled[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(window,1)))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_lstm, y_lstm, epochs=2, batch_size=128, verbose=0)

    return xgb, lstm, scaler

xgb, lstm, scaler = train_models()

# ---------------- MAIN HEADER ----------------
st.markdown('<div class="main-title">Atmospheric Water Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid AI Model | 28-State Training | Seasonal Insights</div>', unsafe_allow_html=True)
st.markdown("---")

if run:

    lat, lon = STATES[selected_state]

    # Past 7 Days
    past = fetch_weather(lat, lon, date.today()-timedelta(days=7), date.today()-timedelta(days=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past["time"],
        y=past["water_yield"],
        mode="lines",
        line=dict(color="#2563EB", width=3)
    ))
    fig.update_layout(
        title="Last 7 Days Water Availability",
        xaxis_title="Date",
        yaxis_title="Water Yield (L/m²/day)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal
    season_df = fetch_weather(lat, lon, date.today()-timedelta(days=90), date.today())
    seasonal_avg = season_df.groupby("season")["water_yield"].mean()

    fig2 = go.Figure([go.Bar(x=seasonal_avg.index, y=seasonal_avg.values)])
    fig2.update_layout(title="Seasonal Water Yield Comparison")
    st.plotly_chart(fig2, use_container_width=True)

    # Future
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }

    f = requests.get(forecast_url, params=params).json()

    future_df = pd.DataFrame({
        "temperature": f["hourly"]["temperature_2m"],
        "humidity": f["hourly"]["relative_humidity_2m"],
        "dew_point": f["hourly"]["dewpoint_2m"],
        "pressure": f["hourly"]["surface_pressure"]
    }).head(24)

    xgb_pred = xgb.predict(future_df)
    scaled_input = scaler.transform(xgb_pred.reshape(-1,1))
    lstm_input = scaled_input.reshape(1,24,1)
    lstm_pred = scaler.inverse_transform(lstm.predict(lstm_input))[0][0]

    hybrid = (np.mean(xgb_pred)+lstm_pred)/2

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(range(1,25)),
        y=xgb_pred,
        mode="lines",
        line=dict(color="#14B8A6", width=3)
    ))
    fig3.update_layout(title="Next 24 Hour Prediction")
    st.plotly_chart(fig3, use_container_width=True)

    st.success(f"Hybrid Final Estimated Yield: {round(hybrid,3)} L/m²/day")
