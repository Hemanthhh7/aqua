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

# ================= SEASON COLORS =================
SEASON_COLORS = {
    "Winter (Dec–Feb)": "#3B82F6",
    "Summer (Mar–May)": "#F97316",
    "Monsoon (Jun–Sep)": "#10B981",
    "Post-Monsoon (Oct–Nov)": "#8B5CF6"
}

# ================= SIDEBAR =================
st.sidebar.title("🌊 AquaGenesis")
st.sidebar.markdown("AI Atmospheric Water Intelligence")

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

# ================= DATA FETCH =================
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
        lambda m: "Winter (Dec–Feb)" if m in [12,1,2] else
        "Summer (Mar–May)" if m in [3,4,5] else
        "Monsoon (Jun–Sep)" if m in [6,7,8,9] else
        "Post-Monsoon (Oct–Nov)"
    )

    return df

# ================= TRAIN MODEL =================
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

# ================= DASHBOARD =================
st.title("Atmospheric Water Intelligence Dashboard")
st.markdown("Hybrid AI Model trained across 28 Indian States.")

if run:

    lat, lon = STATES[state]

    # -------- Past 7 Days --------
    past = fetch_weather(lat, lon, date.today()-timedelta(days=7), date.today()-timedelta(days=1))

    current_month = date.today().month
    if current_month in [12,1,2]:
        current_season = "Winter (Dec–Feb)"
    elif current_month in [3,4,5]:
        current_season = "Summer (Mar–May)"
    elif current_month in [6,7,8,9]:
        current_season = "Monsoon (Jun–Sep)"
    else:
        current_season = "Post-Monsoon (Oct–Nov)"

    st.subheader("Past 7 Days Water Availability")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=past["time"],
        y=past["water_yield"],
        mode="lines",
        line=dict(color=SEASON_COLORS[current_season], width=3),
        name=current_season
    ))

    fig1.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Yield (L/m²/day)"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # -------- Seasonal Comparison --------
    season_df = fetch_weather(lat, lon, date.today()-timedelta(days=90), date.today())
    seasonal_avg = season_df.groupby("season")["water_yield"].mean().reset_index()

    colors = [SEASON_COLORS[s] for s in seasonal_avg["season"]]

    st.subheader("Seasonal Water Yield Comparison")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=seasonal_avg["season"],
        y=seasonal_avg["water_yield"],
        marker_color=colors,
        text=seasonal_avg["water_yield"].round(3),
        textposition="outside"
    ))

    fig2.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Yield (L/m²/day)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # -------- Future Prediction --------
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

    st.subheader("Next 24 Hour Prediction")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(range(1,25)),
        y=xgb_pred,
        mode="lines",
        line=dict(color="#0EA5E9", width=3),
        name="Predicted Yield"
    ))

    fig3.update_layout(
        xaxis_title="Hours from Now",
        yaxis_title="Predicted Yield (L/m²/day)"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.success(f"Hybrid Final Estimated Yield: {round(hybrid,3)} L/m²/day")
