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
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide")

# ================= 3D GLASS CSS =================
st.markdown("""
<style>

/* Animated Water Gradient Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #dbeafe, #e0f2fe, #f0f9ff, #ccfbf1);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.3);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.4);
    margin-bottom: 30px;
}

/* Metric Glass */
.metric-glass {
    background: rgba(255,255,255,0.35);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
    box-shadow: 0 6px 30px rgba(0,0,0,0.1);
}

.title {
    font-size: 50px;
    font-weight: 700;
    text-align: center;
    color: #0f172a;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #334155;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="title">🌊 AquaGenesis 3D Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid AI Atmospheric Water Decision Platform</div>', unsafe_allow_html=True)

# ================= LOTTIE =================
def load_lottie(url):
    return requests.get(url).json()

lottie = load_lottie("https://assets10.lottiefiles.com/packages/lf20_j1adxtyb.json")
st_lottie(lottie, height=250)

# ================= STATES =================
STATES = {
    "Andhra Pradesh (Amaravati)": (16.5730, 80.3575),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Maharashtra (Mumbai)": (19.0760, 72.8777),
    "Karnataka (Bengaluru)": (12.9716, 77.5946),
    "Telangana (Hyderabad)": (17.3850, 78.4867)
}

state = st.selectbox("Select State", list(STATES.keys()))

# ================= DATA FUNCTION =================
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

# ================= TRAIN MODELS =================
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

    xgb = XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=5)
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

if st.button("🚀 Run 3D Analysis"):

    lat, lon = STATES[state]

    # Past 7 Days
    past = fetch_weather(lat, lon, date.today()-timedelta(days=7), date.today()-timedelta(days=1))

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Last 7 Days Water Availability")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past["time"],
        y=past["water_yield"],
        mode="lines",
        line=dict(color="#2563EB", width=4)
    ))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Yield (L/m²/day)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Seasonal
    season_df = fetch_weather(lat, lon, date.today()-timedelta(days=90), date.today())
    seasonal_avg = season_df.groupby("season")["water_yield"].mean()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌦 Seasonal Comparison")

    fig2 = go.Figure([go.Bar(x=seasonal_avg.index, y=seasonal_avg.values)])
    fig2.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Water Yield"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔮 Next 24 Hour Prediction")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(range(1,25)),
        y=xgb_pred,
        mode="lines",
        line=dict(color="#14B8A6", width=4)
    ))
    fig3.update_layout(
        xaxis_title="Hours from Now",
        yaxis_title="Predicted Water Yield"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.success(f"Hybrid Estimated Yield: {round(hybrid,3)} L/m²/day")
    st.markdown('</div>', unsafe_allow_html=True)
