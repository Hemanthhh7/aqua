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

SEASON_ORDER = [
    "Winter (Dec–Feb)",
    "Summer (Mar–May)",
    "Monsoon (Jun–Sep)",
    "Post-Monsoon (Oct–Nov)"
]

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

# ================= DASHBOARD =================
st.title("Atmospheric Water Intelligence Dashboard")

if run:

    lat, lon = STATES[state]

    # -------- Seasonal Comparison (Real Data, Ordered) --------
    season_df = fetch_weather(lat, lon, date.today()-timedelta(days=365), date.today())

    seasonal_avg = season_df.groupby("season")["water_yield"].mean()

    # Sort manually in Dec → Nov order
    seasonal_avg = seasonal_avg.reindex(
        [s for s in SEASON_ORDER if s in seasonal_avg.index]
    )

    colors = [SEASON_COLORS[s] for s in seasonal_avg.index]

    st.subheader("Seasonal Water Yield Comparison (Dec → Nov Order)")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=seasonal_avg.index,
        y=seasonal_avg.values,
        marker_color=colors,
        text=seasonal_avg.values.round(3),
        textposition="outside"
    ))

    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Water Yield (L/m²/day)"
    )

    st.plotly_chart(fig, use_container_width=True)
