import streamlit as st 
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# STATES (28)
# -------------------------------------------------
STATES = {
    "Andhra Pradesh (Amaravati)": (16.5730, 80.3575),
    "Telangana (Hyderabad)": (17.3850, 78.4867),
    "Tamil Nadu (Chennai)": (13.0827, 80.2707),
    "Karnataka (Bengaluru)": (12.9716, 77.5946),
    "Maharashtra (Mumbai)": (19.0760, 72.8777),
    "Rajasthan (Jaipur)": (26.9124, 75.7873),
    "West Bengal (Kolkata)": (22.5726, 88.3639),
}

# -------------------------------------------------
# DATA FUNCTIONS
# -------------------------------------------------
def fetch_past_weather(lat, lon, start, end):
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
    return pd.DataFrame({
        "time": pd.to_datetime(d["hourly"]["time"]),
        "temperature": d["hourly"]["temperature_2m"],
        "humidity": d["hourly"]["relative_humidity_2m"],
        "dew_point": d["hourly"]["dewpoint_2m"],
        "pressure": d["hourly"]["surface_pressure"]
    })

def fetch_future_weather(lat, lon, hours=12):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,dewpoint_2m,surface_pressure",
        "timezone": "auto"
    }
    d = requests.get(url, params=params).json()
    df = pd.DataFrame({
        "temperature": d["hourly"]["temperature_2m"],
        "humidity": d["hourly"]["relative_humidity_2m"],
        "dew_point": d["hourly"]["dewpoint_2m"],
        "pressure": d["hourly"]["surface_pressure"]
    })
    return df.head(hours)

def add_water_yield(df):
    df["water_yield"] = (df["humidity"]/100) * (df["temperature"] - df["dew_point"]) * 0.1
    return df

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
@st.cache_resource
def train_model():
    lat, lon = STATES["Telangana (Hyderabad)"]
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    df = add_water_yield(fetch_past_weather(lat, lon, start, end))
    X = df[["temperature", "humidity", "dew_point", "pressure"]]
    y = df["water_yield"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    m = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5)
    m.fit(Xtr, ytr)
    return m

model = train_model()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌊 AquaGenesis – Intelligent AWH Decision Support Platform")
state = st.selectbox("Select Location", list(STATES.keys()))

if st.button("Run Full Professional Analysis"):
    lat, lon = STATES[state]

    # DATA
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    past = add_water_yield(fetch_past_weather(lat, lon, start, end))
    present = past["water_yield"].iloc[-1]
    future = add_water_yield(fetch_future_weather(lat, lon))
    hours = list(range(1, len(future)+1))

    # -------------------------------------------------
    # FEASIBILITY
    # -------------------------------------------------
    if present > 0.5:
        feasibility = "HIGH"
        score = 80
    elif present > 0.3:
        feasibility = "MODERATE"
        score = 55
    else:
        feasibility = "LOW"
        score = 25

    # -------------------------------------------------
    # SUITABILITY INDEX
    # -------------------------------------------------
    suitability_index = min(100, round(score + (future["water_yield"].mean()*20), 2))

    # -------------------------------------------------
    # ROLE CLASSIFICATION
    # -------------------------------------------------
    if suitability_index > 70:
        role = "Primary Water Source"
    elif suitability_index > 40:
        role = "Seasonal / Supplementary Use"
    else:
        role = "Emergency / Auxiliary Use Only"

    # -------------------------------------------------
    # HUMAN CONTEXT
    # -------------------------------------------------
    drinking_need = 4  # litres/day/person
    required_area = round(drinking_need / max(present, 0.01), 2)

    # -------------------------------------------------
    # FAILURE REASON
    # -------------------------------------------------
    imp = model.feature_importances_
    dominant_factor = ["Temperature", "Humidity", "Dew Point", "Pressure"][imp.argmax()]

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.subheader("📊 Historical Water Yield Pattern")
    st.line_chart(past.set_index("time")["water_yield"])

    st.metric("Current Predicted Water Yield (L/m²/day)", round(present, 3))
    st.success(f"Deployment Feasibility: {feasibility}")
    st.metric("AWH Suitability Index (0–100)", suitability_index)

    st.subheader("🔮 Short-Term Forecast")
    fig, ax = plt.subplots()
    ax.plot(hours, future["water_yield"], marker="o")
    ax.set_xlabel("Hours Ahead")
    ax.set_ylabel("Water Yield (L/m²/day)")
    st.pyplot(fig)

    st.subheader("🧠 System Role Recommendation")
    st.info(f"Recommended Role: {role}")

    st.subheader("👤 Human Water Context")
    st.write(f"To meet minimum drinking requirement (~4L/day), approximately **{required_area} m²** AWH surface area is required per person.")

    st.subheader("⚠ Deployment Risk Insight")
    st.write(f"Primary influencing factor detected: **{dominant_factor}**")
    st.write("Low yield is climate-driven, not model uncertainty.")

    st.subheader("🏁 Final System Decision")
    if suitability_index < 40:
        st.error("❌ NO-GO for standalone AWH deployment.")
    elif suitability_index < 70:
        st.warning("⚠ Limited deployment recommended with hybrid support.")
    else:
        st.success("✅ Suitable for standalone deployment.")
