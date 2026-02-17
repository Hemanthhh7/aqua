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
# TRAIN MODEL USING ALL 28 STATES
# -------------------------------------------------
@st.cache_resource
def train_model():
    all_data = []

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)

    for state, (lat, lon) in STATES.items():
        try:
            df = fetch_past_weather(lat, lon, start, end)
            df = add_water_yield(df)
            all_data.append(df)
        except:
            continue

    full_df = pd.concat(all_data, ignore_index=True)

    X = full_df[["temperature", "humidity", "dew_point", "pressure"]]
    y = full_df["water_yield"]

    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(Xtr, ytr)

    return model

model = train_model()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌊 AquaGenesis – Decision Support System (India Model)")
state = st.selectbox("Select Indian State", list(STATES.keys()))

if st.button("Run Full Analysis"):

    lat, lon = STATES[state]

    # PAST DATA
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)
    past = add_water_yield(fetch_past_weather(lat, lon, start, end))

    present = past["water_yield"].iloc[-1]

    # FUTURE DATA
    future = add_water_yield(fetch_future_weather(lat, lon))
    hours = list(range(1, len(future)+1))

    # FEASIBILITY
    if present > 0.5:
        feasibility = "🟢 HIGH – Suitable for installation"
    elif present > 0.3:
        feasibility = "🟡 MODERATE – Seasonal use recommended"
    else:
        feasibility = "🔴 LOW – Not recommended"

    # BEST TIME
    best_hour = future["water_yield"].idxmax() + 1

    # ENERGY RATIO
    energy_ratio = round(present * 3.2, 2)

    # ALERT
    alert = "⚠️ Low water availability expected" if future["water_yield"].mean() < 0.3 else "✅ Conditions are favorable"

    # EXPLAINABILITY
    imp = pd.DataFrame({
        "Factor": ["Temperature", "Humidity", "Dew Point", "Pressure"],
        "Impact (%)": model.feature_importances_ * 100
    }).sort_values(by="Impact (%)", ascending=False)

    # OUTPUT
    st.subheader("📊 Past Water Availability")
    st.line_chart(past.set_index("time")["water_yield"])

    st.metric("💧 Current Water Yield (L/m²/day)", round(present, 3))
    st.success(feasibility)

    st.subheader("🔮 Future Water Availability (Hours Ahead)")
    fig, ax = plt.subplots()
    ax.plot(hours, future["water_yield"], marker="o")
    ax.set_xlabel("Hours Ahead")
    ax.set_ylabel("Water Yield (L/m²/day)")
    st.pyplot(fig)

    st.info(f"⏰ Best harvesting time: after {best_hour} hour(s)")
    st.info(f"⚡ Energy–Water Tradeoff: ~{energy_ratio} litres per unit electricity")
    st.warning(alert)

    st.subheader("🔍 Explainability (Feature Importance)")
    st.table(imp)

    st.subheader("🚧 Future Scope")
    st.markdown("""
    - District-level mapping  
    - Seasonal comparison  
    - Climate change projection (2030–2050)  
    - Population water demand integration  
    """)
