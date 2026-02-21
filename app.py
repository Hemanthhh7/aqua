import streamlit as st
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="AquaGenesis", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F8FAFC;
}

/* Main Title */
.main-title {
    font-size: 44px;
    font-weight: 700;
    color: #0F172A;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #475569;
}

/* Card Style */
.card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

/* Metric Card */
.metric-card {
    background: linear-gradient(135deg, #2563EB, #14B8A6);
    padding: 25px;
    border-radius: 20px;
    color: white;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    box-shadow: 0px 10px 20px rgba(0,0,0,0.08);
}

/* Section Heading */
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #1E293B;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🌊 AquaGenesis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Atmospheric Water Decision Intelligence</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOTTIE 3D ANIMATION ----------------
def load_lottie(url):
    return requests.get(url).json()

lottie_water = load_lottie("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

col_anim, col_info = st.columns([1,2])

with col_anim:
    st_lottie(lottie_water, height=200)

with col_info:
    st.markdown("""
    ### Intelligent Water Harvesting

    This system predicts atmospheric water availability  
    using advanced AI models (XGBoost + LSTM Hybrid).

    Designed for:
    - Government planning  
    - NGOs  
    - Smart infrastructure  
    - Climate adaptation
    """)

st.markdown("---")

# ---------------- METRIC CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="metric-card">💧 Hybrid Estimate<br><br>0.64 L/m²/day</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="metric-card">⏰ Best Harvest Time<br><br>Next 5 Hours</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="metric-card">🌦 Feasibility<br><br>Moderate</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------- PAST GRAPH CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">📊 Last 7 Days Water Availability</div>', unsafe_allow_html=True)

hours = list(range(1,8))
values = [0.35, 0.42, 0.38, 0.50, 0.47, 0.52, 0.44]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hours,
    y=values,
    mode='lines+markers',
    line=dict(color='#2563EB', width=3),
    marker=dict(size=8)
))

fig.update_layout(
    xaxis_title="Days",
    yaxis_title="Water Yield (L/m²/day)",
    template="plotly_white",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FUTURE GRAPH ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">🔮 Next 24 Hour Prediction</div>', unsafe_allow_html=True)

hours = list(range(1,25))
values = [0.30 + i*0.015 for i in range(24)]

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=hours,
    y=values,
    mode='lines',
    line=dict(color='#14B8A6', width=4)
))

fig2.update_layout(
    xaxis_title="Hours from Now",
    yaxis_title="Predicted Water Yield (L/m²/day)",
    template="plotly_white",
    height=400
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.write("© 2026 AquaGenesis | Designed for Smart Water Infrastructure")
