import streamlit as st
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide")

# ==================== FULL 3D WATER BACKGROUND ====================
st.markdown("""
<style>

/* Remove default padding */
.block-container {
    padding-top: 0rem;
}

/* Full screen animated canvas */
#water-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    overflow: hidden;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border-radius: 35px;
    padding: 45px;
    box-shadow: 0 8px 60px rgba(0,0,0,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    margin-bottom: 60px;
}

/* Hero */
.hero {
    text-align: center;
    padding-top: 120px;
    padding-bottom: 60px;
}

.hero h1 {
    font-size: 70px;
    font-weight: 800;
    color: #0f172a;
}

.hero p {
    font-size: 22px;
    color: #334155;
}

/* Glow effect */
.glow {
    box-shadow: 0 0 40px rgba(37,99,235,0.4);
}

/* Metric box */
.metric {
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(20px);
    border-radius: 30px;
    padding: 40px;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    box-shadow: 0 8px 40px rgba(0,0,0,0.15);
}

</style>

<div id="water-bg">
<canvas id="canvas"></canvas>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let waves = [];

for(let i=0;i<5;i++){
    waves.push({
        y: Math.random()*canvas.height,
        length: Math.random()*200+200,
        amplitude: Math.random()*30+20,
        speed: Math.random()*0.02+0.01
    });
}

function animate(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = "#e0f2fe";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    waves.forEach(wave=>{
        ctx.beginPath();
        for(let x=0;x<canvas.width;x++){
            let y = wave.y + Math.sin(x*0.01+Date.now()*wave.speed)*wave.amplitude;
            ctx.lineTo(x,y);
        }
        ctx.strokeStyle="rgba(37,99,235,0.15)";
        ctx.lineWidth=3;
        ctx.stroke();
    });

    requestAnimationFrame(animate);
}
animate();
</script>
""", unsafe_allow_html=True)

# ==================== HERO ====================
st.markdown("""
<div class="hero">
<h1>🌊 AquaGenesis</h1>
<p>Atmospheric Water Intelligence Engine</p>
</div>
""", unsafe_allow_html=True)

# ==================== LOTTIE ====================
def load_lottie(url):
    return requests.get(url).json()

water = load_lottie("https://assets10.lottiefiles.com/packages/lf20_j1adxtyb.json")
st_lottie(water, height=300)

# ==================== METRICS ====================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric glow">💧 Hybrid Yield<br><br>0.71 L/m²/day</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric glow">⏰ Best Time<br><br>Next 3 Hours</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric glow">🌦 Feasibility<br><br>High</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ==================== PAST GRAPH ====================
st.markdown('<div class="glass glow">', unsafe_allow_html=True)

st.markdown("## 📊 Past 7 Days Atmospheric Water")

days = list(range(1,8))
values = [0.42,0.48,0.39,0.55,0.51,0.60,0.47]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=days,
    y=values,
    mode='lines+markers',
    line=dict(color='#2563EB', width=5),
    marker=dict(size=12)
))
fig.update_layout(
    xaxis_title="Days",
    yaxis_title="Water Yield (L/m²/day)",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==================== FUTURE GRAPH ====================
st.markdown('<div class="glass glow">', unsafe_allow_html=True)

st.markdown("## 🔮 24 Hour AI Prediction")

hours = list(range(1,25))
future = [0.35+i*0.02 for i in range(24)]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=hours,
    y=future,
    mode='lines',
    line=dict(color='#14B8A6', width=5)
))
fig2.update_layout(
    xaxis_title="Hours From Now",
    yaxis_title="Predicted Yield",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
