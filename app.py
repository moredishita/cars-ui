import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a3e, #24243e);
    color: #e0e0f0;
}

/* Card containers */
.section-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #7c6ff7;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(124,111,247,0.25);
}

/* Header */
.hero-header {
    text-align: center;
    padding: 48px 0 36px;
}

.hero-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.hero-header p {
    color: #8888aa;
    font-size: 16px;
    font-weight: 300;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #7c6ff7, #4f46e5);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(124, 111, 247, 0.35);
    margin-top: 10px;
}

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    margin-bottom: 12px;
}

.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}

.result-sub {
    color: rgba(255,255,255,0.6);
    font-size: 13px;
    margin-top: 10px;
}

/* Inputs */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #b0b0cc !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

/* Button */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #7c6ff7, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 16px 0;
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 8px;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(124,111,247,0.5);
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Columns gap */
.block-container { padding-top: 0rem !important; }

/* Metric */
.stMetric { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("ridge_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

FEATURE_NAMES = [
    'symboling', 'wheelbase', 'carlength', 'carwidth',
    'curbweight', 'enginesize', 'horsepower', 'citympg',
    'carbody_hardtop', 'carbody_hatchback', 'carbody_sedan', 'carbody_wagon',
    'drivewheel_fwd', 'drivewheel_rwd',
    'enginelocation_rear',
    'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc',
    'enginetype_ohcf', 'enginetype_ohcv', 'enginetype_rotor',
    'cylindernumber_five', 'cylindernumber_four', 'cylindernumber_six',
    'cylindernumber_three', 'cylindernumber_twelve', 'cylindernumber_two',
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🚗 Car Price Predictor</h1>
    <p>Enter vehicle specifications below to estimate the market price using Ridge Regression</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ──────────────────────────────────────────────────────────────────────
left, right = st.columns([2, 1], gap="large")

with left:

    # — Section 1: Basic Specs —
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📐 Basic Specifications</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        symboling = st.number_input(
            "Symboling (Risk Rating)",
            min_value=-3, max_value=3, value=0, step=1,
            help="Insurance risk rating: -3 (safe) to +3 (risky)"
        )
    with c2:
        wheelbase = st.number_input(
            "Wheelbase (inches)", min_value=80.0, max_value=130.0,
            value=98.4, step=0.1, format="%.1f"
        )
    with c3:
        carlength = st.number_input(
            "Car Length (inches)", min_value=140.0, max_value=210.0,
            value=172.3, step=0.1, format="%.1f"
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        carwidth = st.number_input(
            "Car Width (inches)", min_value=60.0, max_value=75.0,
            value=65.7, step=0.1, format="%.1f"
        )
    with c5:
        curbweight = st.number_input(
            "Curb Weight (lbs)", min_value=1500, max_value=5000,
            value=2504, step=10
        )
    with c6:
        citympg = st.number_input(
            "City MPG", min_value=10, max_value=70,
            value=26, step=1
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # — Section 2: Engine —
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Engine Details</div>', unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        enginesize = st.number_input(
            "Engine Size (cc)", min_value=60, max_value=330,
            value=126, step=1
        )
    with e2:
        horsepower = st.number_input(
            "Horsepower (hp)", min_value=50, max_value=300,
            value=100, step=1
        )
    with e3:
        enginelocation = st.selectbox(
            "Engine Location",
            options=["Front", "Rear"],
            index=0
        )

    e4, e5 = st.columns(2)
    with e4:
        enginetype = st.selectbox(
            "Engine Type",
            options=["dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"],
            index=3,
            help="dohc = Dual Overhead Cam, ohc = Overhead Cam, etc."
        )
    with e5:
        cylindernumber = st.selectbox(
            "Number of Cylinders",
            options=["two", "three", "four", "five", "six", "eight", "twelve"],
            index=2
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # — Section 3: Body & Drive —
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏎️ Body & Drivetrain</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        carbody = st.selectbox(
            "Car Body Style",
            options=["convertible", "hardtop", "hatchback", "sedan", "wagon"],
            index=3
        )
    with b2:
        drivewheel = st.selectbox(
            "Drive Wheel",
            options=["4wd", "fwd", "rwd"],
            index=1,
            help="4wd = 4-Wheel Drive, fwd = Front, rwd = Rear"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Right panel: Result ────────────────────────────────────────────────────────
with right:
    st.markdown("<br>", unsafe_allow_html=True)

    # Summary card
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)

    summary_data = {
        "Symboling": symboling,
        "Wheelbase": f"{wheelbase} in",
        "Car Length": f"{carlength} in",
        "Car Width": f"{carwidth} in",
        "Curb Weight": f"{curbweight} lbs",
        "Engine Size": f"{enginesize} cc",
        "Horsepower": f"{horsepower} hp",
        "City MPG": f"{citympg} mpg",
        "Body Style": carbody.capitalize(),
        "Drive Wheel": drivewheel.upper(),
        "Engine Type": enginetype.upper(),
        "Cylinders": cylindernumber.capitalize(),
        "Engine Loc.": enginelocation,
    }

    for k, v in summary_data.items():
        col_a, col_b = st.columns([1, 1])
        col_a.markdown(f"<small style='color:#7878aa'>{k}</small>", unsafe_allow_html=True)
        col_b.markdown(f"<small style='color:#e0e0f0;font-weight:500'>{v}</small>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    predict_btn = st.button("🔮  Predict Price", use_container_width=True)

    if predict_btn:
        # ── Build feature vector ──
        # One-hot: carbody
        carbody_hardtop  = 1 if carbody == "hardtop"  else 0
        carbody_hatchback= 1 if carbody == "hatchback" else 0
        carbody_sedan    = 1 if carbody == "sedan"    else 0
        carbody_wagon    = 1 if carbody == "wagon"    else 0

        # One-hot: drivewheel
        drivewheel_fwd = 1 if drivewheel == "fwd" else 0
        drivewheel_rwd = 1 if drivewheel == "rwd" else 0

        # One-hot: enginelocation
        enginelocation_rear = 1 if enginelocation == "Rear" else 0

        # One-hot: enginetype
        enginetype_dohcv = 1 if enginetype == "dohcv" else 0
        enginetype_l     = 1 if enginetype == "l"     else 0
        enginetype_ohc   = 1 if enginetype == "ohc"   else 0
        enginetype_ohcf  = 1 if enginetype == "ohcf"  else 0
        enginetype_ohcv  = 1 if enginetype == "ohcv"  else 0
        enginetype_rotor = 1 if enginetype == "rotor" else 0

        # One-hot: cylindernumber
        cylindernumber_five   = 1 if cylindernumber == "five"   else 0
        cylindernumber_four   = 1 if cylindernumber == "four"   else 0
        cylindernumber_six    = 1 if cylindernumber == "six"    else 0
        cylindernumber_three  = 1 if cylindernumber == "three"  else 0
        cylindernumber_twelve = 1 if cylindernumber == "twelve" else 0
        cylindernumber_two    = 1 if cylindernumber == "two"    else 0

        features = np.array([[
            symboling, wheelbase, carlength, carwidth,
            curbweight, enginesize, horsepower, citympg,
            carbody_hardtop, carbody_hatchback, carbody_sedan, carbody_wagon,
            drivewheel_fwd, drivewheel_rwd,
            enginelocation_rear,
            enginetype_dohcv, enginetype_l, enginetype_ohc,
            enginetype_ohcf, enginetype_ohcv, enginetype_rotor,
            cylindernumber_five, cylindernumber_four, cylindernumber_six,
            cylindernumber_three, cylindernumber_twelve, cylindernumber_two,
        ]])

        scaled   = scaler.transform(features)
        price    = model.predict(scaled)[0]
        price    = max(price, 0)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Price</div>
            <div class="result-price">${price:,.0f}</div>
            <div class="result-sub">Predicted by Ridge Regression Model</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Extra breakdown metrics
        m1, m2 = st.columns(2)
        m1.metric("Engine Size", f"{enginesize} cc")
        m2.metric("Horsepower", f"{horsepower} hp")
        m3, m4 = st.columns(2)
        m3.metric("Curb Weight", f"{curbweight} lbs")
        m4.metric("City MPG", f"{citympg} mpg")
    else:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px dashed rgba(255,255,255,0.12);
            border-radius: 16px;
            padding: 40px 24px;
            text-align: center;
            color: #555577;
        ">
            <div style="font-size:40px;margin-bottom:12px">🔮</div>
            <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:600;color:#6666aa">
                Fill in the specs on the left<br>and hit Predict Price
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#44446a;font-size:12px;padding:12px 0 24px">
    Ridge Regression · 27 Features · StandardScaler Preprocessing
</div>
""", unsafe_allow_html=True)