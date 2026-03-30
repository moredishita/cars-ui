import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Title
st.title("🚗 Car Price Prediction App")
st.markdown("Enter car details to predict price")
st.markdown("---")

# SECTION 1: Basic Car Info
st.subheader("📌 Basic Information")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2026)

with col2:
    present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0)

with col3:
    kms_driven = st.number_input("Kilometers Driven", min_value=0)

# SECTION 2: Car Details
st.subheader("⚙️ Car Details")

col4, col5, col6 = st.columns(3)

with col4:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

with col5:
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

with col6:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# SECTION 3: Ownership
st.subheader("👤 Ownership Details")

col7, col8 = st.columns(2)

with col7:
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col8:
    st.write("")  # spacing

st.markdown("---")

# Convert categorical to numeric (IMPORTANT)
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
transmission_map = {"Manual": 0, "Automatic": 1}

# Button
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])

with col_btn2:
    predict = st.button("🚀 Predict Car Price")

# Prediction
if predict:
    try:
        features = np.array([[
            year,
            present_price,
            kms_driven,
            fuel_map[fuel_type],
            seller_map[seller_type],
            transmission_map[transmission],
            owner
        ]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)

        st.success(f"💰 Estimated Car Price: ₹ {round(prediction[0], 2)} Lakhs")

    except Exception as e:
        st.error("⚠️ Error in prediction. Check inputs or model compatibility.")