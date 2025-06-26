
import streamlit as st
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Crop Recommendation System", layout="centered")
st.title("🌾 Crop Recommendation System")
st.markdown("Predict the most suitable crop based on soil and climate conditions.")

st.markdown("<style>label { color: white !important; }</style>", unsafe_allow_html=True)

# User input fields
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
    K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, step=0.1)
    ph = st.number_input("pH Level", min_value=0.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)

if st.button("Predict Crop"):
    # Prepare input for model
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    predicted_crop = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🌱 Recommended Crop: **{predicted_crop.upper()}**")
    st.balloons()

