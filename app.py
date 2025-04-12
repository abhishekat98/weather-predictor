import os
st.write("Current directory contents:", os.listdir("."))
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Load model and label encoder
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "compressed_weather_rf_model.pkl.gz")
encoder_path = os.path.join(current_dir, "lable_encoder.pkl")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)


# Set page config
st.set_page_config(page_title="Weather Predictor 🌦️", layout="centered")

# App title
st.title("🌤️ Weather Condition Predictor")
st.markdown("Use the sliders below to enter weather parameters and get a real-time prediction.")

# User Inputs
temperature = st.slider("🌡️ Temperature (°C)", -30.0, 50.0, 25.0)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
wind_kph = st.slider("🌬️ Wind Speed (km/h)", 0, 150, 10)
wind_degree = st.slider("🧭 Wind Direction (°)", 0, 360, 90)
pressure_mb = st.slider("📉 Pressure (mb)", 900, 1100, 1013)
precip_mm = st.slider("🌧️ Precipitation (mm)", 0.0, 50.0, 0.0)
cloud = st.slider("☁️ Cloud Coverage (%)", 0, 100, 25)

# Predict button
if st.button("🔍 Predict Weather"):
    input_data = np.array([[temperature, humidity, wind_kph, wind_degree, pressure_mb, precip_mm, cloud]])
    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🌈 **Predicted Weather Condition:** {predicted_label}")

    # Save prediction to session state
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Temp (°C)": temperature,
        "Humidity (%)": humidity,
        "Wind (kph)": wind_kph,
        "Wind Dir (°)": wind_degree,
        "Pressure (mb)": pressure_mb,
        "Precip (mm)": precip_mm,
        "Cloud (%)": cloud,
        "Prediction": predicted_label
    })

# Show history
if 'history' in st.session_state and st.session_state.history:
    st.markdown("### 🕓 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
