import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load Models & Encoders
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
log_model = joblib.load("log_model.pkl")

# Example: load encoder if used
# encoder = joblib.load("encoder.pkl")

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="🌊 Flood Prediction System", layout="wide")

st.title("🌊 AI-Based Flood Early Warning System")
st.markdown("This app predicts the **likelihood of a flood event** using trained ML models.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🛠 Input Features")

def user_input_features():
    # ⚠️ Replace placeholders with actual dataset columns
    rainfall = st.sidebar.number_input("🌧 Rainfall (mm)", min_value=0.0, max_value=2000.0, step=0.1)
    humidity = st.sidebar.slider("💧 Humidity (%)", 0, 100, 60)
    temperature = st.sidebar.number_input("🌡 Temperature (°C)", min_value=-20.0, max_value=60.0, step=0.1)
    river_level = st.sidebar.number_input("🏞 River Level (m)", min_value=0.0, max_value=100.0, step=0.1)

    # Additional sample feature
    soil_moisture = st.sidebar.slider("🪨 Soil Moisture (%)", 0, 100, 40)

    data = {
        "Rainfall": rainfall,
        "Humidity": humidity,
        "Temperature": temperature,
        "River_Level": river_level,
        "Soil_Moisture": soil_moisture
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("📥 Input Data")
st.write(input_df)

# -----------------------------
# Model Selection
# -----------------------------
model_choice = st.sidebar.radio("🤖 Select Model", ("Random Forest", "Logistic Regression"))

if model_choice == "Random Forest":
    model = rf_model
else:
    model = log_model

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Flood Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prediction")
        st.success("🌊 Flood Expected" if prediction == 1 else "✅ No Flood Risk")

    with col2:
        st.subheader("Confidence Level")
        st.progress(int(probability * 100))
        st.metric(label="Flood Probability", value=f"{probability:.2%}")

    # -----------------------------
    # Advanced: Compare both models
    # -----------------------------
    st.subheader("📊 Model Comparison")
    rf_prob = rf_model.predict_proba(input_df)[0][1]
    log_prob = log_model.predict_proba(input_df)[0][1]

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "Flood Probability": [rf_prob, log_prob]
    })

    st.bar_chart(comparison_df.set_index("Model"))

    # -----------------------------
    # Downloadable Results
    # -----------------------------
    result = input_df.copy()
    result["Selected_Model"] = model_choice
    result["Prediction"] = "Flood" if prediction == 1 else "No Flood"
    result["Probability"] = probability

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction as CSV",
        data=csv,
        file_name="flood_prediction.csv",
        mime="text/csv"
    )
