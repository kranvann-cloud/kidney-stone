import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Title and Header
# ------------------------------
st.set_page_config(page_title="Kidney Stone Prediction App", page_icon="💧", layout="centered")
st.title("💧 Kidney Stone Prediction App")
st.write("Predict the likelihood of kidney stone formation based on biochemical data.")

# ------------------------------
# Load model and scaler safely
# ------------------------------
MODEL_PATH = "best_decision_tree_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error(
        "❌ Model or scaler file not found.\n\n"
        "Please make sure the following files are uploaded to your GitHub repository:\n"
        "- `best_decision_tree_model.pkl`\n"
        "- `scaler.pkl`"
    )
    st.stop()

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------------
# User Input Section
# ------------------------------
st.subheader("🧪 Enter Patient Test Values")

col1, col2 = st.columns(2)
with col1:
    gravity = st.number_input("Specific Gravity", min_value=1.000, max_value=1.050, step=0.001, value=1.020)
    ph = st.number_input("pH Value", min_value=4.0, max_value=9.0, step=0.1, value=6.5)
    osmo = st.number_input("Osmolality", min_value=0, max_value=1500, step=10, value=500)
with col2:
    cond = st.number_input("Conductivity", min_value=0.0, max_value=40.0, step=0.1, value=15.0)
    urea = st.number_input("Urea Concentration", min_value=0, max_value=1000, step=5, value=200)
    calc = st.number_input("Calcium Concentration", min_value=0.0, max_value=20.0, step=0.1, value=8.0)

# Prepare input data
input_data = pd.DataFrame({
    'gravity': [gravity],
    'ph': [ph],
    'osmo': [osmo],
    'cond': [cond],
    'urea': [urea],
    'calc': [calc]
})

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict"):
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"⚠️ **High risk of Kidney Stone Formation!** (Probability: {probability:.2%})")
    else:
        st.success(f"✅ **Low risk of Kidney Stone Formation.** (Probability: {probability:.2%})")

# ------------------------------
# Footer
# ------------------------------
st.write("---")
st.caption("Developed by **Your Name / DS Trainer Suganya R Batch** | Decision Tree Classifier Model")


print("✅ Model and scaler loaded successfully!")
