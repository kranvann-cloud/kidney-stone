import streamlit as st
import numpy as np
import joblib

# ------------------------------
# Load saved model and scaler
# ------------------------------
model = joblib.load("best_decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# App Title & Description
# ------------------------------
st.set_page_config(page_title="Kidney Stone Prediction", page_icon="💧", layout="centered")

st.title("💧 Kidney Stone Prediction App")
st.markdown("""
This application uses a **Decision Tree Classifier** to predict whether 
a person is at risk of developing kidney stones based on urine analysis parameters.
""")

# ------------------------------
# Sidebar inputs
# ------------------------------
st.sidebar.header("🧪 Input Parameters")

gravity = st.sidebar.number_input("Specific Gravity", min_value=1.0, max_value=1.3, step=0.001, value=1.015)
ph = st.sidebar.number_input("pH Value", min_value=4.0, max_value=9.0, step=0.1, value=6.0)
osmo = st.sidebar.number_input("Osmolality", min_value=0, max_value=2000, step=10, value=500)
cond = st.sidebar.number_input("Conductivity", min_value=0.0, max_value=30.0, step=0.1, value=10.0)
urea = st.sidebar.number_input("Urea", min_value=0, max_value=500, step=10, value=200)
calc = st.sidebar.number_input("Calcium", min_value=0.0, max_value=10.0, step=0.1, value=2.5)

# ------------------------------
# Predict button
# ------------------------------
if st.sidebar.button("🔍 Predict"):
    user_input = np.array([[gravity, ph, osmo, cond, urea, calc]])
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]

    st.subheader("🩺 Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk of Kidney Stones Detected!")
    else:
        st.success("✅ No Risk of Kidney Stones Detected.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn | Trained by Suganya R (DS Trainer)")

