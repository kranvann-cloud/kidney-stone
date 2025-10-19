import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
try:
    model = joblib.load('best_decision_tree_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please make sure 'best_decision_tree_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()


st.title("🩺 Kidney Stone Prediction App")

st.write("Enter the patient's information to predict the risk of kidney stones.")

# Create input fields for each feature
gravity = st.number_input("Specific Gravity", min_value=1.0, max_value=1.3, value=1.015, step=0.001)
ph = st.number_input("pH Value", min_value=4.0, max_value=9.0, value=5.5, step=0.1)
osmo = st.number_input("Osmolality (mOsm/kg)", min_value=0, max_value=2000, value=500, step=10)
cond = st.number_input("Conductivity (mS/cm)", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
urea = st.number_input("Urea (mg/dl)", min_value=0, max_value=500, value=200, step=10)
calc = st.number_input("Calcium (mg/dl)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

# Create a button to trigger prediction
if st.button("Predict"):
    # Prepare the input data as a numpy array
    user_input = np.array([[gravity, ph, osmo, cond, urea, calc]])

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(user_input)

    # Make a prediction
    prediction = model.predict(scaled_input)[0]

    # Display the prediction result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ Based on the provided information, there is a High Risk of Kidney Stones.")
    else:
        st.success("✅ Based on the provided information, there is No Kidney Stone Risk Detected.")
        import numpy as np

# Sample input data (replace with your actual new data)
# The order of values should match the features used during training: gravity, ph, osmo, cond, urea, calc
new_data = np.array([[1.015, 5.5, 500, 15.0, 200, 2.0]]) # Example values

# Scale the new data using the loaded scaler
scaled_new_data = loaded_scaler.transform(new_data)

# Make a prediction using the loaded model
prediction = loaded_model.predict(scaled_new_data)

# Interpret the prediction
if prediction[0] == 1:
    print("Prediction: High Risk of Kidney Stones")
else:
    print("Prediction: No Kidney Stone Risk Detected")
    import joblib

loaded_model = joblib.load("best_decision_tree_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

print("✅ Model and scaler loaded successfully!")
