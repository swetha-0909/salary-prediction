import streamlit as st
import joblib

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💼 Salary Prediction App")
st.markdown("Predict whether a person earns more than $50K/year.")

# User input
age = st.number_input("Age", 18, 100, 30)
education_num = st.number_input("Education Level (numeric)", 1, 16, 10)
capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 100000, 0)
hours_per_week = st.number_input("Hours per Week", 1, 100, 40)

# Create a dummy input for all 14 features
input_data = [[age, 0, 0, education_num, 0, 0, 0, 0, 0, 0, capital_gain, capital_loss, hours_per_week, 0]]
input_scaled = scaler.transform(input_data)

if st.button("Predict Salary"):
    prediction = model.predict(input_scaled)
    result = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"✅ Predicted Salary: {result}")
