import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# -----------------------------
# Load Model & Preprocessor
# -----------------------------
model = tf.keras.models.load_model("salary_model.h5", compile=False)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.set_page_config(page_title="💰 Salary Prediction App", layout="centered")
st.title("💰 Customer Salary Prediction using ANN")

# -----------------------------
# Collect Inputs
# -----------------------------
geography = st.selectbox("🌍 Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("👩‍💼 Gender", ["Female", "Male"])
age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
balance = st.number_input("📊 Balance ($)", min_value=0.0, value=0.0)
credit_score = st.number_input("📊 Credit Score", min_value=0, max_value=1000, value=500)
tenure = st.number_input("📅 Tenure (Years)", min_value=0, max_value=10, value=1)
num_products = st.number_input("📦 Number of Products", min_value=1, max_value=10, value=1)
has_credit_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("✅ Is Active Member", [0, 1])

# -----------------------------
# Create DataFrame
# -----------------------------
input_df = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'Tenure': [tenure],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member]
})

# -----------------------------
# Transform & Predict
# -----------------------------
input_scaled = preprocessor.transform(input_df)

if st.button("🔮 Predict Salary"):
    prediction = model.predict(input_scaled)
    st.success(f"💰 Predicted Salary: ${prediction[0][0]:.2f}")