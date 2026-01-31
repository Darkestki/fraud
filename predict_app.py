# =====================================
# 🚨 Fraud Detection Streamlit App
# =====================================

import streamlit as st
import joblib
import pandas as pd

# ------------------------------
# 🎨 Page Settings
# ------------------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Credit Card Fraud Detection System")
st.write("Enter transaction details and predict whether it is **Fraud or Safe** 💳")

# ------------------------------
# 📦 Load Model (only once)
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model (1).joblib")

try:
    model = load_model()
    st.success("✅ Model Loaded Successfully")
except Exception:
    st.error("❌ Model file missing or xgboost not installed.\nAdd it to requirements.txt")
    st.stop()


# ------------------------------
# 🧾 Features
# ------------------------------
features = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

st.subheader("📊 Enter Transaction Values")

# ------------------------------
# 🎛 Inputs
# ------------------------------
inputs = []
cols = st.columns(3)

for i, f in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f, value=0.0, format="%.6f")
        inputs.append(val)


# ------------------------------
# 🔮 Prediction
# ------------------------------
if st.button("🔍 Predict Fraud"):

    data = pd.DataFrame([inputs], columns=features)

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    st.divider()

    if pred == 1:
        st.error("🚨 FRAUD DETECTED!")
        st.write(f"⚠️ Fraud Probability: **{prob:.2%}**")
        st.toast("Suspicious Transaction!", icon="🚨")
    else:
        st.success("✅ SAFE TRANSACTION")
        st.write(f"✔️ Fraud Probability: **{prob:.2%}**")
        st.toast("Transaction Safe", icon="✅")


# ------------------------------
# Footer
# ------------------------------
st.divider()
st.caption("Made with ❤️ using Streamlit + XGBoost")
