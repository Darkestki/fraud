# =====================================
# 🚨 Fraud Detection Streamlit App
# Upload CSV → Predict Probability Only
# =====================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# 🎨 Page Config
# ------------------------------
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Credit Card Fraud Detection")
st.write("📂 Upload transaction file → Get **Fraud Probability** instantly")


# ------------------------------
# 📦 Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.joblib")

try:
    model = load_model()
    st.success("✅ Model Loaded")
except:
    st.error("❌ Model or xgboost missing. Add in requirements.txt")
    st.stop()


# ------------------------------
# 📂 Upload Section
# ------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload CSV File",
    type=["csv"]
)


# ------------------------------
# 🔮 Prediction Logic
# ------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    try:
        # Predict probability only
        fraud_prob = model.predict_proba(df)[:, 1]

        df["Fraud_Probability (%)"] = fraud_prob * 100

        st.divider()
        st.subheader("🚨 Prediction Results")

        st.dataframe(df)

        # Simple summary
        st.info(f"📊 Average Fraud Risk: {fraud_prob.mean()*100:.2f}%")

        # Download button
        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "fraud_predictions.csv",
            "text/csv"
        )

        st.success("✅ Prediction Completed!")

    except Exception:
        st.error("❌ Feature mismatch!\nMake sure CSV has same 29 features (V1–V28 + Amount)")
        st.stop()


# ------------------------------
# Footer
# ------------------------------
st.divider()
st.caption("Made with ❤️ using Streamlit + XGBoost")
