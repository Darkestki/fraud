# =========================================
# 🚨 Fraud Detection App (CSV Upload Version)
# =========================================

import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# 🎨 Page Settings
# ---------------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Credit Card Fraud Detection")
st.write("Upload transaction data and check **Fraud Probability** instantly 📊")


# ---------------------------
# 📦 Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.joblib")

try:
    model = load_model()
    st.success("✅ Model Loaded Successfully")
except:
    st.error("❌ Model not found or xgboost missing.\nAdd it in requirements.txt")
    st.stop()


# ---------------------------
# 📁 Upload CSV
# ---------------------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV File (29 features only)",
    type=["csv"]
)


# ---------------------------
# 🔮 Prediction after upload
# ---------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        probs = model.predict_proba(df)[:, 1]
        df["Fraud_Probability"] = probs

        st.divider()
        st.subheader("🚨 Fraud Probability Results")

        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

        st.success("✅ Prediction Completed Successfully!")

    except Exception as e:
        st.error("❌ Feature mismatch!\nMake sure CSV has same 29 features used in training.")
        st.stop()


# ---------------------------
# Footer
# ---------------------------
st.divider()
st.caption("Made with ❤️ using Streamlit + XGBoost")
