# =====================================
# 🚨 Fraud Detection App (Probability Only)
# =====================================

import streamlit as st
import joblib
import pandas as pd

# ------------------------------
# 🎨 Page Config
# ------------------------------
st.set_page_config(
    page_title="Fraud Probability Checker",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Fraud Probability Predictor")
st.write("Upload transaction data and check **Fraud Risk % only** 📊")

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
    st.error("❌ Model not found. Add xgboost_model.joblib")
    st.stop()


# ------------------------------
# 📂 File Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload CSV file (29 features only)",
    type=["csv"]
)


# ------------------------------
# 🔮 Prediction
# ------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("🔍 Check Fraud Probability"):

        probs = model.predict_proba(df)[:, 1]  # probability of fraud

        result = pd.DataFrame({
            "Fraud Probability (%)": (probs * 100).round(2)
        })

        st.divider()
        st.subheader("📊 Results")

        st.dataframe(result)

        avg_prob = probs.mean()

        if avg_prob > 0.5:
            st.error(f"🚨 High Risk Detected ({avg_prob:.2%})")
        else:
            st.success(f"✅ Low Risk ({avg_prob:.2%})")


# ------------------------------
# Footer
# ------------------------------
st.divider()
st.caption("Made with ❤️ using Streamlit + XGBoost")
