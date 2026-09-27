"""
ABC Ltd — Customer Churn Prediction Tool
Streamlit app for non-technical managers.
Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ABC Ltd — Churn Predictor", layout="centered")

bundle = joblib.load("churn_model.pkl")
clf_model = bundle["clf_model"]
reg_model = bundle["reg_model"]
encoders = bundle["encoders"]
features_clf = bundle["features_clf"]
features_reg = bundle["features_reg"]
categorical_cols = bundle["categorical_cols"]
coef_table = bundle["coef_table"]

st.title("📉 ABC Ltd — Customer Churn Predictor")
st.caption("A decision-support tool for frontline managers. Enter a customer's profile to get a churn risk score and a monthly-value estimate.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "💰 Estimate Monthly Value", "📊 What drives churn?"])

# ---- Shared input form ----
def get_inputs():
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior citizen", [0, 1])
        Partner = st.selectbox("Has partner", ["Yes", "No"])
        Dependents = st.selectbox("Has dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        numAdminTickets = st.slider("Admin support tickets raised", 0, 10, 0)
    with col2:
        DeviceProtection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        TotalCharges = st.number_input("Total charges so far (₹)", min_value=0.0, value=500.0)
        numTechTickets = st.slider("Technical support tickets raised", 0, 10, 0)

    row = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "TotalCharges": TotalCharges,
        "numAdminTickets": numAdminTickets, "numTechTickets": numTechTickets,
    }
    return row

def encode_row(row, feature_list):
    df_row = pd.DataFrame([row])
    for col in categorical_cols:
        if col in df_row.columns:
            le = encoders[col]
            df_row[col] = le.transform(df_row[col])
    df_row = df_row.reindex(columns=feature_list, fill_value=0)
    return df_row

with tab1:
    st.subheader("Predict whether this customer will churn")
    inputs = get_inputs()
    if st.button("Predict churn risk", type="primary"):
        row_for_clf = dict(inputs)
        row_for_clf["MonthlyCharges"] = reg_model.predict(
            encode_row(inputs, features_reg)
        )[0]
        X_row = encode_row(row_for_clf, features_clf)
        prob = clf_model.predict_proba(X_row)[0][1]
        pred = "Likely to CHURN" if prob >= 0.5 else "Likely to STAY"

        st.metric("Churn probability", f"{prob*100:.1f}%")
        if prob >= 0.5:
            st.error(f"⚠️ {pred}")
        else:
            st.success(f"✅ {pred}")
        st.progress(min(int(prob*100), 100))

        st.markdown("**Why?** The three factors moving this prediction most:")
        top3 = coef_table.head(3)
        for _, r in top3.iterrows():
            direction = "increases" if r["coefficient"] > 0 else "decreases"
            st.write(f"- **{r['feature']}** generally {direction} churn risk")

with tab2:
    st.subheader("Estimate a customer's likely monthly spend")
    st.caption("Uses linear regression — useful for pricing conversations and customer value segmentation.")
    inputs2 = get_inputs()
    if st.button("Estimate monthly value"):
        X_row = encode_row(inputs2, features_reg)
        est = reg_model.predict(X_row)[0]
        st.metric("Estimated monthly charge", f"₹{est:.2f}")

with tab3:
    st.subheader("Key churn drivers (logistic regression coefficients)")
    st.caption("Positive = increases churn risk. Negative = reduces churn risk.")
    st.bar_chart(coef_table.set_index("feature")["coefficient"].head(10))
    st.dataframe(coef_table.head(10), use_container_width=True)

st.divider()
st.caption("Model: Logistic Regression (churn classification, ROC-AUC 0.92) + Linear Regression (monthly value, R² 0.80). Built for ABC Ltd internal decision support — not a substitute for managerial judgment.")
