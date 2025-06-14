import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))
try:
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    scaler = None

feature_columns = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes',
    'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
    'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes',
    'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
]

st.title("🏦Loan Approval Prediction App")

applicant_income = st.slider("Applicant's Income", value=500000)
coapplicant_income = st.slider("Coapplicant's Income", value=700000)
loan_amount = st.slider("Loan Amount", value=1000000)
loan_amount_term = st.slider("Loan Amount Term", value=360)
credit_history = st.selectbox("Credit History (1 for good, 0 for bad)", [1, 0])

st.header("Personal Information")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button('Predict'):
    input_data = {
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Gender': [gender],
        'Married': [married],
        'Education': [education],
        'Self_Employed': [self_employed],
        'Property_Area': [property_area]
    }

    input_df = pd.DataFrame(input_data)
    input_df = pd.get_dummies(input_df)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    if scaler:
        input_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = scaler.transform(
            input_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
        )

    st.write("Prepared Input Data:", input_df)

    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df) if hasattr(model, "predict_proba") else None

    if prediction_prob is not None:

        approval_probability = prediction_prob[0][1] if prediction_prob is not None else None
    st.write("Approval Probability:", f"{approval_probability * 100:.2f}%")

    threshold = 0.5
    if approval_probability and approval_probability > threshold:
        st.success("🎉 Congratulations! Your Loan Is Likely To Be Approved")
    else:
        st.error("❌ Unfortunately, Your Loan Is Likely To Be Rejected")
