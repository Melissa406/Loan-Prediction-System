import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#load the trained model
model_path = "svm_joblib"
model = joblib.load(model_path)

#define streamlit UI
st.title("Loan Prediction App")
st.write("This app predicts the likelihoods of s loan to be approved or rejected based on various factors")

#user input
Gender = st.selectbox("Gender", ["Male", "Female"])
Married	= st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education	= st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self_Employed", ["Yes", "No"])	
ApplicantIncome	= st.number_input("ApplicantIncome", min_value=0)
CoapplicantIncome = st.number_input("CoapplicantIncome", min_value=0)
LoanAmount = st.number_input("LoanAmount", min_value=0)
Loan_Amount_Term = st.number_input("Loan_Amount_Term", min_value=0)	
Credit_History = st.selectbox("Credit_History", ["0", "1"])
Property_Area = st.selectbox("Property_Area", ["Rural", "Urban", "Semi-Urban"])
Loan_Status = st.selectbox("Loan_Status", ["Yes", "No"])

#compute total income
Total_Income = ApplicantIncome + CoapplicantIncome

#convert categorical values to numerical values(manual Label Encodeing)
category_mappings ={
    "Gender": {"Male": 0, "Female": 1},
    "Married": {"Yes": 0, "No": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 0, "No": 1},
    "Property_Area": {"Rural": 0, "Urban": 1, "Semi-Urban":2}
}

#apply encoding
encoded_data = [
    category_mappings["Gender"][Gender],
    category_mappings["Married"][Married],
    category_mappings["Dependents"][Dependents],
    category_mappings["Education"][Education],
    category_mappings["Self_Employed"][Self_Employed],
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    category_mappings["Property_Area"][Property_Area],
    Total_Income
]

#convert to numpy array
input_data = np.array([encoded_data])

#standardize numerical values
scaler = StandardScaler()
numerical_indices = [5, 6, 7, 8, 10]
input_data[:, numerical_indices] = scaler.fit_transform(input_data[:, numerical_indices])

#predict on user input
if st.button("Predict on Loan Approval"):
    prediction = model.predict(input_data)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    st.success(f"Loan Status: {result}")