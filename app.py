import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pickle
# 1. Load the trained ML pipeline
pipe =pickle.load(open("gradientboosting.pkl","rb"))
df=pickle.load(open("dataframe.pkl","rb"))
st.title("Loan Status Prediction App")

st.write(
    "Enter the applicant's details below to predict whether "
    "the loan will be approved or rejected."
)

# 2. Input fields

education = st.selectbox(
    "Education",
    options=[0, 1],
    format_func=lambda x: "Graduate" if x == 1 else "Not Graduate"
)

self_employed = st.selectbox(
    "Self Employed",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000,
    step=10000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1500000,
    step=50000
)

loan_term = st.number_input(
    "Loan Term",
    min_value=1,
    value=12,
    step=1
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=200000,
    step=10000
)

total_assets = st.number_input(
    "Total Assets",
    min_value=0,
    value=1000000,
    step=10000
)

# 3. Prediction

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame(
        [[
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            bank_asset_value,
            total_assets
        ]],

        columns=[
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "bank_asset_value",
            "total_assets"
        ]
    )

    try:
        # Actual model prediction
        prediction = pipe.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.write("Model Prediction:", prediction)

        # Display probability if supported
        if hasattr(pipe, "predict_proba"):

            probabilities = pipe.predict_proba(input_data)[0]

            st.write("Prediction Probabilities:")

            probability_df = pd.DataFrame({
                "Class": pipe.classes_,
                "Probability": probabilities
            })

            st.dataframe(probability_df)

    except Exception as e:
        st.error(f"Prediction Error: {e}")