import streamlit as st
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Load the pre-trained model
try:
    model = pickle.load(open('./Model/ML_Model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found! Please ensure the file path is correct.")
    st.stop()

def run():
    # Display logo and title
    img1 = Image.open('bank.png')
    img1 = img1.resize((156, 145))
    st.sidebar.image(img1, use_column_width=True)
    st.sidebar.title("About")
    st.sidebar.info("This application predicts loan eligibility based on user-provided details using a machine learning model.")

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Bank Loan Prediction System</h1>", unsafe_allow_html=True)

    # Input fields with modern design
    st.markdown("### Please Fill the Below Details")

    account_no = st.text_input('Account Number', help="Enter your unique bank account number.")
    fn = st.text_input('Full Name', help="Enter your full name.")

    gen_display = ('Female', 'Male')
    gen = st.selectbox("Gender", range(len(gen_display)), format_func=lambda x: gen_display[x])

    mar_display = ('No', 'Yes')
    mar = st.selectbox("Marital Status", range(len(mar_display)), format_func=lambda x: mar_display[x])

    dep_display = ('No', 'One', 'Two', 'More than Two')
    dep = st.selectbox("Dependents", range(len(dep_display)), format_func=lambda x: dep_display[x])

    edu_display = ('Not Graduate', 'Graduate')
    edu = st.selectbox("Education", range(len(edu_display)), format_func=lambda x: edu_display[x])

    emp_display = ('Job', 'Business')
    emp = st.selectbox("Employment Status", range(len(emp_display)), format_func=lambda x: emp_display[x])

    prop_display = ('Rural', 'Semi-Urban', 'Urban')
    prop = st.selectbox("Property Area", range(len(prop_display)), format_func=lambda x: prop_display[x])

    cred_display = ('Between 300 to 500', 'Above 500')
    cred = st.selectbox("Credit Score", range(len(cred_display)), format_func=lambda x: cred_display[x])

    mon_income = st.number_input("Applicant's Monthly Income($)", value=0, help="Enter your monthly income in USD.")
    co_mon_income = st.number_input("Co-Applicant's Monthly Income($)", value=0, help="Enter your co-applicant's monthly income in USD.")
    loan_amt = st.number_input("Loan Amount", value=0, help="Enter the loan amount you are applying for.")

    dur_display = ['2 Months', '6 Months', '8 Months', '1 Year', '16 Months']
    dur = st.selectbox("Loan Duration", range(len(dur_display)), format_func=lambda x: dur_display[x])

    # Loan duration mapping
    duration_mapping = [60, 180, 240, 360, 480]
    duration = duration_mapping[dur]

    if st.button("Submit"):
        if not account_no.isdigit():
            st.error("Account number must be numeric.")
            return
        if not fn.strip():
            st.error("Full Name cannot be empty.")
            return

        # Prepare input features
        features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)[0]

        # Display prediction results
        if prediction[0] == 0:
            st.error(
                f"Hello: {fn} || Account number: {account_no} || According to our calculations, you will not get the loan from the bank."
            )
        else:
            st.success(
                f"Hello: {fn} || Account number: {account_no} || Congratulations! You will get the loan from the bank."
            )

        # Visualization of prediction probability with enhanced design
        st.markdown("### Prediction Probability")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(['Not Approved', 'Approved'], prediction_prob, color=['#ff9999', '#66b3ff'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Loan Approval Probability")
        st.pyplot(fig)

        # Display statistical summary as a table
        st.markdown("### Statistical Summary")
        data = {
            'Feature': ['Gender', 'Marital Status', 'Dependents', 'Education', 'Employment', 'Applicant Income', 'Co-Applicant Income', 'Loan Amount', 'Loan Duration', 'Credit Score', 'Property Area'],
            'Value': [
                gen_display[gen], mar_display[mar], dep_display[dep], edu_display[edu], emp_display[emp],
                f"${mon_income}", f"${co_mon_income}", f"${loan_amt}", dur_display[dur],
                cred_display[cred], prop_display[prop]
            ]
        }
        df = pd.DataFrame(data)
        st.table(df)

run()
