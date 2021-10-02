#Core Pkgs
from pandas.core.arrays import categorical
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib


def display():
    st.title('Fill below')

def load_PreTrainedModelDetails():
    PreTrainedModelDetails = joblib.load("classification/models/leo_ml_classification.joblib")
    return PreTrainedModelDetails


    
def prediction(input_df):
    PreTrainedModelDetails = load_PreTrainedModelDetails()

    #Random Forest Classifier
    RandomForestClassifier = PreTrainedModelDetails.get('model')
    
    #PreFitted Encoder
    PreFittedEncoder = PreTrainedModelDetails.get('encoder')
    
    #PreFitted Scaler
    PreFittedScaler = PreTrainedModelDetails.get('scaler')
    
    numeric_cols = PreTrainedModelDetails.get('numeric_cols')
    
    categorical_cols = PreTrainedModelDetails.get('categorical_cols')
    
    encoded_cols = PreTrainedModelDetails.get('encoded_cols')
    
    train_score = PreTrainedModelDetails.get('train_score')
    
    val_score = PreTrainedModelDetails.get('val_score')
    
    input_df[encoded_cols] = PreFittedEncoder.transform(input_df[categorical_cols])
    input_df[numeric_cols] = PreFittedScaler.transform(input_df[numeric_cols])
    
    inputs_for_prediction = input_df[numeric_cols+encoded_cols]
    
    prediction = RandomForestClassifier.predict(inputs_for_prediction)
    
    st.write(input_df)
    
    st.write(prediction)
    
    
def get_user_input():
    form = st.form(key='user input form')
    employment_years = form.number_input("Employment Years",1,70,key='emp year')
    home_ownership = form.radio("Home Ownership",['Any','Mortgage','None','Other','Own','Rent'],key='home years')
    annual_inc= form.number_input("Annual Income",1000,100000000,key='annual inc')
    income_category = form.radio("Income Category",['Low','Medium','High'],key='in cat')
    dti = form.number_input("DTI",0,100,key='dti val')
    loan_amount = form.number_input("Loan Amount",1000,100000000,key='loan amt')
    grade = form.radio("Loan Grade",['A','B','C','D','E','F','G'],key='loan grd')
    application_type = form.radio("Application Type",['Individual','Joint'],key='appl typ')
    installment = form.number_input("Installment",1000,100000000,key='install')
    interest_rate = form.number_input("Interest Rate",0,100,key='ins rate')
    term = form.radio("Term",['36','60'],key='term val')
    purpose = form.radio("Purpose",['Car','Credit Card','Debt Consolidation','Educational','Home Improvement',
                                  'House','Major Purchase','Medical','Moving','Renewable Energy','Small Business','Vacation',
                                  'Wedding','Other'],key='purpose cat')
    
    submitButton = form.form_submit_button(label ='Predict Loan Grant Risk')

    if submitButton:
        
        st.write(purpose)
        SingleUserInput = {
            'employment_years':employment_years,
            'annual_inc':annual_inc,
            'loan_amount':loan_amount,
            'interest_rate':interest_rate,
            'dti':dti,
            'installment':installment,
            'home_ownership':home_ownership,
            'income_category':income_category,
            'term':term,
            'application_type':application_type,
            'purpose':purpose,
            'grade':grade,
            
        }
        
        st.write(pd.DataFrame([SingleUserInput]))
        return pd.DataFrame([SingleUserInput])
    
       


  
    
def main():
    display()    
    input_details_df = get_user_input()
    prediction(input_details_df)
  
    
    
if __name__ == '__main__':
    main()