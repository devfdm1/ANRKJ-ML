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

import uuid

def display():
    st.title('Fill below')

def load_PreTrainedModelDetails():
    PreTrainedModelDetails = joblib.load("models/classification/randomForestClassifier.joblib")
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
    
    input_df[encoded_cols] = PreFittedEncoder.transform(input_df[categorical_cols])
    input_df[numeric_cols] = PreFittedScaler.transform(input_df[numeric_cols])
    
    inputs_for_prediction = input_df[numeric_cols+encoded_cols]
    
    prediction = RandomForestClassifier.predict(inputs_for_prediction)
    
    st.write(input_df)
    
    st.write(prediction)
    
    
    
    
       
    
def getUserInput():
    input_income = st.number_input("Income",15000,1000000,key=uuid.uuid4())
    age = st.number_input("Age",18,70,key=uuid.uuid4())
    marital_status = st.radio("Marital Status",['Married','Single'],key=uuid.uuid4())
    profession = st.text_input("Profession",key=uuid.uuid4())
    current_job_yrs = st.number_input("Years of current job",1,50,key=uuid.uuid4())
    experience = st.number_input("Years of experience",1,50,key=uuid.uuid4())
    house_ownership = st.radio("House Ownership",['No','Yes'],key=uuid.uuid4())
    current_house_yrs = st.number_input("Years of house ",1,50,key=uuid.uuid4())
    car_ownership = st.radio("Car Ownership",['Yes','No'],key=uuid.uuid4())
    
    SingleUserInput = {
        'Income':input_income,
        'Age':age,
        'Experience':experience,
        'Married/Single':marital_status,
        'House_Ownership':house_ownership,
        'Car_Ownership':car_ownership,
        'Profession':profession,
        'CURRENT_JOB_YRS':current_job_yrs,
        'CURRENT_HOUSE_YRS':experience
        
    }
    
    return pd.DataFrame([SingleUserInput])

  
    
def main():
    display()
    input_df = getUserInput()
    if st.button("Predict Loan Grant Risk"):
        prediction(input_df)
    
if __name__ == '__main__':
    main()