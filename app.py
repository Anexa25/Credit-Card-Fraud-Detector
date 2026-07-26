import streamlit as st
import pandas as pd
from model import predict_transaction

st.title("Credit Card Fruad Detector")

st.write("""This application uses a Random Forest Machine Learning Model to detect
            potenitally fraudulant credit card transations.""")


uploaded_file = st.file_uploader(
    "Upload Transaction CSV file",
    type="csv"
)


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)
    
    st.subheader("Uploaded Data")
    
    st.dataframe(data.head())
    
    
    
    if st.button("Analyse Transactions"):
        st.info("Loading model and analysing transactions")
        
        predictions = []
        probabilities = []
        
        
        for _, row in data.iterrows():
            prediction, probability = predict_transaction(
                pd.DataFrame([row])
            )
            
            predictions.append(prediction)
            probabilities.append(probability)
            
        data["Predictions"] = predictions
        data["Fraud Probability"] = probabilities
        
        
        st.subheader("Results")
        st.dataframe(data)
        
    
    