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
        
        
        # Replaces 1 & 0 with Fraud & Legitimate
        data["Predictions"] = data["Predictions"].map({
            1 : "Fraud Detected",
            0 : "Legitimate Transaction"
        })
        
        
        
        # Fraud probability as percentage
        data["Fraud Probability"] = (
            data["Fraud Probability"] * 100
        ).round(2).astype(str) + "%"
        
        
        
        st.subheader("Results")
        st.dataframe(data)
        
        
        
        #Summary table
        fraud_count = predictions.count(1)
        total_transactions = len(predictions)
        
        fraud_percentage = (fraud_count/total_transactions)*100
        
        st.subheader("Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", total_transactions)
            
        with col2:
            st.metric("Fraud Detected", fraud_count)
            
        with col3:
            st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
            
            
        #Download results
        
        csv = data.to_csv(index=False)
        
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_detection_results",
            mime="text/csv"
        )
        
    
    