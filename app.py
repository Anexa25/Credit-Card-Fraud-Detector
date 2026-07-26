import streamlit as st
import pandas as pd
from model import predict_transactions

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
        
        
        
        # Data going through model
        # predictions = []
        # probabilities = []
        
        
        # for _, row in data.iterrows():
        #     prediction, probability = predict_transaction(
        #         pd.DataFrame([row])
        #     )
            
        #     predictions.append(prediction)
        #     probabilities.append(probability)
        
        predictions, probabilities = predict_transactions(data)
            
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
        fraud_count = (predictions==1).sum()
        total_transactions = len(predictions)
        
        fraud_percentage = (fraud_count/total_transactions)*100
        
        
        #Risk calculation
        if fraud_percentage < 1:
            risk_level = "Low risk"
        elif fraud_percentage < 5:
            risk_level = "Medium risk"
        else:
            risk_level = "High risk"
        
        
        st.subheader("Summary")
        
        col1, col2, col3, col4= st.columns(4)
        
        with col1:
            st.metric(label="Total Transactions", 
                      value=f"{total_transactions:,}",
                      help="The total number of transactions analysed by the fraud detection model.")
            
            
        with col2:
            st.metric(label="Fraud Detected", 
                      value=fraud_count,
                      help="The number of transactions classified as potentially fraudulent by the model.")
            
        with col3:
            st.metric(label="Fraud Rate", 
                      value=f"{fraud_percentage:.2f}%",
                      help="The percentage of analysed transactions predicted to be fraudulent.")
            
        with col4:
            st.metric(label="Risk Level", 
                      value=risk_level,
                      help="An overall risk classification based on the percentage of transcations predicted as fraudulant.")
            
            
        #Download results
        
        csv = data.to_csv(index=False)
        
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )
        
    
    