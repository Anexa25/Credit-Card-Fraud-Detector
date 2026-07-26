import streamlit as st
import pandas as pd

st.title("Credit Card Fruad Detector")

st.write("""This application uses a Random Forest Machine Learning Model to detect
            potenitally fraudulant credit card transations.""")


uploaded_file = st.file_uploader(
    "Upload Transaction CSV file",
    type="csv"
)


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Data")
    
    st.dataframe(data.head())
    
    