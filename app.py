import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('final_kmeans_retail_model.pkl')
scaler = joblib.load('retail_standard_scaler.pkl')

st.title("🛍️ Retail Customer Segmentation Tool")
st.write("Enter the customer's transaction metrics below to predict their segment.")

# User inputs
quantity = st.number_input("Quantity Ordered", min_value=1, value=10)
unit_price = st.number_input("Unit Price ($)", min_value=0.1, value=2.5)
total_amount = quantity * unit_price
invoice_variety = st.number_input("Unique Items in Invoice", min_value=1, value=3)

st.write(f"**Calculated Total Amount:** ${total_amount:.2f}")

if st.button("Predict Customer Cluster"):
    # Preprocess inputs exactly like training
    log_q = np.log1p(quantity)
    log_p = np.log1p(unit_price)
    log_a = np.log1p(total_amount)
    
    features = np.array([[log_q, log_p, log_a, invoice_variety]])
    scaled_features = scaler.transform(features)
    
    # Predict
    cluster = model.predict(scaled_features)[0]
    
    st.success(f"🎯 This customer belongs to **Cluster {cluster}**")
