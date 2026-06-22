import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up page configuration
st.set_page_config(page_title="Retail Analytics Suite", layout="wide")

# --- LOAD ASSETS Safely ---
@st.cache_resource
def load_models():
    # Load your customer segmentation models
    model = joblib.load('final_kmeans_retail_model.pkl')
    scaler = joblib.load('retail_standard_scaler.pkl')
    
    # Load your product recommendation assets (Precomputed Cosine Similarity Matrix)
    # Note: Ensure you generate and upload 'product_similarity_matrix.pkl' and 'product_list.pkl'
    try:
        similarity_matrix = joblib.load('product_similarity_matrix.pkl')
        product_list = joblib.load('product_list.pkl')
    except FileNotFoundError:
        similarity_matrix, product_list = None, None
        
    return model, scaler, similarity_matrix, product_list

model, scaler, similarity_matrix, product_list = load_models()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
module_selection = st.sidebar.radio("Go to:", ["🎯 Product Recommendation", "👥 Customer Segmentation"])

# ==============================================================================
# 🎯 MODULE 1: PRODUCT RECOMMENDATION MODULE
# ==============================================================================
if module_selection == "🎯 Product Recommendation":
    st.title("🎯 Product Recommendation Module")
    st.write("Discover items frequently bought together or highly matching customer purchase behaviors using item-based collaborative filtering.")
    
    if similarity_matrix is None or product_list is None:
        st.warning("⚠️ Recommendation system files (`product_similarity_matrix.pkl` / `product_list.pkl`) not detected in your repository yet. Please build and commit them from your notebook.")
    else:
        # Text input box / select box for finding products
        selected_product = st.selectbox("Type or Select a Product Name:", product_list)
        
        if st.button("Get Recommendations"):
            # Fetch index of the selected product
            idx = list(product_list).index(selected_product)
            
            # Extract top 5 similarity scores (excluding the selected item itself)
            similarity_scores = list(enumerate(similarity_matrix[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            top_5_recommendations = similarity_scores[1:6]
            
            st.markdown("### ✨ Top 5 Recommended Products for You:")
            
            # Display recommendations nicely in styled columns
            cols = st.columns(5)
            for col_idx, (rec_idx, score) in enumerate(top_5_recommendations):
                with cols[col_idx]:
                    st.info(f"**{product_list[rec_idx]}**")
                    st.caption(f"Match Score: {score:.2%}")

# ==============================================================================
# 🎯 MODULE 2: CUSTOMER SEGMENTATION MODULE
# ==============================================================================
elif module_selection == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Module")
    st.write("Input behavior metrics below to predict exactly which business customer segment an individual or client profile occupies.")
    
    # 3 Number Inputs for RFM Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=365, value=30)
    with col2:
        frequency = st.number_input("Frequency (Total number of purchases)", min_value=1, max_value=500, value=5)
    with col3:
        monetary = st.number_input("Monetary (Total spending value in $)", min_value=0.1, max_value=100000.0, value=250.0)
        
    # Extra mandatory engineered structural feature from your project training matrix
    invoice_variety = st.number_input("Unique Items in Invoice (Variety Count)", min_value=1, value=4)

    if st.button("Predict Cluster"):
        # Preprocess input explicitly exactly how the training logic requires (log1p + scaled mapping)
        log_q = np.log1p(frequency)
        log_p = np.log1p(monetary / max(1, frequency)) # Derived average UnitPrice mapping proxy
        log_a = np.log1p(monetary)
        
        # Organize array data matrix using the exact original feature keys
        features = np.array([[log_q, log_p, log_a, invoice_variety]])
        scaled_features = scaler.transform(features)
        
        # Pass feature array headers safely back into a DataFrame placeholder to fix warnings
        feature_names = ['Log_Quantity', 'Log_UnitPrice', 'Log_TotalAmount', 'InvoiceVarietyCount']
        final_scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
        
        # Run prediction
        cluster_id = model.predict(final_scaled_df)[0]
        
        # Convert cluster IDs into clear human-interpretable business personas
        # Modify the numbers below to align with your final cluster analysis mapping!
        cluster_mapping = {
            0: ("🔴 At-Risk Customer", "Low frequency, old recency values. High risk of churn. Needs re-engagement incentives."),
            1: ("🟡 Occasional Shopper", "Moderate activity baseline. Standard retail transaction trends."),
            2: ("🟢 Regular Value Client", "Frequent orders, reliable purchase history. Solid core volume driver."),
            3: ("💎 High-Value Corporate/Wholesale", "Extreme total order valuations, high transaction frequency, large quantities.")
        }
        
        persona_name, persona_desc = cluster_mapping.get(cluster_id, (f"Cluster {cluster_id}", "Custom customer segment partition."))
        
        st.markdown("---")
        st.success(f"🎯 **Prediction Result:** The customer profile sits inside **{persona_name}**")
        st.write(f"💡 **Business Strategy:** {persona_desc}")
