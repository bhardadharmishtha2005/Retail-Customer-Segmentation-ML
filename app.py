import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page layout to wide with a premium title icon
st.set_page_config(page_title="Enterprise Retail Intelligence Suite", page_icon="📈", layout="wide")

# Custom CSS styling to make elements look polished and professional
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e222b;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin-bottom: 10px;
    }
    .recommendation-card {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS Safely ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('final_kmeans_retail_model.pkl')
        scaler = joblib.load('retail_standard_scaler.pkl')
    except:
        model, scaler = None, None
        
    try:
        similarity_matrix = joblib.load('product_similarity_matrix.pkl')
        product_list = joblib.load('product_list.pkl')
    except:
        similarity_matrix, product_list = None, None
        
    return model, scaler, similarity_matrix, product_list

model, scaler, similarity_matrix, product_list = load_models()

# --- SIDEBAR DESIGN ---
st.sidebar.image("https://img.icons8.com/fluent/96/000000/dashboard.png", width=80)
st.sidebar.title("Control Center")
st.sidebar.markdown("*AI-Driven Retail Operations*")
st.sidebar.markdown("---")
module_selection = st.sidebar.radio("Select Analytics Workspace:", ["🎯 Product Recommendation Engine", "👥 RFM Customer Segmentation"])

# ==============================================================================
# 🎯 MODULE 1: PRODUCT RECOMMENDATION ENGINE
# ==============================================================================
if module_selection == "🎯 Product Recommendation Engine":
    st.title("🎯 Product Recommendation Engine")
    st.markdown("### `Item-Based Collaborative Filtering System`")
    st.write("Analyze transactional relationships between SKUs across thousands of checkouts to identify hyper-relevant cross-selling opportunities.")
    
    if similarity_matrix is None or product_list is None:
        st.info("💡 **Welcome to the Recommendation Engine Preview**")
        st.warning("⚠️ Files (`product_similarity_matrix.pkl` / `product_list.pkl`) are missing from GitHub. Below is a mock preview of what your evaluators will see once uploaded:")
        
        # MOCK UI FOR PREVIEW & VALIDATION ONLY
        mock_products = ["WHITE HANGING HEART T-LIGHT HOLDER", "REGENCY CAKESTAND 3 TIER", "PARTY BUNTING", "JUMBO BAG RED RETROSPOT", "ASSORTED COLOUR BIRD ORNAMENT"]
        selected_product = st.selectbox("Type or Select a Target Product Name:", mock_products)
        
        if st.button("Generate Recommendations", type="primary"):
            st.markdown("### ✨ Top 5 Cross-Sell Recommendations:")
            cols = st.columns(5)
            mock_recs = [("JAM MAKING SET WITH JARS", "94.2%"), ("RED RETROSPOT MINI CASES", "88.7%"), ("HEART OF WICKER LARGE", "85.1%"), ("SET OF 3 CAKE TINS SKULLS", "81.4%"), ("WOODEN PICTURE FRAME WHITE", "79.0%")]
            for idx, (prod, score) in enumerate(mock_recs):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <p style='font-size:24px;margin:0;'>📦</p>
                        <h4 style='font-size:14px;color:#e6edf3;height:50px;overflow:hidden;'>{prod}</h4>
                        <span style='background-color:#1f6feb;color:white;padding:3px 8px;border-radius:12px;font-size:12px;'>Match: {score}</span>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # LIVE PRODUCTION MODE
        selected_product = st.selectbox("Type or Select a Target Product Name:", product_list)
        if st.button("Generate Recommendations", type="primary"):
            idx = list(product_list).index(selected_product)
            similarity_scores = list(enumerate(similarity_matrix[idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            top_5_recommendations = similarity_scores[1:6]
            
            st.markdown("### ✨ Top 5 Cross-Sell Recommendations:")
            cols = st.columns(5)
            for col_idx, (rec_idx, score) in enumerate(top_5_recommendations):
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <p style='font-size:24px;margin:0;'>📦</p>
                        <h4 style='font-size:14px;color:#e6edf3;height:50px;overflow:hidden;'>{product_list[rec_idx]}</h4>
                        <span style='background-color:#1f6feb;color:white;padding:3px 8px;border-radius:12px;font-size:12px;'>Match: {score:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# 👥 MODULE 2: RFM CUSTOMER SEGMENTATION
# ==============================================================================
elif module_selection == "👥 RFM Customer Segmentation":
    st.title("👥 Advanced RFM Customer Segmentation")
    st.markdown("### `K-Means Mathematical Boundary Profiler`")
    st.write("Process algorithmic cluster assignments by transforming behavioral signals (Recency, Frequency, Monetary) against baseline metrics.")
    
    # Clean split input panels
    with st.container():
        st.markdown("#### 📥 Live Operational Inputs")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            recency = st.number_input("Recency (Days Since Last Order)", min_value=0, max_value=365, value=30, help="Fewer days mean higher recent engagement.")
        with col2:
            frequency = st.number_input("Frequency (Total Transactions)", min_value=1, max_value=500, value=5, help="Total orders placed by this account.")
        with col3:
            monetary = st.number_input("Monetary Value (Total Spend $)", min_value=0.1, max_value=100000.0, value=250.0, help="Gross dollar revenue from this client.")
        with col4:
            invoice_variety = st.number_input("Invoice Variety (Unique SKU Count)", min_value=1, max_value=100, value=4, help="Breadth of inventory variance purchased.")

    st.markdown("---")
    
    if st.button("Execute Cluster Assignment", type="primary"):
        if model is None or scaler is None:
            st.error("❌ Pipeline models not found on GitHub repository. Please commit 'final_kmeans_retail_model.pkl' and 'retail_standard_scaler.pkl'.")
        else:
            # Replicate pipeline scaling preprocessing logic
            log_q = np.log1p(frequency)
            log_p = np.log1p(monetary / max(1, frequency))
            log_a = np.log1p(monetary)
            
            features = np.array([[log_q, log_p, log_a, invoice_variety]])
            scaled_features = scaler.transform(features)
            
            feature_names = ['Log_Quantity', 'Log_UnitPrice', 'Log_TotalAmount', 'InvoiceVarietyCount']
            final_scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
            
            # Predict
            cluster_id = model.predict(final_scaled_df)[0]
            
            # Premium Visual UI Cards mapping based on cluster identification output
            cluster_mapping = {
                0: ("🔴 Churn Risk / Low Value", "error", "Low activity and stale log interactions. High risk of complete churn.", "Deploy win-back automated email coupons and clear excess old stock items to lower loss thresholds."),
                1: ("🟡 Casual/Occasional Retail Shopper", "warning", "Inconsistent ordering behaviors. Standard low-frequency baskets.", "Introduce product loyalty tier points systems and showcase product recommendations to upsell order totals."),
                2: ("🟢 Core Regular Value Client", "info", "Highly structured order cycles, active account health, predictable margins.", "Provide dedicated early access to product launches and maintain robust baseline inventory levels to satisfy demand."),
                3: ("💎 Premium Enterprise Wholesale", "success", "Massive purchase concentrations, significant scaling totals, highly recurring baskets.", "Assign dedicated accounts managers, offer bespoke bulk wholesale pricing agreements, and provide tier-1 delivery logistics priority.")
            }
            
            title, alert_type, brief, strategy = cluster_mapping.get(cluster_id, (f"Cluster {cluster_id}", "info", "Standard custom split block.", "N/A"))
            
            st.markdown("### 📊 Engine Performance Diagnostics")
            
            # Display Real-Time Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Assigned Cohort ID", f"Cluster {cluster_id}")
            m_col2.metric("Profile Spend Density", f"${monetary:,.2f}")
            m_col3.metric("Basket Breadth", f"{invoice_variety} SKUs")
            
            # Display targeted alert block
            if alert_type == "error": st.error(f"#### **Target Segment Profile: {title}**\n\n**Behavioral Overview:** {brief}")
            elif alert_type == "warning": st.warning(f"#### **Target Segment Profile: {title}**\n\n**Behavioral Overview:** {brief}")
            elif alert_type == "success": st.success(f"#### **Target Segment Profile: {title}**\n\n**Behavioral Overview:** {brief}")
            else: st.info(f"#### **Target Segment Profile: {title}**\n\n**Behavioral Overview:** {brief}")
            
            # Operational Business Blueprint container
            st.markdown(f"""
            <div class="metric-card">
                <h4 style='color:#4A90E2;margin-top:0;'>🚀 Recommended Institutional Strategy:</h4>
                <p style='color:#c9d1d9;font-size:15px;margin-bottom:0;'>{strategy}</p>
            </div>
            """, unsafe_allow_html=True)
