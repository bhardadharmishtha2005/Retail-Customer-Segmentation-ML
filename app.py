import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page layout configuration to wide with a premium analytics icon
st.set_page_config(page_title="Enterprise Retail Intelligence Suite", page_icon="📈", layout="wide")

# Custom UI Styling to make elements look modern and professional
st.markdown("""
    <style>
    .metric-box {
        background-color: #1e222b;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4A90E2;
        margin-bottom: 15px;
    }
    .card-item {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHE DATA AND MODELS ASSETS ---
@st.cache_resource
def load_assets():
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

model, scaler, similarity_matrix, product_list = load_assets()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/fluent/96/000000/dashboard.png", width=70)
st.sidebar.title("Control Center")
st.sidebar.markdown("*AI-Driven Retail Operations*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate Workspace:", ["🏠 Home Overview", "👥 Customer Clustering", "🎯 Product Recommendations"])

# ==============================================================================
# 🏠 PAGE 1: HOME OVERVIEW
# ==============================================================================
if page == "🏠 Home Overview":
    st.title("🏠 Enterprise Retail Intelligence Suite")
    st.markdown("### `System Infrastructure Overview`")
    st.write("Welcome to the AI-driven retail command center. This system integrates unsupervised machine learning structures with real-time transactional collaborative pipelines.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.info("#### 👥 RFM Segmentation Module\nProcesses structural behavioral signals (Recency, Frequency, Monetary) through mathematical K-Means boundaries to classify distinct customer personas.")
    with col2:
        st.success("#### 🎯 Product Recommendation Engine\nUtilizes item-based Collaborative Filtering matrices and Cosine Similarity equations to generate automated cross-sell strategies.")

# ==============================================================================
# 👥 PAGE 2: CUSTOMER CLUSTERING
# ==============================================================================
elif page == "👥 Customer Clustering":
    st.title("👥 Advanced RFM Customer Segmentation")
    st.markdown("### `K-Means Mathematical Boundary Profiler`")
    st.write("Input structural operational behaviors below to map target account metrics into actionable enterprise cohorts.")
    
    with st.container():
        st.markdown("#### 📥 Live Operational Inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.number_input("Recency (Days Since Last Order)", min_value=0, max_value=365, value=30)
        with c2:
            frequency = st.number_input("Frequency (Total Transactions)", min_value=1, max_value=500, value=5)
        with c3:
            monetary = st.number_input("Monetary Value (Total Spend $)", min_value=0.1, max_value=100000.0, value=250.0)
            
    # Hardcoded/fallback index for structural array dimensioning
    invoice_variety = 4 

    st.markdown("---")
    if st.button("Predict Segment Profile", type="primary"):
        if model is None or scaler is None:
            st.error("❌ Model serialization files (`final_kmeans_retail_model.pkl` / `retail_standard_scaler.pkl`) not detected in repository.")
        else:
            # Replicate pipeline scaling preprocessing transformations
            log_q = np.log1p(frequency)
            log_p = np.log1p(monetary / max(1, frequency))
            log_a = np.log1p(monetary)
            
            features = np.array([[log_q, log_p, log_a, invoice_variety]])
            scaled_features = scaler.transform(features)
            
            feature_names = ['Log_Quantity', 'Log_UnitPrice', 'Log_TotalAmount', 'InvoiceVarietyCount']
            final_scaled_df = pd.DataFrame(scaled_features, columns=feature_names)
            
            cluster_id = model.predict(final_scaled_df)[0]
            
            # Premium Visual Profile Mapping
            cluster_mapping = {
                0: ("🔴 Churn Risk Account", "error", "Low order activity with trailing operational dates.", "Deploy win-back automated marketing emails and clear slow-moving inventory items."),
                1: ("🟡 Casual Occasional Shopper", "warning", "Standard retail consumer baseline with erratic transactional frequencies.", "Introduce tiered loyalty program milestones and push targeted notifications."),
                2: ("🟢 Core Regular Value Client", "info", "Predictable baseline order behaviors, high margin consistency, solid account health.", "Provide priority early access to inventory drops and dedicated support structures."),
                3: ("💎 Premium Enterprise Wholesale", "success", "Massive transaction sizes, scaling wholesale distribution totals.", "Assign dedicated account contract management and premium logistical delivery priority.")
            }
            
            title, alert_type, brief, strategy = cluster_mapping.get(cluster_id, (f"Cluster {cluster_id}", "info", "Identified custom split sequence.", "N/A"))
            
            st.markdown("### 📊 Engine Performance Diagnostics")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Assigned Cohort ID", f"Cluster {cluster_id}")
            m_col2.metric("Profile Spend Density", f"${monetary:,.2f}")
            m_col3.metric("Calculated Baseline Group", title)
            
            if alert_type == "error": st.error(f"#### **Profile Target: {title}**\n\n**Behavioral Baseline Summary:** {brief}")
            elif alert_type == "warning": st.warning(f"#### **Profile Target: {title}**\n\n**Behavioral Baseline Summary:** {brief}")
            elif alert_type == "success": st.success(f"#### **Profile Target: {title}**\n\n**Behavioral Baseline Summary:** {brief}")
            else: st.info(f"#### **Profile Target: {title}**\n\n**Behavioral Baseline Summary:** {brief}")
            
            st.markdown(f"""
            <div class="metric-box">
                <h4 style='color:#4A90E2;margin-top:0;'>🚀 Strategic Operation Guideline:</h4>
                <p style='color:#c9d1d9;font-size:15px;margin-bottom:0;'>{strategy}</p>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# 🎯 PAGE 3: PRODUCT RECOMMENDATIONS
# ==============================================================================
elif page == "🎯 Product Recommendations":
    st.title("🎯 Item Recommendation Engine")
    st.markdown("### `Item-Based Collaborative Filtering Model`")
    st.write("Scan product inventory pairings to generate optimized cross-selling predictions instantly.")
    
    if similarity_matrix is None or product_list is None:
        st.warning("⚠️ Recommendation asset files (`product_similarity_matrix.pkl` / `product_list.pkl`) are missing from GitHub.")
    else:
        selected_product = st.selectbox("Type or Select a Target Product Name:", product_list)
        
        if st.button("Generate Recommendations", type="primary"):
            top_10_recommendations = similarity_matrix.get(selected_product, [])
            top_5_recommendations = top_10_recommendations[:5]
            
            if not top_5_recommendations:
                st.info("No definitive product pairings detected for this selection.")
            else:
                st.markdown("### ✨ Top 5 Optimized Cross-Sell Recommendations:")
                cols = st.columns(5)
                for col_idx, (rec_idx, score) in enumerate(top_5_recommendations):
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="card-item">
                            <p style='font-size:26px;margin:0;'>📦</p>
                            <h5 style='font-size:13px;color:#e6edf3;height:55px;overflow:hidden;'>{product_list[rec_idx]}</h5>
                            <span style='background-color:#1f6feb;color:white;padding:4px 10px;border-radius:12px;font-size:11px;'>Match: {score:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
