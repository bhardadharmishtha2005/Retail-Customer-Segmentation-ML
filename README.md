# Retail-Customer-Segmentation-ML 📈

An AI-driven retail operations command center that bridges unsupervised machine learning and automated cross-selling pipelines. This platform optimizes customer lifetime value (LTV) by providing businesses with advanced customer segmentation and intelligent real-time product recommendations.

🌐 **Live Application:** [View Live Streamlit App](https://retail-customer-segmentation-ml-vuqhvkcyhhuw9mxcbqrtvv.streamlit.app/)

---

## 🚀 Key Platform Features

### 👥 1. Advanced RFM Customer Segmentation
* **Objective:** Processes behavioral customer footprints to classify accounts into actionable enterprise cohorts.
* **Functionality:** Accepts real-time numerical inputs for **Recency** (days since last order), **Frequency** (total transactions), and **Monetary Value** (total spend).
* **Algorithmic Engine:** Uses a K-Means Mathematical Boundary Profiler paired with Log-Transformation data-scaling pipelines.
* **Output:** Instantly assigns customer profiles to targeted behavioral groups (e.g., *Premium Enterprise Wholesale*, *Core Regular Value*, *Casual Shopper*, *Churn Risk*) along with institutional marketing strategies.

### 🎯 2. Item Recommendation Engine
* **Objective:** Scans historical transaction trends to identify hyper-relevant cross-selling opportunities across inventory SKUs.
* **Functionality:** High-speed text selection dropdown for products.
* **Algorithmic Engine:** Uses an item-based Collaborative Filtering system utilizing Cosine Similarity matrices, compressed into optimized top-10 neighbor lookups for lightning-fast cloud delivery (< 12ms).
* **Output:** Displays the top 5 matching cross-sell items inside a clean, modern card view grid showing exact percentage match scores.

---

## 🛠️ Technical Architecture & Tags

`Pandas` `NumPy` `DataCleaning` `FeatureEngineering` `EDA` `RFMAnalysis` `CustomerSegmentation` `KMeansClustering` `CollaborativeFiltering` `CosineSimilarity` `ProductRecommendation` `ScikitLearn` `StandardScaler` `StreamlitApp` `MachineLearning` `DataVisualization` `PivotTables` `DataTransformation` `RealTimePrediction`

---

## 📁 Repository Structure

```text
├── app.py                             # Main Streamlit web application source code
├── final_kmeans_retail_model.pkl      # Serialized K-Means clustering model 
├── retail_standard_scaler.pkl         # Trained StandardScaler preprocessing matrix
├── product_similarity_matrix.pkl      # Compressed Top-10 neighborhood similarity dictionary
├── product_list.pkl                   # Unique product inventory index mapping
├── github_retail_data.csv             # Lightweight optimized dataset for repo storage
├── requirements.txt                   # Platform deployment dependencies
└── README.md                          # System documentation
