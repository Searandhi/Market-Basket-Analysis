import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime, timedelta

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------------------------------------------
# APP CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Retail Market Basket Intelligence", layout="wide", page_icon="🛒")
st.title("🛍️ Retail Market Basket Intelligence Dashboard")
st.caption("AI-powered insights for demand, behavior, cross-sell combos, and financial forecasting")

# -----------------------------------------------------------
# LOAD MODELS & DATA
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        if os.path.exists("models/product_demand_rf.pkl"):
            models["product_demand"] = joblib.load("models/product_demand_rf.pkl")
            st.sidebar.success("✅ Product Demand model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load model: {e}")
    return models

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/market_basket_cleaned_with_ids_regenerated.csv")
        rules = pd.read_csv("outputs/association_rules_manual.csv") if os.path.exists("outputs/association_rules_manual.csv") else pd.DataFrame()
        return df, rules
    except Exception as e:
        st.warning(f"⚠️ Could not load data: {e}")
        return pd.DataFrame(), pd.DataFrame()

models = load_models()
df, rules = load_data()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
mode = st.sidebar.radio("Select:", [
    "Dashboard Overview",
    "Product Demand Prediction",
    "Customer Behaviour Prediction",
    "Cross-Selling Prediction",
    "Sales Forecasting",
    "MongoDB Analytics"
])

# -----------------------------------------------------------
# DASHBOARD OVERVIEW
# -----------------------------------------------------------
if mode == "Dashboard Overview":
    st.subheader("📊 Dataset Overview")
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", len(df["Transaction_ID"].unique()))
        col2.metric("Customers", len(df["Customer_ID"].unique()))
        col3.metric("Products", len([c for c in df.columns if c.startswith("Product_")]))
        
        st.write("### Recent Transactions")
        st.dataframe(df.head(10), use_container_width=True)
        
        if not rules.empty:
            st.markdown("### 🔗 Association Rules (Cross-Sell)")
            st.dataframe(rules.head(10), use_container_width=True)
        else:
            st.info("ℹ️ No association rules found yet.")
    else:
        st.error("❌ No data loaded. Check if CSV files exist in `data/` folder.")

# -----------------------------------------------------------
# PRODUCT DEMAND PREDICTION
# -----------------------------------------------------------
elif mode == "Product Demand Prediction":
    st.subheader("📦 Product Demand Prediction")

    if "product_demand" not in models:
        st.error("❌ Product Demand model not loaded. Train the model first.")
    else:
        model = models["product_demand"]
        
        # Get model feature requirements
        n_features = model.n_features_in_ if hasattr(model, "n_features_in_") else 24
        
        st.info(f"🔧 Model Input: **{n_features} features**")
        
        # Create feature array
        input_features = np.zeros(n_features)
        
        col1, col2 = st.columns(2)
        with col1:
            qty = st.number_input("📦 Quantity", min_value=1, value=10, step=1)
        with col2:
            amt = st.number_input("💰 Total Amount (₹)", min_value=10.0, value=500.0, step=10.0)
        
        cross = st.slider("🔗 Cross-Sell Potential", 0.0, 1.0, 0.5)
        
        # Fill feature array
        input_features[0] = qty
        input_features[1] = amt
        input_features[2] = cross
        input_features[3] = datetime.now().month
        
        if st.button("🔮 Predict Demand", use_container_width=True):
            try:
                input_data = pd.DataFrame([input_features])
                prediction = model.predict(input_data)[0]
                st.success(f"✅ **Predicted Demand: {prediction:.2f} units**")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# -----------------------------------------------------------
# CUSTOMER BEHAVIOUR
# -----------------------------------------------------------
elif mode == "Customer Behaviour Prediction":
    st.subheader("🧠 Customer Behaviour Prediction")
    if not df.empty:
        cust_ids = sorted(df["Customer_ID"].unique())
        selected_cust = st.selectbox("👤 Select Customer ID", cust_ids)
        
        cust_data = df[df["Customer_ID"] == selected_cust]
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", len(cust_data))
        col2.metric("Total Spent", f"₹{cust_data['Total_Amount'].sum():.2f}")
        col3.metric("Avg Purchase", f"₹{cust_data['Total_Amount'].mean():.2f}")
        
        st.write("### Customer Purchase History")
        st.dataframe(cust_data, use_container_width=True)
    else:
        st.error("❌ No customer data available.")

# -----------------------------------------------------------
# CROSS-SELLING RECOMMENDATIONS
# -----------------------------------------------------------
elif mode == "Cross-Selling Prediction":
    st.title("🛒 Cross-Selling Recommendations")
    st.write("Discover which products are frequently bought together — powered by Association Rules Mining.")
    
    OUTPUT_DIR = "outputs"
    rules_path = os.path.join(OUTPUT_DIR, "association_rules_manual.csv")
    
    if os.path.exists(rules_path):
        try:
            cross_rules = pd.read_csv(rules_path)
            st.dataframe(cross_rules.head(20), use_container_width=True)
            st.success(f"✅ **Found {len(cross_rules)} association rules**")
        except Exception as e:
            st.error(f"❌ Error reading rules: {e}")
    else:
        st.error("❌ No association rules file found in `outputs/` folder.")

# -----------------------------------------------------------
# SALES FORECASTING
# -----------------------------------------------------------
elif mode == "Sales Forecasting":
    st.subheader("📊 Store-wise Sales Forecasting")
    
    if os.path.exists("models"):
        store_models = [f for f in os.listdir("models") if f.startswith("sales_forecast_") and f.endswith(".pkl")]
        
        if not store_models:
            st.warning("⚠️ No forecasting models found. Train models first.")
        else:
            store_names = [f.replace("sales_forecast_", "").replace(".pkl", "").capitalize() for f in store_models]
            selected_store = st.selectbox("🏬 Select Store", store_names)
            forecast_days = st.slider("📅 Forecast next (days)", 7, 60, 30)
            
            st.success(f"✅ Model loaded for **{selected_store}**")
            st.info(f"📈 Forecasting sales for next **{forecast_days}** days")
    else:
        st.error("❌ Models folder not found.")

# -----------------------------------------------------------
# MONGODB ANALYTICS
# -----------------------------------------------------------
elif mode == "MongoDB Analytics":
    st.subheader("📊 MongoDB Analytics Dashboard")
    st.caption("Live insights from your MongoDB database")
    
    try:
        from pymongo import MongoClient
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()
        
        db = client["market_basket_ai"]
        st.success("✅ **Connected to MongoDB**")
        
        collections = db.list_collection_names()
        
        if collections:
            selected_collection = st.selectbox("📚 Choose Collection", collections)
            
            data = list(db[selected_collection].find({}, {"_id": 0}).limit(100))
            
            if data:
                df_mongo = pd.DataFrame(data)
                st.write(f"### 🧾 Data: `{selected_collection}` ({len(data)} records)")
                st.dataframe(df_mongo.head(10), use_container_width=True)
            else:
                st.info("📭 No data in this collection")
        else:
            st.warning("⚠️ No collections found in MongoDB")
            
    except Exception as e:
        st.error(f"❌ MongoDB connection error: {e}")
        st.info("💡 Make sure MongoDB is running on `localhost:27017`")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("📊 Developed for DWDM Project — Market Basket AI Dashboard (Stable Version)")