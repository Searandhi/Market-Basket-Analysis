import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# =========================
# 🎯 APP CONFIG
# =========================
st.set_page_config(
    page_title="Retail Market Basket Intelligence Dashboard",
    layout="wide",
    page_icon="🛒"
)

st.title("🛍️ Retail Market Basket Intelligence Dashboard")
st.markdown("Gain insights into product demand, customer behavior, cross-sell opportunities, and sales forecasts — all powered by your trained ML models.")

# =========================
# 📦 LOAD MODELS AND DATA
# =========================
@st.cache_resource
def load_models():
    models = {}
    try:
        models["product_demand_rf"] = pickle.load(open("models/product_demand_rf.pkl", "rb"))
        models["customer_lookup"] = pickle.load(open("models/customer_top_product_lookup.pkl", "rb"))
        models["customer_behavior_lr"] = pickle.load(open("models/customer_behavior_lr.pkl", "rb"))
        models["sales_forecast_gbr"] = pickle.load(open("models/sales_forecast_gbr.pkl", "rb"))
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

@st.cache_data
def load_data():
    df = pd.read_csv("data/market_basket_cleaned_with_ids_regenerated.csv")
    rules = pd.read_csv("outputs/association_rules_manual.csv")
    return df, rules

models = load_models()
df, rules = load_data()

# =========================
# 🔍 SIDEBAR
# =========================
st.sidebar.header("🔎 Explore Options")
mode = st.sidebar.radio("Select View:", [
    "Dashboard Overview",
    "Predict Product Demand",
    "Customer Insights",
    "Cross-Sell Explorer",
    "Sales Forecasting"
])

# =========================
# 📊 DASHBOARD OVERVIEW
# =========================
if mode == "Dashboard Overview":
    st.subheader("📈 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions", len(df["Transaction_ID"].unique()))
    col2.metric("Customers", len(df["Customer_ID"].unique()))
    col3.metric("Products", len([c for c in df.columns if c.startswith("Product_")]))

    st.dataframe(df.head(10))

    st.markdown("### 🔗 Association Rules Summary")
    st.dataframe(rules.head(10))

# =========================
# 🤖 PRODUCT DEMAND PREDICTION
# =========================
elif mode == "Predict Product Demand":
    st.subheader("🎯 Predict Product Demand")
    numeric_cols = ['Quantity', 'Price_per_Unit', 'Discount', 'Product_Demand_Score']
    input_data = {}
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col}", min_value=0.0, max_value=100.0, value=10.0)
    df_input = pd.DataFrame([input_data])

    if st.button("Predict Demand"):
        model = models["product_demand_rf"]
        prediction = model.predict(df_input)[0]
        st.success(f"📦 Predicted Product Demand: {'High' if prediction==1 else 'Low'}")

# =========================
# 👥 CUSTOMER INSIGHTS
# =========================
elif mode == "Customer Insights":
    st.subheader("🧍 Customer Purchase Pattern")

    cust_ids = df["Customer_ID"].dropna().unique()
    customer_id = st.selectbox("Select Customer ID", sorted(cust_ids))

    lookup = models["customer_lookup"]
    if customer_id in lookup:
        product = lookup[customer_id]
        st.info(f"🛒 Last purchased / most likely product: **{product}**")
    else:
        st.warning("No record found for this customer.")

# =========================
# 🔁 CROSS-SELL EXPLORER
# =========================
elif mode == "Cross-Sell Explorer":
    st.subheader("🔁 Cross-Sell Recommendations")
    product = st.selectbox("Select a Product", rules["Product_A"].unique())
    recs = rules[rules["Product_A"] == product].sort_values(by="Confidence", ascending=False)

    if recs.empty:
        st.warning("No cross-sell suggestions available for this product.")
    else:
        st.write("Recommended complementary products:")
        st.dataframe(recs[["Product_B", "Support", "Confidence", "Lift"]].reset_index(drop=True))

# =========================
# 💰 SALES FORECASTING
# =========================
elif mode == "Sales Forecasting":
    st.subheader("📉 Sales Forecast Projection")
    model = models["sales_forecast_gbr"]

    # Create dummy data for visualization (industry demo)
    future_months = np.arange(1, 7)
    features = pd.DataFrame({
        "Month": future_months,
        "Total_Amount": np.linspace(1000, 5000, len(future_months))
    })

    forecast = model.predict(features)
    forecast_df = pd.DataFrame({
        "Month": future_months,
        "Predicted_Sales": forecast
    })

    st.line_chart(forecast_df.set_index("Month"))
    st.write(forecast_df)

# =========================
# 🧾 FOOTER
# =========================
st.markdown("---")
st.caption("📊 Developed for Market Basket Analysis — DWDM Project | Streamlit ML Showcase")
