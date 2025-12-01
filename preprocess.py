# ============================================
# MARKET BASKET ANALYSIS - DATA PREPROCESSING (FIXED)
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("market_basket_raw_dataset.csv")

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)

# ============================================
# 1️⃣ HANDLE MISSING VALUES
# ============================================
print("\n🔹 Checking Missing Values...")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Fill numeric with median (avoid inplace warnings)
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ============================================
# 2️⃣ RECTIFY CUSTOMER CATEGORY (NO DATA LEAKAGE)
# ============================================
print("\n🔹 Fixing Data Leakage in 'Customer_Category'...")

# Sort by Customer_ID and Transaction_Date if exists
sort_cols = ["Customer_ID"]
if "Transaction_Date" in df.columns:
    sort_cols.append("Transaction_Date")
df = df.sort_values(by=sort_cols).reset_index(drop=True)

# Function to assign dynamic category
def compute_dynamic_category(df):
    category_list = []
    last_customer = None
    visit_count = {}
    
    for cust_id in df["Customer_ID"]:
        # Initialize if first encounter
        if cust_id not in visit_count:
            visit_count[cust_id] = 1
        else:
            visit_count[cust_id] += 1
        
        # Assign labels dynamically
        if visit_count[cust_id] == 1:
            category_list.append("New")
        elif visit_count[cust_id] <= 3:
            category_list.append("Rare")
        else:
            category_list.append("Regular")
    
    return category_list

# Apply the function
df["Customer_Category"] = compute_dynamic_category(df)

# ============================================
# 3️⃣ ENCODING CATEGORICAL FEATURES
# ============================================
print("\n🔹 Encoding Categorical Columns...")

label_enc = LabelEncoder()
label_cols = ["Customer_Category", "Customer_Gender", "Payment_Method"]

for col in label_cols:
    df[col] = label_enc.fit_transform(df[col])

# One-Hot Encode high-cardinality columns
df = pd.get_dummies(df, columns=["Product", "Store_Location"], drop_first=True)

# ============================================
# 4️⃣ SCALING NUMERICAL FEATURES
# ============================================
print("\n🔹 Scaling Numerical Columns...")
scaler = MinMaxScaler()
scale_cols = [
    "Quantity", "Price_per_Unit", "Total_Amount", "Discount",
    "Customer_Age", "Product_Demand_Score", "Cross_Sell_Potential"
]
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ============================================
# 5️⃣ TIME-BASED FEATURE ENGINEERING
# ============================================
def map_quarter(month):
    if month in [1, 2, 3]:
        return "Q1"
    elif month in [4, 5, 6]:
        return "Q2"
    elif month in [7, 8, 9]:
        return "Q3"
    else:
        return "Q4"

if "Month" in df.columns:
    df["Sales_Quarter"] = df["Month"].apply(map_quarter)
    df = pd.get_dummies(df, columns=["Sales_Quarter"], drop_first=True)

# ============================================
# 6️⃣ REMOVE UNUSED COLUMNS
# ============================================
df.drop(columns=["Transaction_ID", "Customer_ID"], inplace=True, errors="ignore")

# ============================================
# ✅ FINAL CLEAN DATA READY FOR MODEL TRAINING
# ============================================
print("\n✅ Preprocessing Complete Successfully!")
print("Final Shape:", df.shape)
print(df.head(5))

# ============================================
# SAVE CLEANED DATA
# ============================================
df.to_csv("market_basket_cleaned.csv", index=False)
print("\n💾 Cleaned dataset saved as 'market_basket_cleaned.csv'")
