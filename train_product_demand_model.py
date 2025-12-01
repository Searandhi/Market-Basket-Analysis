# ============================================
# train_product_demand_model.py
# Standalone script for training the Product Demand Prediction model
# ============================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
import joblib

# --- Paths ---
DATA_PATH = os.path.join("data", "market_basket_cleaned_with_ids_regenerated.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load dataset ---
print(f"Loading dataset → {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("✅ Loaded:", df.shape)

# --- Identify product columns ---
product_cols = [c for c in df.columns if c.startswith("Product_")]
print(f"Detected {len(product_cols)} product columns")

# --- Build transaction-level dataframe ---
if "Transaction_ID" not in df.columns:
    df["Transaction_ID"] = [f"T{i}" for i in range(len(df))]

tx_features = df[
    ["Transaction_ID", "Product_Demand_Score", "Quantity", "Total_Amount",
     "Cross_Sell_Potential", "Month"] + product_cols
].copy()

# --- Remove duplicate columns if any ---
if tx_features.columns.duplicated().any():
    dups = list(tx_features.columns[tx_features.columns.duplicated()])
    print(f"⚠️ Duplicate columns detected: {dups} → Keeping first occurrence only")
    tx_features = tx_features.loc[:, ~tx_features.columns.duplicated()]

# --- Ensure numeric columns are clean ---
for col in ["Product_Demand_Score", "Quantity", "Total_Amount", "Cross_Sell_Potential", "Month"]:
    if col in tx_features.columns:
        col_data = tx_features[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        tx_features[col] = pd.to_numeric(col_data.squeeze(), errors="coerce").fillna(0.0)
    else:
        print(f"⚠️ Missing column {col}, filling with zeros.")
        tx_features[col] = 0.0

# --- Compute quantile thresholds for labeling ---
score_series = tx_features["Product_Demand_Score"]
if isinstance(score_series, pd.DataFrame):
    score_series = score_series.iloc[:, 0]
score_series = pd.to_numeric(score_series, errors="coerce").fillna(0.0)

low_thr = float(score_series.quantile(0.35))
high_thr = float(score_series.quantile(0.70))
print(f"Demand thresholds → Low: {low_thr:.4f}, High: {high_thr:.4f}")

# --- Label demand ---
def demand_label(score):
    """Convert demand score into binary label (0=Low, 1=High)"""
    try:
        if isinstance(score, (pd.Series, np.ndarray, list)):
            score = float(score[0]) if len(score) > 0 else 0.0
        else:
            score = float(score)
    except Exception:
        score = 0.0
    if score >= high_thr:
        return 1
    elif score <= low_thr:
        return 0
    else:
        return np.nan

tx_features["reorder_needed"] = tx_features["Product_Demand_Score"].apply(demand_label)
tx_features = tx_features.dropna(subset=["reorder_needed"]).copy()
tx_features["reorder_needed"] = tx_features["reorder_needed"].astype(int)

# --- Balance dataset ---
majority = tx_features[tx_features["reorder_needed"] == tx_features["reorder_needed"].mode()[0]]
minority = tx_features[tx_features["reorder_needed"] != tx_features["reorder_needed"].mode()[0]]

if len(minority) > 0:
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    df_bal = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
else:
    df_bal = tx_features.copy()

# --- Prepare training data ---
feat_cols = ["Quantity", "Total_Amount", "Cross_Sell_Potential", "Month"] + product_cols
X = df_bal[feat_cols].fillna(0).astype(float)
y = df_bal["reorder_needed"].astype(int)

# --- Compute class weights ---
classes = np.unique(y)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = {cls: float(w) for cls, w in zip(classes, cw)}

# --- Train model ---
print("Training Product Demand Model ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight=class_weight_dict,
    max_depth=None,
    min_samples_split=2
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, digits=3))

# --- Save model ---
MODEL_PATH = os.path.join(MODEL_DIR, "product_demand_rf.pkl")
joblib.dump(model, MODEL_PATH)
print(f"💾 Saved model → {MODEL_PATH}")
print(f"✅ Training Complete | Samples: {len(X)} | Classes: {np.bincount(y)}")

# --- Bonus: show products with high vs low average demand score ---
try:
    print("\n📊 Computing product demand summary...")

    # Ensure "Product_Demand_Score" is 1D numeric
    if isinstance(df["Product_Demand_Score"], pd.DataFrame):
        df["Product_Demand_Score"] = df["Product_Demand_Score"].iloc[:, 0]
    df["Product_Demand_Score"] = pd.to_numeric(df["Product_Demand_Score"], errors="coerce").fillna(0.0)

    # Melt product columns to long format
    melted = df.melt(
        id_vars=["Product_Demand_Score"],
        value_vars=product_cols,
        var_name="Product",
        value_name="Purchased"
    )

    # Compute mean demand score per product
    prod_demand_summary = (
        melted[melted["Purchased"] == 1]
        .groupby("Product")["Product_Demand_Score"]
        .mean()
        .sort_values(ascending=False)
    )

    # Label demand categories (High, Medium, Low)
    high_cut = prod_demand_summary.quantile(0.70)
    low_cut = prod_demand_summary.quantile(0.30)
    summary_df = prod_demand_summary.reset_index()
    summary_df["Demand_Category"] = summary_df["Product_Demand_Score"].apply(
        lambda s: "High" if s >= high_cut else ("Low" if s <= low_cut else "Medium")
    )

    # Save and print results
    OUT_PATH = os.path.join("outputs", "product_demand_summary.csv")
    os.makedirs("outputs", exist_ok=True)
    summary_df.to_csv(OUT_PATH, index=False)

    print("\nTop 5 High Demand Products:")
    print(summary_df[summary_df["Demand_Category"] == "High"].head(5))

    print("\nTop 5 Low Demand Products:")
    print(summary_df[summary_df["Demand_Category"] == "Low"].head(5))

    print(f"\n💾 Saved detailed demand summary → {OUT_PATH}")

except Exception as e:
    print(f"⚠️ Could not generate demand summary: {e}")
