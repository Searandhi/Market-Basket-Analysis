# ============================================
# train_sales_forecast_per_store.py
# ============================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = os.path.join("data", "market_basket_cleaned_with_ids_regenerated.csv")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print(f"📂 Loading dataset → {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")

# Identify store dummy columns
store_cols = [c for c in df.columns if c.startswith("Store_Location_")]
if not store_cols:
    raise ValueError("No Store_Location_ columns found!")

forecast_summary = []

for col in store_cols:
    store_name = col.replace("Store_Location_", "")
    print(f"\n🏬 Training forecasting model for: {store_name}")

    # Filter store transactions
    mask = df[col] == 1
    df_store = df.loc[mask].copy()
    
    if df_store.empty:
        print(f"⚠️ No transactions for {store_name}. Skipping.")
        continue

    # Aggregate daily sales
    df_daily = (
        df_store.groupby(df_store["Transaction_Date"].dt.date)
        .agg(daily_sales=("Total_Amount", "sum"))
        .reset_index()
        .rename(columns={"Transaction_Date": "date"})
    )

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date")

    # Create lag features
    df_daily["lag_1"] = df_daily["daily_sales"].shift(1)
    df_daily["lag_7"] = df_daily["daily_sales"].shift(7)
    df_daily["rolling_7"] = df_daily["daily_sales"].shift(1).rolling(7, min_periods=1).mean()
    df_daily = df_daily.dropna()

    if len(df_daily) < 20:
        print(f"⚠️ Not enough data to train {store_name} (only {len(df_daily)} records). Skipping.")
        continue

    # Train-test split
    X = df_daily[["lag_1", "lag_7", "rolling_7"]]
    y = df_daily["daily_sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = GradientBoostingRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ {store_name} → MAE: {mae:.2f}, R²: {r2:.3f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"sales_forecast_{store_name.lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Saved model → {model_path}")

    forecast_summary.append({"Store": store_name, "Records": len(df_daily), "MAE": mae, "R2": r2})

# Save summary
summary_df = pd.DataFrame(forecast_summary)
summary_path = os.path.join(OUTPUTS_DIR, "sales_forecast_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n📊 Forecast summary saved → {summary_path}")
print("\n🎯 Training complete for all stores!")
