# ============================================================
# sync_all_to_mongo.py
# ============================================================
# Unified script to push:
# - Preprocessed Dataset
# - Cross-Sell Rules
# - Model Metrics (auto-detect)
# - Recent Predictions (optional)
# ============================================================

import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import joblib

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "market_basket_ai"

DATA_PATH = "data/market_basket_cleaned_with_ids_regenerated.csv"
RULES_PATH = "outputs/association_rules_manual.csv"
MODELS_DIR = "models"

# ============================================================
# 🔌 CONNECT TO MONGODB
# ============================================================
def connect_mongo(uri=MONGO_URI, db_name=DB_NAME):
    """Connect to MongoDB."""
    client = MongoClient(uri)
    db = client[db_name]
    print(f"✅ Connected to MongoDB database: {db_name}")
    return db


# ============================================================
# 📤 UPLOAD PREPROCESSED DATA
# ============================================================
def upload_preprocessed_data(db, csv_path=DATA_PATH):
    if not os.path.exists(csv_path):
        print(f"❌ Preprocessed file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    coll = db["preprocessed_data"]
    coll.delete_many({})
    df["_upload_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coll.insert_many(df.to_dict(orient="records"))
    print(f"✅ Uploaded {len(df)} rows → `preprocessed_data` collection.")


# ============================================================
# 🔗 UPLOAD CROSS-SELL RULES
# ============================================================
def upload_cross_sell_rules(db, csv_path=RULES_PATH):
    if not os.path.exists(csv_path):
        print(f"⚠️ No cross-sell rules file found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    coll = db["cross_sell_rules"]
    coll.delete_many({})
    df["_upload_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    coll.insert_many(df.to_dict(orient="records"))
    print(f"✅ Uploaded {len(df)} rules → `cross_sell_rules` collection.")


# ============================================================
# 📈 UPLOAD MODEL METRICS (AUTO-DETECT FROM MODELS)
# ============================================================
def upload_model_metrics(db, models_dir=MODELS_DIR):
    coll = db["model_metrics"]
    coll.delete_many({})

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    records = []
    for f in model_files:
        model_name = f.replace(".pkl", "")
        file_path = os.path.join(models_dir, f)
        size_kb = round(os.path.getsize(file_path) / 1024, 2)
        records.append({
            "Model_Name": model_name,
            "File_Size_KB": size_kb,
            "Upload_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if records:
        coll.insert_many(records)
        print(f"✅ Logged {len(records)} model files → `model_metrics` collection.")
    else:
        print("⚠️ No model files found to log.")


# ============================================================
# 🧠 OPTIONAL: UPLOAD SAMPLE PREDICTIONS (IF ANY)
# ============================================================
def upload_sample_predictions(db):
    """Simulate prediction logs for demo."""
    sample_preds = [
        {"Feature": "Product Demand", "Prediction": "High", "Confidence": 0.88, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Feature": "Customer Behavior", "Prediction": "Butter", "Confidence": 0.92, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Feature": "Sales Forecasting", "Prediction": "₹18350", "Confidence": 0.85, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ]
    coll = db["prediction_logs"]
    coll.delete_many({})
    coll.insert_many(sample_preds)
    print(f"✅ Inserted {len(sample_preds)} sample predictions → `prediction_logs` collection.")


# ============================================================
# 🚀 MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("🔄 Syncing all processed data and model info to MongoDB...\n")
    db = connect_mongo()

    upload_preprocessed_data(db)
    upload_cross_sell_rules(db)
    upload_model_metrics(db)
    upload_sample_predictions(db)

    print("\n🎯 All sync operations completed successfully!")
    print("🧭 Collections now available:")
    print("  • preprocessed_data")
    print("  • cross_sell_rules")
    print("  • model_metrics")
    print("  • prediction_logs")
