# regenerate_and_train.py
"""
Robust dataset regeneration + model training pipeline (fixed)
Produces:
 - data/market_basket_cleaned_with_ids_regenerated.csv
 - models/*.pkl
 - outputs/association_rules_manual.csv
 - outputs/df_daily_for_forecast.csv (if forecasting data insufficient)
"""
import os, sys, traceback
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Paths
INPUT = os.path.join("data", "market_basket_cleaned_with_ids.csv")
OUT_CSV = os.path.join("data", "market_basket_cleaned_with_ids_regenerated.csv")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_and_check():
    if not os.path.exists(INPUT):
        raise FileNotFoundError(f"Expected dataset at: {INPUT}")
    df = pd.read_csv(INPUT)
    print("Loaded original CSV shape:", df.shape)
    return df

def ensure_store_location(df):
    if "Store_Location" not in df.columns:
        print("⚠️ 'Store_Location' missing — filling with defaults.")
        df["Store_Location"] = np.random.choice(["Chennai","Bangalore","Mumbai","Delhi"], size=len(df))
    df["Store_Location"] = df["Store_Location"].astype(str)
    return df

def make_transaction_date(df):
    if "Transaction_Date" not in df.columns:
        base = pd.to_datetime("2025-01-01")
        df["Transaction_Date"] = [base + pd.Timedelta(days=(i % 180)) for i in range(len(df))]
    else:
        df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
        df["Transaction_Date"].fillna(pd.to_datetime("2025-01-01"), inplace=True)
    return df

def ensure_ids(df):
    if "Transaction_ID" not in df.columns:
        df["Transaction_ID"] = ["T" + str(10000 + i) for i in range(len(df))]
    if "Customer_ID" not in df.columns:
        df["Customer_ID"] = ["C" + str(i % 500) for i in range(len(df))]
    return df

def ensure_numeric_columns(df):
    if "Quantity" not in df.columns:
        df["Quantity"] = np.random.randint(1,5,size=len(df))
    if "Total_Amount" not in df.columns:
        df["Total_Amount"] = (df.get("Quantity",1) * np.random.uniform(20,200,size=len(df))).round(2)
    if "Cross_Sell_Potential" not in df.columns:
        df["Cross_Sell_Potential"] = np.random.uniform(0.0,1.0,size=len(df))
    if "Month" not in df.columns:
        df["Month"] = pd.to_datetime(df["Transaction_Date"]).dt.month
    return df

def cast_product_columns_int(df, product_cols):
    for c in product_cols:
        df[c] = df[c].fillna(0)
        df[c] = df[c].astype(int)
    return df

def inject_co_purchases(df, product_cols, frac=0.25):
    prod_counts = df[product_cols].sum(axis=1)
    avg_products = prod_counts.mean()
    print("Average products/transaction before injection:", round(avg_products,3))
    if avg_products < 1.5:
        np.random.seed(42)
        tx_indices = df.sample(frac=frac, random_state=42).index
        top_cols = df[product_cols].sum().sort_values(ascending=False).head(10).index.tolist()
        top = [t.replace("Product_","") for t in top_cols]
        for idx in tx_indices:
            possible = [p for p in top if df.at[idx, f"Product_{p}"] == 0]
            if not possible:
                continue
            k = np.random.choice([1,2], p=[0.75,0.25])
            pick = possible[:k]
            for p in pick:
                df.at[idx, f"Product_{p}"] = int(1)
        prod_counts2 = df[product_cols].sum(axis=1)
        print("Injected co-purchases. New avg products/transaction:", round(prod_counts2.mean(),3))
    else:
        print("No injection needed (adequate multi-product transactions).")
    return df

def compute_customer_category_by_visit(df):
    df = df.sort_values(["Customer_ID","Transaction_Date"]).reset_index(drop=True)
    visit_count = {}
    cats = []
    for _, row in df.iterrows():
        cid = row["Customer_ID"]
        visit_count[cid] = visit_count.get(cid,0) + 1
        vc = visit_count[cid]
        if vc == 1:
            cats.append("New")
        elif vc <= 3:
            cats.append("Rare")
        else:
            cats.append("Regular")
    df["Customer_Category"] = cats
    return df

def melt_products_safe(df, product_cols):
    id_cols = [c for c in df.columns if c not in product_cols]
    df_long = df.melt(id_vars=id_cols, value_vars=product_cols, var_name="Product", value_name="Purchased")
    df_long = df_long[df_long["Purchased"] == 1].drop(columns=["Purchased"]).copy()
    df_long["Product"] = df_long["Product"].str.replace("Product_","", regex=False)
    return df_long

def compute_product_demand_score(df_long):
    df_long["Month"] = pd.to_datetime(df_long["Transaction_Date"]).dt.month
    prod_month = df_long.groupby(["Product","Month"]).size().reset_index(name="count")
    prod_month["month_max"] = prod_month.groupby("Month")["count"].transform("max")
    prod_month["demand_score"] = prod_month["count"] / prod_month["month_max"]
    df_long = df_long.merge(prod_month[["Product","Month","demand_score"]], on=["Product","Month"], how="left")
    demand_tx = df_long.groupby("Transaction_ID")["demand_score"].mean().reset_index().rename(columns={"demand_score":"Product_Demand_Score_calc"})
    return demand_tx

def train_all_models(df, df_long):
    print("DEBUG: df columns before training:", df.columns.tolist())

    # TX-level features (ensure presence)
    tx_features = df[["Transaction_ID", "Product_Demand_Score", "Store_Location",
                      "Quantity", "Total_Amount", "Cross_Sell_Potential", "Month"]].drop_duplicates(subset=["Transaction_ID"])
    for col in ["Product_Demand_Score", "Store_Location", "Quantity", "Total_Amount", "Cross_Sell_Potential", "Month"]:
        if col not in tx_features.columns:
            print(f"⚠️ Missing tx feature '{col}', filling defaults.")
            if col == "Store_Location":
                tx_features[col] = "Unknown"
            elif col == "Product_Demand_Score":
                tx_features[col] = 0.5
            else:
                tx_features[col] = 0

    # Merge long + tx features (raw)
    df_pd_raw = df_long.merge(tx_features, on="Transaction_ID", how="left")

    # ---- Ensure Product_Demand_Score numeric and sensible ----
    df_pd_raw["Product_Demand_Score"] = pd.to_numeric(df_pd_raw.get("Product_Demand_Score", 0.5), errors="coerce").fillna(0.5)

    # ---- Build numeric block (keep it separate before dummies) ----
    numeric_cols = ["Quantity", "Total_Amount", "Cross_Sell_Potential", "Month", "Product_Demand_Score"]
    # ensure each numeric col exists in df_pd_raw, otherwise create default
    for col in numeric_cols:
        if col not in df_pd_raw.columns:
            if col == "Product_Demand_Score":
                df_pd_raw[col] = 0.5
            else:
                df_pd_raw[col] = 0.0

    numeric_block = df_pd_raw[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0.0)
    # keep Transaction_ID & Customer_ID for later joins / grouping if needed
    id_block = df_pd_raw[["Transaction_ID", "Customer_ID", "Product", "Transaction_Date"]].copy()

    # ---- Identify categorical columns (strictly object/category and exclude ids & numeric) ----
    exclude = set(["Transaction_ID", "Customer_ID", "Transaction_Date"] + numeric_cols)
    cat_cols = [c for c in df_pd_raw.columns if (df_pd_raw[c].dtype == "object" or str(df_pd_raw[c].dtype).startswith("category")) and c not in exclude]

    # defensive: restrict categorical columns to known safe set
    # We definitely want to one-hot encode Product and Customer_Category (if present).
    safe_cat = []
    if "Product" in df_pd_raw.columns:
        safe_cat.append("Product")
    if "Customer_Category" in df_pd_raw.columns:
        safe_cat.append("Customer_Category")
    # merge other object columns only if they are truly categorical (small unique count)
    for c in cat_cols:
        if c not in safe_cat:
            nunique = df_pd_raw[c].nunique(dropna=True)
            if nunique <= 50:  # heuristic
                safe_cat.append(c)

    cat_cols = sorted(list(set(safe_cat)))

    # ---- One-hot encode categorical columns (defensive) ----
    if len(cat_cols) > 0:
        print("Encoding categorical columns:", cat_cols)
        df_dummies = pd.get_dummies(df_pd_raw[cat_cols].fillna("NA"), prefix_sep="_", drop_first=True)
    else:
        df_dummies = pd.DataFrame(index=df_pd_raw.index)

    # ---- Reconstruct the cleaned dataframe used for modeling ----
    df_model = pd.concat([id_block.reset_index(drop=True), numeric_block.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)

    # Build feature list
    numeric_core = ["Quantity", "Total_Amount", "Cross_Sell_Potential", "Month"]
    dummy_cols = [c for c in df_model.columns if c.startswith("Store_Location_") or c.startswith("Product_") or c.startswith("Customer_Category_")]
    feat_cols = [c for c in numeric_core if c in df_model.columns] + dummy_cols
    feat_cols = [c for c in feat_cols if c in df_model.columns]

    if len(feat_cols) == 0:
        raise RuntimeError("No features available for product demand model after encoding. Check dataset.")

    # ---- Labeling (balanced bins) ----
    # compute thresholds on Product_Demand_Score column from numeric_block
        # --- Clean duplicates (_x, _y) that came from merge ---
    for c in df_model.columns:
        if c.endswith("_x") and c[:-2] + "_y" in df_model.columns:
            df_model[c[:-2]] = df_model[c].combine_first(df_model[c[:-2] + "_y"])
    df_model = df_model.loc[:, ~df_model.columns.str.endswith(("_x", "_y"))]

    # ---- Force Product_Demand_Score to be a clean numeric Series ----
        # ---- Force Product_Demand_Score to be a clean numeric Series ----
    score_cols = [c for c in df_model.columns if "Product_Demand_Score" in c]
    if len(score_cols) > 1:
        print(f"⚠️ Multiple demand score columns found: {score_cols}. Merging into one.")
        # combine by taking the first non-null numeric value
        df_model["Product_Demand_Score"] = (
            df_model[score_cols]
            .apply(lambda row: pd.to_numeric(row, errors="coerce").dropna().head(1).values[0] if any(pd.to_numeric(row, errors="coerce").notna()) else np.nan, axis=1)
            .fillna(0.5)
        )
        df_model.drop(columns=[c for c in score_cols if c != "Product_Demand_Score"], inplace=True, errors="ignore")
    elif len(score_cols) == 1:
        df_model["Product_Demand_Score"] = pd.to_numeric(df_model[score_cols[0]], errors="coerce").fillna(0.5)
    else:
        df_model["Product_Demand_Score"] = 0.5


    # --- Compute thresholds safely ---
        # --- Compute thresholds safely ---
    score_series = df_model["Product_Demand_Score"]

    # Handle accidental DataFrame/duplicate columns
    if isinstance(score_series, pd.DataFrame):
        print("⚠️ Product_Demand_Score is a DataFrame — flattening to 1D numeric.")
        score_series = score_series.iloc[:, 0]  # take first column

    score_series = pd.to_numeric(score_series, errors="coerce").fillna(0.5).astype(float)

    # Compute quantiles robustly as floats
    try:
        low_thr = float(np.nanquantile(score_series, 0.35))
        high_thr = float(np.nanquantile(score_series, 0.70))
    except Exception as e:
        print("⚠️ Quantile computation failed, falling back to defaults:", e)
        low_thr, high_thr = 0.3, 0.7

    print(f"Demand thresholds → Low: {low_thr:.4f}, High: {high_thr:.4f}")


    # --- Safe scalar labeling function ---
    def demand_label(val):
        try:
            v = float(val)
        except:
            v = 0.0
        if v >= high_thr:
            return 1
        elif v <= low_thr:
            return 0
        else:
            return np.nan

    df_model["reorder_needed"] = score_series.apply(demand_label)
    df_model = df_model.dropna(subset=["reorder_needed"]).copy()
    df_model["reorder_needed"] = df_model["reorder_needed"].astype(int)


    # ---- Balance dataset via upsampling minority if required ----
    from sklearn.utils import resample
    if df_model["reorder_needed"].nunique() < 2:
        print("Product demand target single class — skipping training.")
    else:
        majority = df_model[df_model["reorder_needed"] == df_model["reorder_needed"].mode()[0]]
        minority = df_model[df_model["reorder_needed"] != df_model["reorder_needed"].mode()[0]]
        if len(minority) > 0:
            minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
            df_pd_bal = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df_pd_bal = df_model.copy()

        X = df_pd_bal[feat_cols].fillna(0).astype(float)
        y = df_pd_bal["reorder_needed"].astype(int)

        # Train product demand model with class_balance
        if len(np.unique(y)) > 1 and len(X) > 20:
            print("Product demand class distribution:", Counter(y))
            classes = np.unique(y)
            cw = compute_class_weight(class_weight='balanced', classes=classes, y=y)
            class_weight_dict = {cls: float(w) for cls, w in zip(classes, cw)}
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            prod_model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight=class_weight_dict)
            prod_model.fit(X_train, y_train)
            joblib.dump(prod_model, os.path.join(MODELS_DIR, "product_demand_rf.pkl"))
            print("Saved product_demand_rf.pkl (balanced)")
            print(classification_report(y_test, prod_model.predict(X_test), digits=3))
        else:
            print("Product demand target single class or too small — model not trained.")

    # ---------------------------
    # Customer behaviour lookup baseline
    # ---------------------------
    cust_prod = df_long.groupby(["Customer_ID", "Product"]).size().reset_index(name="count")
    top_prod = cust_prod.sort_values(["Customer_ID", "count"], ascending=[True, False]).groupby("Customer_ID").first().reset_index()
    lookup_map = dict(zip(top_prod["Customer_ID"], top_prod["Product"]))
    joblib.dump(lookup_map, os.path.join(MODELS_DIR, "customer_top_product_lookup.pkl"))
    print("Saved customer_top_product_lookup.pkl (lookup baseline)")

    # Optional supervised behaviour classifier (same as before but robust)
    try:
        last_prod = df_long.groupby("Customer_ID").apply(lambda g: g.sort_values("Transaction_Date").iloc[-1]).reset_index(drop=True)[["Customer_ID", "Product"]].rename(columns={"Product": "last_product"})
        hist = df_long.groupby("Customer_ID")["Product"].agg(lambda x: x.value_counts().index[0]).reset_index().rename(columns={"Product": "hist_top_product"})
        mb = hist.merge(last_prod, on="Customer_ID").dropna()
        if len(mb) > 60 and len(mb["last_product"].unique()) > 1:
            le = LabelEncoder()
            mb["y"] = le.fit_transform(mb["last_product"])
            mb["hist_enc"] = le.transform(mb["hist_top_product"])
            Xc = mb[["hist_enc"]]; yc = mb["y"]
            X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
            clf = LogisticRegression(max_iter=500)
            clf.fit(X_train, y_train)
            joblib.dump((clf, le), os.path.join(MODELS_DIR, "customer_behavior_lr.pkl"))
            print("Saved customer_behavior_lr.pkl (optional supervised)")
        else:
            print("Skipped supervised behaviour classifier (insufficient variety).")
    except Exception as e:
        print("Behaviour classifier skipped due to error:", e)

    # ---------------------------
    # Cross-sell rules (pair counting)
    # ---------------------------
    trans_products = df_long.groupby("Transaction_ID")["Product"].apply(lambda x: list(set(x))).to_dict()
    n_transactions = len(trans_products)
    pair_counts, single_counts = Counter(), Counter()
    for products in trans_products.values():
        for p in products:
            single_counts[p] += 1
        for a, b in combinations(sorted(products), 2):
            pair_counts[(a, b)] += 1

    rules = []
    min_support = 0.001; min_confidence = 0.02
    for (a, b), cnt in pair_counts.items():
        support = cnt / max(1, n_transactions)
        conf_ab = cnt / single_counts[a] if single_counts[a] > 0 else 0
        lift_ab = conf_ab / (single_counts[b] / n_transactions) if single_counts[b] > 0 else 0
        if support >= min_support and conf_ab >= min_confidence:
            rules.append({"antecedent": a, "consequent": b, "support": support, "confidence": conf_ab, "lift": lift_ab})
        conf_ba = cnt / single_counts[b] if single_counts[b] > 0 else 0
        if support >= min_support and conf_ba >= min_confidence:
            rules.append({"antecedent": b, "consequent": a, "support": support, "confidence": conf_ba, "lift": lift_ab})

    rules_df = pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False) if len(rules) > 0 else pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    rules_df.to_csv(os.path.join(OUTPUTS_DIR, "association_rules_manual.csv"), index=False)
    print("Saved association_rules_manual.csv (cross-sell rules). Count:", len(rules_df))

    # ---------------------------
    # Sales forecasting (unchanged logic but defensive)
    # ---------------------------
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
    df_daily = df.groupby([df["Transaction_Date"].dt.date, "Store_Location"]).agg(daily_sales=("Total_Amount", "sum")).reset_index()
    df_daily.rename(columns={"Transaction_Date": "date"}, inplace=True)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values(["Store_Location", "date"])
    df_daily["lag_1"] = df_daily.groupby("Store_Location")["daily_sales"].shift(1)
    df_daily["lag_7"] = df_daily.groupby("Store_Location")["daily_sales"].shift(7)
    df_daily["rolling_7"] = df_daily.groupby("Store_Location")["daily_sales"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    train_df = df_daily.dropna().copy()
    if len(train_df) < 20:
        train_df.to_csv(os.path.join(OUTPUTS_DIR, "df_daily_for_forecast.csv"), index=False)
        print("Not enough daily rows to train forecasting model (saved df_daily_for_forecast.csv)")
    else:
        train_df = pd.get_dummies(train_df, columns=["Store_Location"], drop_first=True)
        feat_cols = [c for c in train_df.columns if c.startswith(("lag_", "rolling_", "Store_Location_"))]
        Xf = train_df[feat_cols]; yf = train_df["daily_sales"]
        X_train, X_test, y_train, y_test = train_test_split(Xf, yf, test_size=0.2, random_state=42)
        sales_model = GradientBoostingRegressor(n_estimators=300, random_state=42)
        sales_model.fit(X_train, y_train)
        joblib.dump((sales_model, feat_cols), os.path.join(MODELS_DIR, "sales_forecast_gbr.pkl"))
        print("Saved sales_forecast_gbr.pkl")
        print("Sales forecasting MAE:", mean_absolute_error(y_test, sales_model.predict(X_test)), "R2:", r2_score(y_test, sales_model.predict(X_test)))

def main():
    try:
        df = load_and_check()
        # ensure product_* columns exist
        product_cols = [c for c in df.columns if c.lower().startswith("product_")]
        if len(product_cols) == 0:
            raise ValueError("No product_* columns found in dataset.")
        # ensure store location exists early
        df = ensure_store_location(df)
        df = ensure_ids(df)
        df = make_transaction_date(df)
        df = ensure_numeric_columns(df)
        # ensure product columns are ints
        df = cast_product_columns_int(df, product_cols)
        # inject co-purchases if needed
        df = inject_co_purchases(df, product_cols, frac=0.25)
        # compute Month if missing
        if "Month" not in df.columns:
            df["Month"] = pd.to_datetime(df["Transaction_Date"]).dt.month
        # compute customer categories deterministically
        df = compute_customer_category_by_visit(df)
        # melt and compute demand
        df_long = melt_products_safe(df, product_cols)
        demand_tx = compute_product_demand_score(df_long)
        df = df.merge(demand_tx, on="Transaction_ID", how="left")
        # finalize Product_Demand_Score
        if "Product_Demand_Score" in df.columns:
            df["Product_Demand_Score"] = df["Product_Demand_Score_calc"].fillna(df["Product_Demand_Score"]).fillna(0.5)
        else:
            df["Product_Demand_Score"] = df["Product_Demand_Score_calc"].fillna(0.5)
        if "Product_Demand_Score_calc" in df.columns:
            df.drop(columns=["Product_Demand_Score_calc"], inplace=True)
        # save regenerated
        df.to_csv(OUT_CSV, index=False)
        print("Saved regenerated dataset:", OUT_CSV)
        # rebuild long to reflect any new changes
        df_long = melt_products_safe(df, product_cols)
        # train
        train_all_models(df, df_long)
        print("All done. Models saved into:", MODELS_DIR, "Outputs into:", OUTPUTS_DIR)
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
