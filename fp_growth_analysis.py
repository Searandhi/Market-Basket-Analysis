# ==========================================================
# MARKET BASKET ANALYSIS – FP-GROWTH ALGORITHM
# ==========================================================

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ==========================================================
# 1️⃣ LOAD THE CLEANED DATA
# ==========================================================
df = pd.read_csv("market_basket_cleaned.csv")
print("✅ Cleaned Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================================================
# 2️⃣ PREPARE TRANSACTION DATA FOR FP-GROWTH
# ==========================================================
# Here we’ll use Product columns (one-hot encoded from preprocessing)
# Extract only product columns
product_cols = [col for col in df.columns if col.startswith("Product_")]

# If product columns not found (in case of other naming), handle manually
if len(product_cols) == 0:
    print("⚠️ No product columns found. Please verify dataset.")
else:
    print(f"🛒 Found {len(product_cols)} product columns for FP-Growth")

# Convert product columns into Boolean values (0 or 1)
basket_data = df[product_cols].astype(bool)

# ==========================================================
# 3️⃣ APPLY FP-GROWTH ALGORITHM
# ==========================================================
# Minimum support can be adjusted (0.01 means item appears in at least 1% of transactions)
frequent_itemsets = fpgrowth(basket_data, min_support=0.01, use_colnames=True)

# Sort by support (descending)
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
print("\n✅ Frequent Itemsets Found:")
print(frequent_itemsets.head(10))

# ==========================================================
# 4️⃣ GENERATE ASSOCIATION RULES
# ==========================================================
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by="lift", ascending=False)

print("\n✅ Top Association Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# ==========================================================
# 5️⃣ SAVE RESULTS
# ==========================================================
frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
rules.to_csv("association_rules.csv", index=False)
print("\n💾 FP-Growth results saved as:")
print(" - frequent_itemsets.csv")
print(" - association_rules.csv")
