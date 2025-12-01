# ======================================================
# 4️⃣ CROSS-SELLING
# ======================================================
elif page == "Cross-Selling Recommendations":
    st.title("🛒 Cross-Selling Suggestions")
    st.write("Discover which product combos perform well together.")
    
    path = os.path.join(OUTPUT_DIR, "association_rules_manual.csv")
    if os.path.exists(path):
        rules_df = pd.read_csv(path)
        st.dataframe(rules_df.head(25))
        
        product = st.selectbox("Search product to view related combos:", sorted(rules_df["antecedent"].unique()))
        related = rules_df[rules_df["antecedent"] == product].sort_values("confidence", ascending=False).head(10)
        if len(related) > 0:
            st.write(f"### 🧩 Products often bought with **{product}:**")
            st.table(related[["consequent", "support", "confidence", "lift"]])
        else:
            st.info("No related combos found for this product.")
    else:
        st.error("No association rules file found in outputs/.")
