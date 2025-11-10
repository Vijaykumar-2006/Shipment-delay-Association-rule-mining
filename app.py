
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


st.set_page_config(
    page_title="Supply Chain Delay Analysis",
    page_icon="⛓️",
    layout="wide"
)

st.title("⛓️ Supply Chain Delay Analysis using Apriori Algorithm")
st.markdown("Discover patterns in delayed shipments and find root causes.")


uploaded_file = st.file_uploader("📤 Upload your supply chain CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")
    st.write("**Dataset shape:**", data.shape)
    st.dataframe(data.head())

    st.subheader("🔍 Dataset Information")
    st.write(data.describe(include='all'))
    st.write("Missing Values per Column:")
    st.write(data.isnull().sum())

    data = data.fillna("Unknown")

    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()

    if not numeric_columns:
        st.error("❌ No numeric columns found! Please upload a dataset with at least one numeric delay column.")
    else:
        delay_col = st.selectbox(
            "Select the column that represents delay (numeric):",
            options=numeric_columns,
            index=0
        )

        if st.button("Filter Delayed Shipments"):
            data[delay_col] = pd.to_numeric(data[delay_col], errors='coerce')
            delayed_data = data[data[delay_col] > 0]
            st.session_state['delayed_data'] = delayed_data

        if 'delayed_data' in st.session_state:
            delayed_data = st.session_state['delayed_data']

            num_delayed = len(delayed_data)
            perc_delayed = (num_delayed / len(data)) * 100 if len(data) > 0 else 0
            st.write("**Number of delayed shipments:**", num_delayed)
            st.write("**Percentage of delayed shipments:**", round(perc_delayed, 2), "%")

            if num_delayed == 0:
                st.warning("⚠ No delayed shipments found. Try selecting another delay column or check your dataset.")
            else:
                st.subheader("🧩 Select Columns for Association Analysis")
                categorical_cols = delayed_data.select_dtypes(include=['object', 'category']).columns.tolist()
                cols = st.multiselect(
                    "Select at least two categorical columns to analyze:",
                    options=categorical_cols
                )

                if len(cols) < 2:
                    st.warning("Please select at least two categorical columns.")
                else:
                    st.success(f"Selected columns: {cols}")

                    st.info("Creating transactions for Apriori mining...")
                    transactions = delayed_data[cols].astype(str).apply(lambda x: ','.join(x), axis=1)
                    transactions = transactions.str.get_dummies(sep=',')
                    
                    st.write("Transaction matrix shape:", transactions.shape)
                    st.dataframe(transactions.head())

                    min_support = st.slider("Select Minimum Support:", 0.01, 0.5, 0.05)
                    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
                    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

                    st.subheader("📊 Frequent Itemsets")
                    if frequent_itemsets.empty:
                        st.warning("⚠ No frequent itemsets found. Try lowering the minimum support value.")
                    else:
                        st.dataframe(frequent_itemsets.head(20))

                        metric_choice = st.selectbox("Metric for Association Rules:", ["lift", "confidence", "support"])

                        if metric_choice in ["confidence", "support"]:
                            min_threshold = st.slider("Minimum Threshold for Metric:", 0.1, 1.0, 0.5)
                        else:
                            min_threshold = st.slider("Minimum Threshold for Metric:", 0.5, 5.0, 1.0)

                        rules = association_rules(frequent_itemsets, metric=metric_choice, min_threshold=min_threshold)
                        st.subheader("🔗 Association Rules")

                        if rules.empty:
                            st.warning("⚠ No rules found with the selected parameters. Try lowering minimum threshold or support.")
                        else:
                            rules = rules.sort_values(by=metric_choice, ascending=False)
                            st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))

                            st.subheader("📈 Support vs Confidence Plot")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.scatter(rules['support'], rules['confidence'], alpha=0.6)
                            ax.set_title("Support vs Confidence")
                            ax.set_xlabel("Support")
                            ax.set_ylabel("Confidence")
                            ax.grid(True)
                            st.pyplot(fig)

                            st.subheader("💾 Download Results")
                            rules_csv = rules.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Association Rules CSV", rules_csv, "delay_association_rules.csv", "text/csv")

                            itemsets_csv = frequent_itemsets.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Frequent Itemsets CSV", itemsets_csv, "frequent_itemsets.csv", "text/csv")

                            st.markdown("### 🧠 Insights Summary")
                            st.markdown("""
                            - Frequent itemsets show which suppliers, components, or routes commonly co-occur in delayed shipments.  
                            - High **lift** values indicate strong correlations — potential root causes for recurring delays.  
                            - Use these patterns for targeted improvements in supplier reliability and logistics planning.
                            """)

else:
    st.info("👆 Please upload your dataset to begin.")
