
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_csv("synthetic_supply_chain.csv")

print("✅ Dataset Loaded Successfully!")
print(f"Shape: {data.shape}")

data = data.fillna("Unknown")

if 'delay_days' not in data.columns:
    raise KeyError("❌ Column 'delay_days' not found! Please check dataset columns.")

data['delay_days'] = pd.to_numeric(data['delay_days'], errors='coerce')
delayed_data = data[data['delay_days'] > 0]

print(f"\n📦 Delayed Shipments: {len(delayed_data)} ({round((len(delayed_data)/len(data))*100, 2)}%)")

expected_cols = ['supplier_name', 'component_name', 'route', 'carrier', 'delay_cause']
cols = [c for c in expected_cols if c in data.columns]
if not cols:
    raise ValueError("❌ No matching columns found in dataset!")

transactions = delayed_data[cols].astype(str).apply(lambda x: ','.join(x), axis=1)
transactions = transactions.str.get_dummies(sep=',')

print("\n✅ Transaction matrix created.")

frequent_itemsets = apriori(transactions, min_support=0.01, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.8)
    rules = rules[rules['consequents'].astype(str).str.contains('delay_cause', na=False)]
    rules = rules.sort_values(by='lift', ascending=False)
else:
    rules = pd.DataFrame()

if not rules.empty:
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 5))
    top_itemsets = frequent_itemsets.head(10)
    top_itemsets['itemset_str'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    sns.barplot(data=top_itemsets, x='support', y='itemset_str', palette='Blues_r')
    plt.title("Top 10 Frequent Itemsets")
    plt.xlabel("Support")
    plt.ylabel("Itemsets")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.7, c=rules['lift'], cmap='viridis')
    plt.colorbar(label='Lift')
    plt.title("Support vs Confidence (Delay-related Rules)")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rules, x='lift', y='confidence', size='support',
                    hue='support', palette='coolwarm', sizes=(50, 400), alpha=0.7)
    plt.title("Lift vs Confidence (Bubble size = Support)")
    plt.xlabel("Lift")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.show()
else:
    print("\nℹ️ No delay-related rules found (try lowering min_support).")

def safe_save(df, base_name):
    """Save file safely even if locked by Excel."""
    file_name = f"{base_name}.csv"
    attempt = 1
    while True:
        try:
            df.to_csv(file_name, index=False, mode='w')
            return file_name
        except PermissionError:
            file_name = f"{base_name}_{attempt}.csv"
            attempt += 1

rules_path = safe_save(rules, "delay_association_rules") if not rules.empty else None
itemsets_path = safe_save(frequent_itemsets, "frequent_itemsets") if not frequent_itemsets.empty else None

print("\n💾 Files saved:")
if rules_path:
    print(f" - {rules_path}")
if itemsets_path:
    print(f" - {itemsets_path}")

print("\n=== Insights Summary ===")
if not rules.empty:
    print("• Frequent itemsets highlight common supplier-route-delay patterns.")
    print("• High 'lift' indicates strong correlations between suppliers/routes and delay causes.")
else:
    print("• No significant rules found — consider lowering min_support to 0.02 or 0.01.")
