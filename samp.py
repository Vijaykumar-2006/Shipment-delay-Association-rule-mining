import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

np.random.seed(42)

items = ['Milk', 'Bread', 'Butter', 'Cheese', 'Eggs', 'Juice', 'Apples', 'Bananas', 'Coffee', 'Tea']
customers = np.arange(1000, 1100)
countries = ['United Kingdom', 'France', 'Germany']

data = []
for i in range(5000):  
    invoice_no = f"INV-{i+1}"
    cust_id = np.random.choice(customers)
    country = np.random.choice(countries, p=[0.7, 0.15, 0.15])
    n_items = np.random.randint(1, 6)
    bought_items = np.random.choice(items, n_items, replace=False)
    for item in bought_items:
        qty = np.random.randint(1, 5)
        data.append([invoice_no, cust_id, country, item, qty])

df = pd.DataFrame(data, columns=['InvoiceNo', 'CustomerID', 'Country', 'Description', 'Quantity'])

print("✅ Synthetic Retail Dataset Created Successfully!")
print(df.head())

df = df[df['Country'] == 'United Kingdom']
df = df[df['Quantity'] > 0]

basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().fillna(0))
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

print("\n🧺 Basket matrix created with shape:", basket.shape)

frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

print("\n✅ Frequent Itemsets:")
print(frequent_itemsets.head())

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by='lift', ascending=False)

if not rules.empty:
    print("\n🔥 Top 10 Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
else:
    print("\n⚠ No rules found with the current parameters. Try lowering min_support or min_threshold.")

rules.to_csv("synthetic_association_rules11.csv", index=False)
print("\n💾 Rules saved to 'synthetic_association_rules11.csv'")
