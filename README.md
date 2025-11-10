⛓️ Supply Chain Delay Analysis using Apriori Algorithm
📘 Overview

The Supply Chain Delay Analysis project aims to identify and understand the underlying factors that contribute to delivery delays in supply chain and logistics operations.
Using the Apriori association rule mining algorithm, this project analyzes supplier, component, route, and carrier data to uncover frequent co-occurrences that lead to shipment delays.
It helps organizations pinpoint root causes, reduce bottlenecks, and enhance overall operational efficiency.

This system is implemented as an interactive Streamlit web application that allows users to upload datasets, run Apriori analysis, visualize results, and download actionable insights.

🚀 Features

Upload your own CSV or Excel dataset

Choose columns for analysis dynamically

Filter and analyze only delayed shipments

Run Apriori algorithm with adjustable support and confidence thresholds

Generate association rules and visualize patterns

Plot Support vs Confidence for better understanding

Download frequent itemsets and association rules as CSV files

Built-in insights summary for interpretation

🧠 Objective

To detect patterns and correlations between supply chain entities such as suppliers, components, carriers, and routes that frequently co-occur with delivery delays — enabling data-driven decision-making and process optimization.

🧩 Tech Stack

Programming Language: Python

Libraries:

pandas – data manipulation

mlxtend – Apriori and association rules

matplotlib – visualizations

streamlit – web app interface

Dataset:

Synthetic Supply Chain Dataset

Online Retail Dataset (UCI ML Repository)

User-uploaded datasets supported

📊 Algorithm Used
Apriori Algorithm

The Apriori algorithm is used to discover frequent itemsets and association rules in the dataset.
It identifies which combinations of items (like supplier, route, or component) frequently occur together in delayed shipments.

Key Metrics:

Support: Frequency of occurrence of an itemset

Confidence: Probability that a rule is true given its antecedent

Lift: Strength of a rule compared to random occurrence

⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/supply-chain-delay-analysis.git
cd supply-chain-delay-analysis

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

📂 Project Structure
supply-chain-delay-analysis/
│
├── app.py                          # Main Streamlit application
├── synthetic_supply_chain.csv       # Sample dataset
├── online+retail.zip                # Real dataset (optional)
├── delay_association_rules.csv      # Output rules
├── frequent_itemsets.csv            # Output frequent itemsets
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation

🧾 Example Datasets
Synthetic Supply Chain Dataset
supplier_name	component_name	route	carrier	delay_cause	delay_days
SupplierA	Motor	Route1	CarrierX	Weather	2
SupplierB	Sensor	Route2	CarrierY	Traffic	0
SupplierC	Wire	Route3	CarrierZ	Customs	4
Online Retail Dataset (UCI)

Used for real-world association mining on product transactions.

📈 Output

Frequent Itemsets:
Shows recurring supplier-component-route combinations that appear in delayed deliveries.

Association Rules:
Displays rules like:
SupplierA + Route1 → DelayCause=Weather
with metrics like support, confidence, and lift.

Visualizations:

Support vs Confidence Scatter Plot

Top Frequent Itemsets Bar Graph

🧭 Insights

Identifies key patterns responsible for delays.

Helps improve supplier selection, inventory planning, and logistics routing.

Enables data-driven decisions to reduce operational risks and inefficiencies.

💾 Outputs Generated

delay_association_rules.csv – mined rules

frequent_itemsets.csv – frequent itemsets

Interactive visualization via Streamlit UI

🔮 Future Enhancements

Integrate machine learning models for delay prediction.

Automate KPI dashboards for monitoring supply chain performance.

Connect with live ERP systems for real-time analysis.
