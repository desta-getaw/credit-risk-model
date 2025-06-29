# 📦 Proxy Target Variable Engineering Script

# ## 📌 Task: Integrate the Target Variable into the Main Dataset
# - Merge the new `is_high_risk` column back into the main processed dataset to prepare for model training.

import pandas as pd

# Load clustered RFM data with high-risk label
rfm = pd.read_csv("data/processed/rfm_with_high_risk_label.csv")

# Load main processed dataset (replace this path with your actual processed data file)
main_data = pd.read_csv("data/processed/processed_data.csv")

# Merge using customer_id to integrate is_high_risk column
merged_data = pd.merge(main_data, rfm[['customer_id', 'is_high_risk']], on='customer_id', how='left')

# Save final merged dataset
merged_data.to_csv("data/processed/final_model_data.csv", index=False)

print("✅ Target variable successfully integrated and saved to data/processed/final_model_data.csv")
