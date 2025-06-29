# 📦 Model Selection, Training, and Hyperparameter Tuning Script

# ## 📌 Task: Split the data, choose models, train them, and improve performance using hyperparameter tuning
# - Purpose: Evaluate and optimize model performance on unseen data using Grid Search and Random Search.

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the final processed data with target variable
final_data = pd.read_csv("data/processed/final_model_data.csv")

# Define features and target
X = final_data.drop(columns=['is_high_risk'])
y = final_data['is_high_risk']

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Data split completed: Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ## ✅ Choose base models
log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(random_state=42)

print("✅ Models initialized: Logistic Regression and Random Forest")

# ## 🚀 Train base models
log_reg_model.fit(X_train, y_train)
print("✅ Logistic Regression model trained")

rf_model.fit(X_train, y_train)
print("✅ Random Forest model trained")

# ## 🔧 Hyperparameter Tuning: Logistic Regression with Grid Search
log_reg_params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
log_reg_grid = GridSearchCV(log_reg_model, log_reg_params, cv=5, scoring='accuracy')
log_reg_grid.fit(X_train, y_train)
print(f"✅ Best Logistic Regression params: {log_reg_grid.best_params_}")

# ## 🔍 Hyperparameter Tuning: Random Forest with Random Search
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_random = RandomizedSearchCV(rf_model, rf_params, cv=5, n_iter=5, scoring='accuracy', random_state=42)
rf_random.fit(X_train, y_train)
print(f"✅ Best Random Forest params: {rf_random.best_params_}")
