# 📦 predict.py - Model Evaluation and Model Registry

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import mlflow

final_data = pd.read_csv("data/processed/final_model_data.csv")

X = final_data.drop(columns=['is_high_risk'])
y = final_data['is_high_risk']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_log_reg = joblib.load("models/best_logistic_regression.pkl")
best_rf = joblib.load("models/best_random_forest.pkl")

log_reg_auc = roc_auc_score(y_test, best_log_reg.predict_proba(X_test)[:, 1])
rf_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])

# Select best model
best_model = best_rf if rf_auc > log_reg_auc else best_log_reg
print(f"✅ Best model based on ROC-AUC: {'Random Forest' if best_model == best_rf else 'Logistic Regression'}")

# Evaluate
for model, name in [(best_log_reg, "Logistic Regression"), (best_rf, "Random Forest")]:
    preds = model.predict(X_test)
    print(f"\n🔍 {name} Performance:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# Register best model in MLflow Model Registry
mlflow.set_experiment("credit-risk-model")
with mlflow.start_run(run_name="best_model_registration"):
    mlflow.sklearn.log_model(best_model, "model")
    mlflow.log_metric("roc_auc", max(log_reg_auc, rf_auc))
    mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id), "CreditRiskModelRegistry")

print("🎉 Best model registered successfully!")
