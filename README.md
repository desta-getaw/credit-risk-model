# Bati Bank credit Scoring

An end-to-end machine learning system to assess creditworthiness using behavioral alternative data. This project aims to support a **Buy-Now-Pay-Later** service partnership between **Bati Bank** and a fast-growing eCommerce platform by identifying customers eligible for micro-loans.
A project to develop and document a credit scoring model, with a focus on business understanding, interpretability, and alignment with Basel II requirements.
## 📌 Project Overview
As an Analytics Engineer at Bati Bank, I'm tasked with developing a Credit Scoring Model to support a buy-now-pay-later service in partnership with an eCommerce company. The objective is to enable the bank to evaluate potential borrowers and assign them a credit score that reflects their likelihood of defaulting on a loan. This project involves various tasks, from understanding credit risk to building and deploying machine learning models, ensuring compliance with Basel II Capital Accord.
This project explores building a credit risk model to predict the likelihood of borrower default.  
Key objectives:
- Define a proxy variable to categorize users as high risk (bad) or low risk (good).
- Identify observable features that are good predictors of default.
- Develop a model to assign risk probability for new customers.
- Create a credit scoring model from risk probability estimates.
- Predict the optimal loan amount and duration for each customer.
- Understand the **business context and regulatory requirements** (Basel II)
- Discuss the **challenges of limited data and proxy variables**
- Compare **simple interpretable models** with **complex high-performance models**
---
## Introduction 
Credit scoring is a crucial process in financial services, quantifying the risk associated with lending to individuals. Traditional credit scoring models rely on historical data and statistical techniques to predict the likelihood of default. The buy-now-pay-later service aims to provide customers with credit to purchase products, necessitating a robust credit scoring system to minimize financial risk. This project encompasses data analysis, feature engineering, model development, and evaluation to create a reliable credit scoring model.
## 📊 Credit Scoring Business Understanding

### Basel II Accord and the Need for Interpretability

The Basel II Capital Accord emphasizes that banks must **measure, monitor, and explain credit risk**.  
This regulatory focus requires credit risk models to be **transparent, interpretable, and well-documented** so that stakeholders — including auditors and regulators — can:
- Understand how scores are calculated
- Verify the underlying assumptions
- Identify and mitigate potential biases

This need for accountability limits the use of purely "black-box" models that cannot provide clear reasoning behind predictions.

---
### Why Create a Proxy for Default, and What Are the Risks?

In many real-world datasets, especially when using alternative or limited data, we may lack a direct "default" label (such as loans with 90+ days overdue).  
To train supervised models, we create a **proxy variable** that approximates default risk — for example, using recent payment delinquency.

**Risks of relying on a proxy:**
- **Misclassification**: True defaulters may be missed; good customers might be incorrectly labeled as risky.
- **Poor generalization**: Predictions may not accurately reflect actual default behavior, leading to higher losses or lost revenue.
- **Regulatory challenge**: Harder to justify the model to regulators who may question the proxy’s validity.

---
### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In our dataset, there is no direct label indicating whether a customer has defaulted on a loan. Therefore, we need to **engineer a proxy target variable** that approximates default risk — for example, using **RFM (Recency, Frequency, Monetary)** patterns to label disengaged customers as high-risk.

While this approach allows us to train a model, it introduces several **business risks**:

- **Label Noise**: Our proxy may misclassify customers — e.g., those who are inactive but not actually high risk — leading to **false positives**.
- **Model Drift**: Proxy definitions may not generalize well across time or customer segments, especially if behavior patterns shift.
- **Regulatory Risk**: If decisions are made using a proxy that doesn’t align with real-world defaults, it could be challenged during audits.
- **Ethical and Fairness Risks**: Proxy definitions could introduce or amplify biases (e.g., excluding newer customers who haven't had enough time to engage).

Therefore, proxy labels must be defined **carefully, transparently**, and **continuously validated** against actual loan repayment data as it becomes available.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Aspect                     | Logistic Regression with WoE                      | Gradient Boosting Machines (GBMs)                  |
|----------------------------|--------------------------------------------------|----------------------------------------------------|
| **Interpretability**       | High – coefficients and WoE bins are explainable | Low – complex ensemble of decision trees           |
| **Regulatory Compliance**  | Preferred – meets audit and documentation needs  | Challenging – often requires surrogate explanation |
| **Model Performance**      | Adequate – especially when features are well-engineered | High – especially for non-linear relationships |
| **Training Speed**         | Fast and lightweight                             | Slower and more resource-intensive                |
| **Transparency**           | Excellent – easy to trace decision boundaries     | Poor – difficult to trace exact logic             |
| **Use Cases**              | Scorecards, legacy systems                       | Data-rich, flexible fintech applications           |

| Aspect                    | Logistic Regression with WoE (Simple, interpretable) | Gradient Boosting (Complex, high-performance) |
|--------------------------|-----------------------------------------------------|-----------------------------------------------|
| Interpretability         | ✅ Easy to explain                                   | ⚠️ Harder to interpret |
| Documentation & Audit    | ✅ Easier to document and validate                   | ⚠️ Needs extra tools like SHAP for explanation |
| Regulatory Compliance    | ✅ Well-aligned with Basel II transparency           | ⚠️ Potential regulatory concern |
| Predictive Performance   | ❗ Often lower                                       | ✅ Usually higher |
| Deployment & Monitoring  | ✅ Simpler and cheaper                               | ⚠️ More complex and resource-intensive |

In practice, financial institutions often **prioritize interpretable models** for production use to satisfy regulatory and operational requirements, while experimenting with complex models in parallel to test potential performance gains.
In regulated environments like banks, **interpretability often outweighs marginal performance gains**. However, complex models can still be used **in conjunction with explainability tools** (e.g., SHAP, LIME) and should be validated through **model governance frameworks**.

---
### Why a Proxy Variable?

As the dataset lacks a labeled `default` outcome, we engineer a **proxy target** using RFM (Recency, Frequency, Monetary) analysis. However:

- **Pros**: Enables training supervised models without real default labels.
- **Risks**: Proxy may not reflect true financial behavior; poor proxy quality = high business risk.
## 🧰 Model Approach

- **Data Preparation**: Handling missing values, outliers, and data transformation.
- **Feature Engineering**: Weight of Evidence (WoE), binning, and domain-driven features.
- **Modeling**: Comparison between:
  - Logistic Regression (simple, interpretable)
  - Gradient Boosting (complex, high-performance)
- **Evaluation**: ROC-AUC, KS statistic, and business-focused metrics.

### Simple vs Complex Models

| Model Type              | Pros                               | Cons                                |
|-------------------------|-------------------------------------|-------------------------------------|
| Logistic Regression + WoE | Interpretable, auditable           | May underperform on complex patterns |
| Gradient Boosting       | High accuracy, handles non-linearity | Less interpretable, complex to explain |

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: scikit-learn, pandas, xverse, woe, FastAPI, MLflow, flake8, pytest
- **MLOps**: MLflow, GitHub Actions (CI/CD), Docker, FastAPI
- **Containerization**: Docker, docker-compose
- **Deployment**: REST API endpoint for real-time prediction
##  Methodologies
1. Exploratory Data Analysis (EDA):
- Understand the dataset structure.
- Compute summary statistics.
- Visualize numerical and categorical feature distributions.
- Analyze feature correlations.
- Identify missing values and outliers.
2. Feature Engineering:
- Create aggregate features (e.g., total transaction amount, average transaction amount).
- Extract features (e.g., transaction hour, day, month).
- Encode categorical variables using one-hot and label encoding.
- Handle missing values through imputation or removal.
- Normalize/standardize numerical features.
3. Default Estimator and WoE Binning:
- Construct a default estimator using RFMS (Recency, Frequency, Monetary, Seasonality) formalism.
- Classify users as good or bad based on RFMS scores.
- Perform Weight of Evidence (WoE) binning.
4. Modeling:
- Split the data into training and testing sets.
- Select models (e.g., Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Machines).
- Train and tune models using techniques like Grid Search and Random Search.
- Evaluate models using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
## 🗂️ Project Structure

```plaintext
credit-risk-model/
├── .github/workflows/ci.yml          # GitHub Actions for CI/CD
├── data/
│   ├── raw/                          # Raw transaction data
│   └── processed/                    # Cleaned and engineered data
├── notebooks/
│   └── 1.0-eda.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── data_processing.py            # Data cleaning, RFM, encoding
│   ├── train.py                      # Model training & tracking
│   ├── predict.py                    # Batch inference
│   └── api/
│       ├── main.py                   # FastAPI entrypoint
│       └── pydantic_models.py        # Request/Response schemas
├── tests/
│   └── test_data_processing.py       # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔍 Tasks Breakdown

### ✅ Task 1: Understanding Credit Risk:
- Basel II Accord
- Credit scoring without labeled defaults
- Trade-offs in model complexity

### ✅ Task 2: Exploratory Data Analysis (EDA)
- Load the dataset and examine its structure.
- Calculate summary statistics for numerical features.
- Visualize distributions of numerical and categorical features.
- Perform correlation analysis.
- Identify and handle missing values.
- Detect and address outliers.
- Distribution analysis
- Correlation heatmaps

### ✅ Task 3: Feature Engineering
- Aggregate metrics: Transaction sum, average, count, std. deviation
- Time features: Hour, day, month
- Categorical encoding: Label/One-hot
- Scaling: Standard/MinMax
- Aggregate features like total, average, and standard deviation of transaction amounts per customer.
- Extract temporal features such as transaction hour, day, month, and year.
- Encode categorical variables using appropriate techniques.
- Handle missing values through imputation or removal.
- Normalize and standardize numerical features.
- Tools: `sklearn.pipeline`, `xverse`, `woe`

### ✅ Task 4: Proxy Target Creation & Default Estimator and WoE Binning:
- RFM metric calculation
- KMeans clustering (k=3)
- Develop a default estimator based on RFMS formalism.
- Assign users into good and bad categories.
- Implement WoE binning to transform categorical variables
- Label high-risk group using business logic
- Add `is_high_risk` column to training data

### ✅ Task 5: Model Training & Tracking
- Models: Logistic Regression, Random Forest, GBM
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Hyperparameter tuning
- Experiment tracking with MLflow
- Model registry setup
- Split the dataset into training and testing sets.
- Select and train multiple models.
- Tune hyperparameters to optimize model performance.
- Evaluate models using relevant metrics to select the best-performing model.

### ✅ Task 6: API Deployment & CI/CD
- FastAPI with `/predict` endpoint
- Docker & docker-compose
- CI pipeline:
  - Linting with flake8
  - Unit tests via pytest

## 🚀 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

# Create virtual environment & install requirements
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run EDA notebook
jupyter notebook notebooks/1.0-eda.ipynb

# Train model
python src/train.py

# Serve API
uvicorn src.api.main:app --reload
```

To run using Docker:

```bash
docker-compose up --build
```

To test and lint:

```bash
pytest
flake8 src/
```

## 📈 Results and Deliverables

- 📘 Blog post / PDF Report (Link)
- 🔗 [GitHub Repo](https://github.com/your-username/credit-risk-model)
- 🧪 MLFlow dashboard with all experiments
- 🚀 FastAPI deployed API for risk prediction

## 🧠 Learning Outcomes

| Domain             | Tools/Techniques                        |
|--------------------|------------------------------------------|
| Credit Modeling    | WoE, IV, RFM Clustering, Proxy Target   |
| Data Engineering   | Pipelines, Feature Extraction, Encoding |
| ML Development     | Model Selection, Hyperparameter Tuning |
| MLOps              | MLflow, Docker, CI/CD (GitHub Actions) |
| Software Engineering | Testing, FastAPI, Pydantic, Logging    |

## 📚 References

- [Basel II Capital Accord (PDF)](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring (HKMA)](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank – Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Developing a Credit Risk Model – Medium article](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Corporate Finance Institute: Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Risk Officer – Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)
- Basel II: [HKMA Summary](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- RFM Scoring: [Investopedia - Credit Risk](https://www.investopedia.com/terms/c/creditrisk.asp)
- Feature Engineering: [xverse](https://pypi.org/project/xverse/), [woe](https://pypi.org/project/woe/)
- Credit Scorecard: [Shichen Scorecard Guide](https://shichen.name/scorecard/)
- MLFlow: [MLflow Docs](https://mlflow.org/)

---

## ✍️ Author & Contributors

- **Desta Getaw Mekonnen** – [LinkedIn](https://www.linkedin.com/in/desta-getaw) | [GitHub](https://github.com/Desta2023)
- 10 Academy
