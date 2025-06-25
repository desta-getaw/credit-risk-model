# Credit Risk Probability Model for Alternative Data

An end-to-end machine learning system to assess creditworthiness using behavioral alternative data. This project aims to support a **Buy-Now-Pay-Later** service partnership between **Bati Bank** and a fast-growing eCommerce platform by identifying customers eligible for micro-loans.

## 📌 Project Overview

- **Organization**: 10 Academy | Bati Bank  
- **Objective**: Build a Credit Scoring Model using transactional data to predict credit risk and optimize loan offers.  
- **Duration**: 25 June – 01 July 2025  
- **Team**: Mahlet, Rediet, Kerod, Rehmet

## 💼 Credit Scoring Business Understanding

### Why Basel II Accord Matters

The Basel II Accord requires financial institutions to adopt **rigorous risk measurement** and **management practices**. Hence, our model must be:

- **Interpretable** (especially in regulated industries),
- **Transparent** in feature transformation and prediction logic,
- **Documented** for auditability and compliance.

### Why a Proxy Variable?

As the dataset lacks a labeled `default` outcome, we engineer a **proxy target** using RFM (Recency, Frequency, Monetary) analysis. However:

- **Pros**: Enables training supervised models without real default labels.
- **Risks**: Proxy may not reflect true financial behavior; poor proxy quality = high business risk.

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

### ✅ Task 1: Understanding Credit Risk
- Basel II Accord
- Credit scoring without labeled defaults
- Trade-offs in model complexity

### ✅ Task 2: Exploratory Data Analysis (EDA)
- Distribution analysis
- Outlier & missing value handling
- Correlation heatmaps

### ✅ Task 3: Feature Engineering
- Aggregate metrics: Transaction sum, average, count, std. deviation
- Time features: Hour, day, month
- Categorical encoding: Label/One-hot
- Scaling: Standard/MinMax
- Tools: `sklearn.pipeline`, `xverse`, `woe`

### ✅ Task 4: Proxy Target Creation
- RFM metric calculation
- KMeans clustering (k=3)
- Label high-risk group using business logic
- Add `is_high_risk` column to training data

### ✅ Task 5: Model Training & Tracking
- Models: Logistic Regression, Random Forest, GBM
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Hyperparameter tuning
- Experiment tracking with MLflow
- Model registry setup

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

- Basel II: [HKMA Summary](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- RFM Scoring: [Investopedia - Credit Risk](https://www.investopedia.com/terms/c/creditrisk.asp)
- Feature Engineering: [xverse](https://pypi.org/project/xverse/), [woe](https://pypi.org/project/woe/)
- Credit Scorecard: [Shichen Scorecard Guide](https://shichen.name/scorecard/)
- MLFlow: [MLflow Docs](https://mlflow.org/)

## 📅 Key Dates

| Milestone         | Date              |
|-------------------|-------------------|
| Kickoff & Case Review | 25 June 2025     |
| Interim Submission | 29 June 2025     |
| Final Submission   | 01 July 2025     |

## ✍️ Author & Contributors

- **Desta Getaw Mekonnen** – [LinkedIn](https://www.linkedin.com/in/desta-getaw) | [GitHub](https://github.com/Desta2023)
- 10 Academy Tutors: Mahlet, Rediet, Kerod, Rehmet
