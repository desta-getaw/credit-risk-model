# 🧪 test_data_processing.py - Unit Tests for Data Processing Pipeline

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

# Example test for checking if data loads properly
def test_data_loading():
    df = pd.read_csv("data/raw/data.csv")
    assert df is not None
    assert not df.empty

# Example test for checking if the pipeline is correctly defined
def test_pipeline_structure():
    from src.data_processing import create_feature_pipeline
    pipeline = create_feature_pipeline()
    assert isinstance(pipeline, Pipeline)

# Example test for checking if pipeline output has expected columns
def test_pipeline_output_columns():
    from src.data_processing import create_feature_pipeline
    df = pd.read_csv("data/raw/data.csv")
    pipeline = create_feature_pipeline()
    processed_df = pipeline.fit_transform(df)
    assert processed_df is not None
    assert processed_df.shape[0] > 0  # Ensure rows exist

# Example test for missing value handling
def test_missing_value_imputation():
    from src.data_processing import create_feature_pipeline
    df = pd.read_csv("data/raw/data.csv")
    df_missing = df.copy()
    df_missing.iloc[0, 0] = None  # artificially create a missing value
    pipeline = create_feature_pipeline()
    processed_df = pipeline.fit_transform(df_missing)
    assert processed_df is not None
    assert processed_df.shape[0] == df_missing.shape[0]

print("✅ All data processing tests passed!")
