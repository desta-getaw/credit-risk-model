# 🧪 tests/test_data_processing.py - Unit Tests for Data Processing Pipeline

import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_processing import create_feature_pipeline
from src.data_processing import create_feature_pipeline
#from src.feature_engineering import create_feature_pipeline


# Path to raw data
DATA_PATH = "data/raw/data.csv"

@pytest.fixture
def raw_data():
    """Fixture to load raw data for tests"""
    return pd.read_csv(DATA_PATH)

def test_data_loading(raw_data):
    """Test if raw data loads properly"""
    assert raw_data is not None
    assert not raw_data.empty
    assert isinstance(raw_data, pd.DataFrame)

def test_pipeline_structure():
    """Test if create_feature_pipeline returns a valid sklearn Pipeline"""
    pipeline = create_feature_pipeline()
    assert isinstance(pipeline, Pipeline)
    expected_steps = [
        'aggregate_transaction_features',
        'extract_datetime_features',
        'encode_categorical',
        'label_encode_categorical',
        'handle_missing',
        'normalize_and_standardize'
    ]
    actual_steps = [name for name, _ in pipeline.steps]
    for step in expected_steps:
        assert step in actual_steps

def test_pipeline_output_not_empty(raw_data):
    """Test if pipeline produces a non-empty transformed dataframe"""
    pipeline = create_feature_pipeline()
    transformed = pipeline.fit_transform(raw_data)
    assert transformed is not None
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] > 0
    assert transformed.shape[1] > 0

def test_pipeline_handles_missing_values(raw_data):
    """Test if pipeline handles missing values gracefully"""
    df_missing = raw_data.copy()
    df_missing.iloc[0, 0] = None  # Introduce missing value
    pipeline = create_feature_pipeline()
    transformed = pipeline.fit_transform(df_missing)
    assert transformed is not None
    assert transformed.shape[0] == df_missing.shape[0]


def create_feature_pipeline():
    return None