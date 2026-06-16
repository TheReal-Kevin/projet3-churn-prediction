"""Data cleaning and preprocessing utilities for the Telco Churn dataset."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    return pd.read_csv(path)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to float; fill blanks with 0 (new customers with no charges yet)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """Remove customerID — carries no predictive value."""
    return df.drop(columns=["customerID"], errors="ignore")


def encode_target(df: pd.DataFrame, col: str = "Churn") -> pd.DataFrame:
    """Map Yes/No target column to 1/0."""
    df = df.copy()
    df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all binary Yes/No columns as 1/0."""
    df = df.copy()
    binary_cols = [
        c for c in df.columns
        if df[c].dtype == object
        and set(df[c].dropna().unique()).issubset({"Yes", "No"})
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot Encode remaining object columns."""
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def normalize_numeric(
    df: pd.DataFrame,
    cols: list,
    scaler: StandardScaler = None
) -> tuple:
    """
    StandardScale numeric columns.
    Pass a fitted scaler to apply transform only (for test set).
    Returns (df, fitted_scaler).
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


def split_features_target(df: pd.DataFrame, target: str = "Churn"):
    """Return X, y from the cleaned dataframe."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def full_pipeline(raw_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    End-to-end preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    df = load_data(raw_path)
    df = fix_total_charges(df)
    df = drop_irrelevant(df)
    df = encode_target(df)
    df = encode_binary_columns(df)
    df = encode_categorical(df)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df, scaler = normalize_numeric(df, num_cols)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()
