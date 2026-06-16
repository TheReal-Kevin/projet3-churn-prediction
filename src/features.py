"""Optional feature engineering extensions."""

import pandas as pd


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add total number of subscribed services as a feature."""
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    existing = [c for c in service_cols if c in df.columns]
    df = df.copy()
    df["service_count"] = df[existing].apply(
        lambda row: sum(1 for v in row if str(v) in {"Yes", "1", 1}), axis=1
    )
    return df


def add_charge_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add TotalCharges / tenure ratio to capture average spend trajectory."""
    df = df.copy()
    df["charge_per_month"] = df.apply(
        lambda r: r["TotalCharges"] / r["tenure"] if r["tenure"] > 0 else r["MonthlyCharges"],
        axis=1,
    )
    return df
