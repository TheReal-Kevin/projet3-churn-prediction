"""Extensions optionnelles de feature engineering."""

import pandas as pd


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le nombre total de services souscrits comme nouvelle feature."""
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    existing = [c for c in service_cols if c in df.columns]
    df = df.copy()
    # Accepte "Yes", "1" ou 1 pour rester valide, que l'encodage binaire ait déjà
    # été appliqué en amont ou non.
    df["service_count"] = df[existing].apply(
        lambda row: sum(1 for v in row if str(v) in {"Yes", "1", 1}), axis=1
    )
    return df


def add_charge_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le ratio TotalCharges / tenure pour capter la trajectoire de dépense moyenne."""
    df = df.copy()
    # Retombe sur MonthlyCharges quand tenure vaut 0, pour éviter une division par zéro.
    df["charge_per_month"] = df.apply(
        lambda r: r["TotalCharges"] / r["tenure"] if r["tenure"] > 0 else r["MonthlyCharges"],
        axis=1,
    )
    return df
