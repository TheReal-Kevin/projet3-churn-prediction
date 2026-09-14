"""Fonctions de nettoyage et de préparation des données pour le dataset Telco Churn."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Colonnes continues normalisées (StandardScaler) — partagées entre le pipeline
# d'entraînement et l'API pour garantir une transformation identique aux deux endroits.
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset brut depuis un fichier CSV."""
    return pd.read_csv(path)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit TotalCharges en float ; remplace les valeurs manquantes par 0
    (clients tout juste arrivés, sans facture encore émise)."""
    df = df.copy()
    # La colonne brute est stockée en texte ; quelques lignes contiennent un espace
    # au lieu d'un nombre pour les clients dont l'ancienneté (tenure) vaut 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime customerID — n'a aucune valeur prédictive."""
    return df.drop(columns=["customerID"], errors="ignore")


def encode_target(df: pd.DataFrame, col: str = "Churn") -> pd.DataFrame:
    """Encode la colonne cible Yes/No en 1/0."""
    df = df.copy()
    df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode toutes les colonnes binaires Yes/No en 1/0."""
    df = df.copy()
    # is_string_dtype couvre à la fois l'ancien dtype "object" et le nouveau dtype
    # "str" natif de pandas (depuis pandas 3.0), pour rester valide sur les deux
    # versions.
    binary_cols = [
        c
        for c in df.columns
        if pd.api.types.is_string_dtype(df[c])
        and set(df[c].dropna().unique()).issubset({"Yes", "No"})
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes textuelles restantes en One-Hot Encoding."""
    cat_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    # drop_first évite le piège de la variable muette (colinéarité parfaite entre
    # les colonnes indicatrices générées).
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def normalize_numeric(df: pd.DataFrame, cols: list, scaler: StandardScaler = None) -> tuple:
    """
    Normalise (StandardScaler) les colonnes numériques.
    Fournir un scaler déjà entraîné pour appliquer uniquement la transformation
    (cas du jeu de test).
    Retourne (df, scaler entraîné).
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        # Transformation seule — réentraîner le scaler sur le jeu de test ferait
        # fuiter ses statistiques dans l'étape de préparation des données.
        df[cols] = scaler.transform(df[cols])
    return df, scaler


def split_features_target(df: pd.DataFrame, target: str = "Churn"):
    """Retourne X, y à partir du dataframe nettoyé."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def full_pipeline(raw_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline complet de préparation des données.
    Retourne X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    df = load_data(raw_path)
    df = fix_total_charges(df)
    df = drop_irrelevant(df)
    df = encode_target(df)
    df = encode_binary_columns(df)
    df = encode_categorical(df)

    df, scaler = normalize_numeric(df, NUMERIC_COLS)

    X, y = split_features_target(df)
    # stratify=y conserve la même proportion de churn dans train et test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()
