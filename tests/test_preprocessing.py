"""Tests unitaires sur les fonctions de preprocessing — src/preprocessing.py"""

import pandas as pd
import pytest
from src.preprocessing import (
    fix_total_charges,
    drop_irrelevant,
    encode_target,
    encode_binary_columns,
    encode_categorical,
)


# ── Données de test ──────────────────────────────────────────────────────────

@pytest.fixture
def df_brut():
    """Petit DataFrame qui simule les données brutes du dataset Telco."""
    return pd.DataFrame({
        "customerID": ["001", "002", "003"],
        "tenure": [1, 12, 24],
        "MonthlyCharges": [29.5, 55.0, 80.0],
        "TotalCharges": ["29.5", " ", "1920.0"],  # espace = valeur manquante
        "Churn": ["Yes", "No", "Yes"],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fix_total_charges_convertit_en_float(df_brut):
    """Les espaces vides doivent devenir 0.0, les vrais nombres doivent être convertis."""
    df = fix_total_charges(df_brut)

    assert df["TotalCharges"].dtype == float
    assert df.loc[1, "TotalCharges"] == 0.0  # l'espace était une valeur manquante
    assert df.loc[0, "TotalCharges"] == 29.5


def test_drop_irrelevant_supprime_customerid(df_brut):
    """La colonne customerID doit être supprimée car elle n'a pas de valeur prédictive."""
    df = drop_irrelevant(df_brut)

    assert "customerID" not in df.columns


def test_encode_target_yes_devient_1(df_brut):
    """La cible Churn : Yes → 1, No → 0."""
    df = encode_target(df_brut)

    assert df.loc[0, "Churn"] == 1  # était "Yes"
    assert df.loc[1, "Churn"] == 0  # était "No"


def test_encode_binary_columns_yes_no(df_brut):
    """Les colonnes binaires Yes/No doivent être encodées en 1/0."""
    df = encode_binary_columns(df_brut)

    assert df.loc[0, "Partner"] == 1   # "Yes" → 1
    assert df.loc[1, "Partner"] == 0   # "No" → 0
    assert df.loc[2, "Dependents"] == 1  # "Yes" → 1


def test_encode_categorical_supprime_colonnes_texte(df_brut):
    """Après l'encodage OHE, il ne doit plus rester de colonnes de type object."""
    # On encode d'abord les colonnes binaires et la cible pour éviter qu'elles interfèrent
    df = encode_target(df_brut)
    df = encode_binary_columns(df)
    df = drop_irrelevant(df)
    df = encode_categorical(df)

    colonnes_texte = df.select_dtypes(include="object").columns.tolist()
    assert len(colonnes_texte) == 0, f"Colonnes texte restantes : {colonnes_texte}"
