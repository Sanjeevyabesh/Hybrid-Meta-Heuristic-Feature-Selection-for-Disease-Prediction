"""
utils/preprocessing.py
------------------------
Multi-disease data loading with:
  ✔ Duplicate row removal
  ✔ Missing value imputation (median per column)
  ✔ Preprocessing stats dict returned to UI
  ✔ Dataset-specific target / drop-column config via data_configs.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_configs import get_config


# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str, config: dict = None):
    """
    Load a disease CSV and apply standard cleaning steps.

    Parameters
    ----------
    path   : str   – CSV file path
    config : dict  – entry from DISEASE_CONFIGS; auto-derived from path if None

    Returns
    -------
    X        : pd.DataFrame  – cleaned feature matrix
    y        : pd.Series     – binary target
    stats    : dict          – preprocessing summary for the UI
    """
    df = pd.read_csv(path)

    # Drop identifier / non-predictive columns
    drop_cols = config.get("drop_cols", []) if config else []
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Duplicate removal ─────────────────────────────────────────────────
    n_before            = len(df)
    df                  = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed  = n_before - len(df)

    # ── Missing value imputation (median per numeric column) ──────────────
    missing_before = int(df.isnull().sum().sum())
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    missing_filled = missing_before - int(df.isnull().sum().sum())

    # ── Split features / target ───────────────────────────────────────────
    target = config.get("target", df.columns[-1]) if config else df.columns[-1]

    if target in df.columns:
        y = df[target].astype(int)
        X = df.drop(columns=[target])
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype(int)

    stats = {
        "total_samples":     len(df),
        "total_features":    X.shape[1],
        "duplicates_removed": duplicates_removed,
        "missing_filled":    missing_filled,
    }

    return X, y, stats


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(X, y, random_state: int = 42):
    """
    StandardScale and 80/20 stratified split.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray / pd.Series
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2,
                            random_state=random_state, stratify=y)