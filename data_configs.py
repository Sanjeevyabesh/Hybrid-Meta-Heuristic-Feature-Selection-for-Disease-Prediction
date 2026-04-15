"""
data_configs.py
---------------
Centralised configuration for all three disease datasets.
Defines file paths, target columns, and class labels.
"""

DISEASE_CONFIGS = {

    # ── Parkinson's Disease ────────────────────────────────────────────────
    "Parkinson's Disease": {
        "file":        "data/parkinsons.csv",
        "target":      "status",
        "drop_cols":   ["name"],
        "class_names": ["Healthy", "Parkinson's"],
        "description": "Voice biomarker analysis for early Parkinson's detection",
        "emoji":       "🧠",
    },

    # ── Diabetes ───────────────────────────────────────────────────────────
    "Diabetes": {
        "file":        "data/diabetes.csv",
        "target":      "Outcome",
        "drop_cols":   [],
        "class_names": ["No Diabetes", "Diabetic"],
        "description": "PIMA Indians Diabetes early-screening panel",
        "emoji":       "💉",
    },

    # ── Heart Disease ──────────────────────────────────────────────────────
    "Heart Disease": {
        "file":        "data/heart_cleveland_upload.csv",
        "target":      "condition",
        "drop_cols":   [],
        "class_names": ["No Disease", "Heart Disease"],
        "description": "Cleveland cardiac risk assessment panel",
        "emoji":       "❤️",
    },
}


def get_config(disease_name: str) -> dict:
    """Retrieve config dict for a given disease name (raises KeyError if not found)."""
    if disease_name not in DISEASE_CONFIGS:
        raise KeyError(f"Unknown disease '{disease_name}'. "
                       f"Valid options: {list(DISEASE_CONFIGS.keys())}")
    return DISEASE_CONFIGS[disease_name]
