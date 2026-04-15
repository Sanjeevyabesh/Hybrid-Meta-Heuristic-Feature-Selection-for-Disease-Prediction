"""
utils/explainability.py
------------------------
SHAP-based feature explanations for the trained RF model.
Falls back gracefully to RF feature importances if SHAP is not installed.
"""

import numpy as np


def get_explanation(model, X_train: np.ndarray, X_instance: np.ndarray,
                    feature_names: list) -> dict:
    """
    Parameters
    ----------
    model         : fitted RandomForestClassifier
    X_train       : training data (used to build SHAP background)
    X_instance    : 1-D array of ONE patient's feature values (selected features only)
    feature_names : list of selected feature column names

    Returns
    -------
    dict with keys:
      type         : 'shap' | 'importance'
      values       : np.ndarray – one value per feature (positive = pushes toward disease)
      base_value   : float     – expected model output (SHAP only)
      feature_names: list[str]
      available    : bool      – False if SHAP not installed
    """
    instance_2d = X_instance.reshape(1, -1)

    try:
        import shap

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance_2d)

        # shap_values is a list [class0_arr, class1_arr] for RF classifiers
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv     = shap_values[1][0]         # class-1 (disease) shap values
            base   = explainer.expected_value
            base   = base[1] if hasattr(base, "__len__") else float(base)
        else:
            sv   = shap_values[0]
            base = float(explainer.expected_value)

        return {
            "type":         "shap",
            "values":       sv,
            "base_value":   base,
            "feature_names": feature_names,
            "available":    True,
        }

    except ImportError:
        # Graceful fallback — RF feature importance
        importances = model.feature_importances_
        return {
            "type":         "importance",
            "values":       importances,
            "base_value":   None,
            "feature_names": feature_names,
            "available":    False,
        }
