"""
utils/metrics.py
----------------
Compute classification metrics for baseline and optimised models.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_test, y_pred):
    """
    Return a dict containing:
      - accuracy           : float
      - confusion_matrix   : np.ndarray (2-D)
      - classification_report : str  (formatted text)
    """
    acc  = accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        'accuracy':              acc,
        'confusion_matrix':      cm,
        'classification_report': report,
    }
