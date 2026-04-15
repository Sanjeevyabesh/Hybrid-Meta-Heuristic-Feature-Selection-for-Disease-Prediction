"""
models/train_model.py
----------------------
Reusable RandomForest training + full evaluation helper.

Previously this file was unused dead code; it is now the single place
where model training lives and is imported by main.py.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_model(X_train, X_test, y_train, y_test,
                selected_features, random_state: int = 42):
    """
    Train a RandomForestClassifier on the selected features and evaluate it.

    Parameters
    ----------
    X_train / X_test       : np.ndarray  – pre-split, standardised feature matrices
    y_train / y_test       : array-like  – class labels
    selected_features      : array-like  – column indices to use
    random_state           : int         – seed for the RF (reproducibility)

    Returns
    -------
    acc     : float       – accuracy on the test set
    cm      : np.ndarray  – confusion matrix (n_classes × n_classes)
    report  : str         – formatted classification report
    model   : fitted RandomForestClassifier (for further use if needed)
    """
    if len(selected_features) == 0:
        # Edge-case: no features selected → return zero accuracy
        n_classes = len(np.unique(y_test))
        return 0.0, np.zeros((n_classes, n_classes), dtype=int), "No features selected.", None

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train[:, selected_features], y_train)
    preds = model.predict(X_test[:, selected_features])

    acc    = accuracy_score(y_test, preds)
    cm     = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    return acc, cm, report, model