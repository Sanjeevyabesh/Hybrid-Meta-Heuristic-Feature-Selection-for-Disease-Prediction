"""
main.py  —  MEDOPT multi-problem dispatcher
--------------------------------------------
Three entry-points consumed by app.py:

  run_problem1  – Feature Selection        (GWO / PSO / WOA)
  run_problem2  – Threshold Optimisation   (PSO)
  run_problem4  – Hyperparameter Tuning    (PSO)

All functions accept a disease_name string and return a results dict.
"""

import numpy as np

from data_configs               import get_config
from utils.preprocessing        import load_data, preprocess

from models.train_model         import train_model
from optimization.gwo           import GWO
from optimization.pso           import PSO
from optimization.woa           import WOA
from problems.problem2_threshold_opt  import ThresholdOptimizer
from problems.problem4_hyperparam_opt import HyperparamOptimizer

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load(disease_name: str):
    cfg = get_config(disease_name)
    X_df, y, stats = load_data(cfg["file"], cfg)
    X_tr, X_te, y_tr, y_te = preprocess(X_df, y, random_state=RANDOM_STATE)
    return cfg, X_df, y, stats, X_tr, X_te, y_tr, y_te


def _build_optimizer(algorithm: str, n_features: int, fast: bool):
    n_agents = 5 if fast else 10
    n_iter   = 10 if fast else 20
    if algorithm == "GWO":
        return GWO(n_wolves=n_agents,    n_features=n_features,
                   max_iter=n_iter,      random_state=RANDOM_STATE)
    if algorithm == "PSO":
        return PSO(n_particles=n_agents, n_features=n_features,
                   max_iter=n_iter,      random_state=RANDOM_STATE)
    if algorithm == "WOA":
        return WOA(n_whales=n_agents,    n_features=n_features,
                   max_iter=n_iter,      random_state=RANDOM_STATE)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _baseline(X_tr, X_te, y_tr, y_te, n_features):
    all_feats = list(range(n_features))
    acc, cm, report, model = train_model(
        X_tr, X_te, y_tr, y_te, all_feats, random_state=RANDOM_STATE
    )
    return acc, cm, report, model


# ─────────────────────────────────────────────────────────────────────────────
# Problem 1 — Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
def run_problem1(disease_name: str, algorithm: str = "GWO",
                 fast_mode: bool = False) -> dict:
    """
    Run one MHO algorithm for feature selection.

    Returns
    -------
    dict with all data needed by app.py (model cached for P2/P4 reuse)
    """
    cfg, X_df, y, stats, X_tr, X_te, y_tr, y_te = _load(disease_name)
    feature_names = list(X_df.columns)
    n_features    = len(feature_names)

    # Baseline (all features, default RF)
    base_acc, base_cm, base_report, _ = _baseline(X_tr, X_te, y_tr, y_te, n_features)

    # Optimise
    optimizer = _build_optimizer(algorithm, n_features, fast_mode)
    best_sol, fitness_curve = optimizer.optimize(X_tr, X_te, y_tr, y_te)
    selected = np.where(best_sol > 0.5)[0]

    # Final RF on selected features
    opt_acc, opt_cm, opt_report, model = train_model(
        X_tr, X_te, y_tr, y_te, selected, random_state=RANDOM_STATE
    )



    return {
        # disease / run meta
        "disease":        disease_name,
        "algorithm":      algorithm,
        "class_names":    cfg["class_names"],
        "feature_names":  feature_names,
        "stats":          stats,
        "X_df":           X_df,
        # baseline
        "baseline_acc":   base_acc,
        "baseline_cm":    base_cm,
        "baseline_report": base_report,
        # optimised
        "optimized_acc":     opt_acc,
        "confusion_matrix":  opt_cm,
        "classification_report": opt_report,
        "selected_indices":  selected,
        "fitness_curve":     fitness_curve,

        # cached objects (for P2 / P4 reuse)
        "model":   model,
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Problem 2 — Clinical Threshold Optimisation
# ─────────────────────────────────────────────────────────────────────────────
def run_problem2(p1: dict, alpha: float = 0.75) -> dict:
    """
    Run PSO threshold optimiser using the model trained in Problem 1.

    Parameters
    ----------
    p1    : result dict from run_problem1()
    alpha : sensitivity weight (0.5 = balanced, 1.0 = max sensitivity)
    """
    from sklearn.metrics import confusion_matrix, accuracy_score

    model    = p1["model"]
    X_te     = p1["X_test"]
    y_te     = p1["y_test"]
    selected = p1["selected_indices"]

    y_prob = model.predict_proba(X_te[:, selected])[:, 1]

    optimizer = ThresholdOptimizer(n_particles=20, max_iter=30,
                                   alpha=alpha, random_state=RANDOM_STATE)
    opt_thresh, thresh_curve = optimizer.optimize(y_prob, y_te)

    # Metrics at default (0.5) threshold
    y_default = (y_prob >= 0.5).astype(int)
    cm_default = confusion_matrix(y_te, y_default, labels=[0, 1])
    tn, fp, fn, tp = cm_default.ravel()
    sens_default = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_default = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Metrics at optimal threshold
    y_opt = (y_prob >= opt_thresh).astype(int)
    cm_opt = confusion_matrix(y_te, y_opt, labels=[0, 1])
    tn2, fp2, fn2, tp2 = cm_opt.ravel()
    sens_opt = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0.0
    spec_opt = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0

    return {
        "optimal_threshold":     opt_thresh,
        "threshold_curve":       thresh_curve,
        "y_prob":                y_prob,
        "y_test":                y_te,
        # default-threshold stats
        "default_threshold":     0.5,
        "default_sensitivity":   sens_default,
        "default_specificity":   spec_default,
        "default_fn":            int(fn),
        "cm_default":            cm_default,
        # optimal-threshold stats
        "opt_sensitivity":       sens_opt,
        "opt_specificity":       spec_opt,
        "opt_fn":                int(fn2),
        "cm_optimal":            cm_opt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Problem 4 — Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────
def run_problem4(p1: dict, fast_mode: bool = False) -> dict:
    """
    PSO tunes RandomForest hyperparameters on the selected features from P1.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    selected = p1["selected_indices"]
    X_tr     = p1["X_train"]
    X_te     = p1["X_test"]
    y_tr     = p1["y_train"]
    y_te     = p1["y_test"]

    n_iter = 15 if fast_mode else 20

    optimizer = HyperparamOptimizer(n_particles=15, max_iter=n_iter,
                                    random_state=RANDOM_STATE)
    best_params, best_cv, hp_curve = optimizer.optimize(X_tr, y_tr, selected)

    # Train final model with tuned params
    tuned_model = RandomForestClassifier(random_state=RANDOM_STATE, **best_params)
    tuned_model.fit(X_tr[:, selected], y_tr)
    y_pred = tuned_model.predict(X_te[:, selected])

    tuned_acc    = accuracy_score(y_te, y_pred)
    tuned_cm     = confusion_matrix(y_te, y_pred)
    tuned_report = classification_report(y_te, y_pred, zero_division=0)

    # Default RF for comparison
    default_rf = RandomForestClassifier(random_state=RANDOM_STATE)
    default_rf.fit(X_tr[:, selected], y_tr)
    default_acc = accuracy_score(y_te, default_rf.predict(X_te[:, selected]))

    default_params = {
        "n_estimators": 100, "max_depth": None,
        "min_samples_split": 2, "min_samples_leaf": 1,
    }

    return {
        "best_params":    best_params,
        "default_params": default_params,
        "best_cv_score":  best_cv,
        "tuned_acc":      tuned_acc,
        "default_acc":    default_acc,
        "tuned_cm":       tuned_cm,
        "tuned_report":   tuned_report,
        "hp_curve":       hp_curve,
        "tuned_model":    tuned_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Patient Diagnosis  (uses cached P1 model + selected features)
# ─────────────────────────────────────────────────────────────────────────────
def diagnose_patient(patient_values: np.ndarray, p1: dict,
                     p4: dict = None, threshold: float = 0.5) -> dict:
    """
    Run inference on one patient.

    Parameters
    ----------
    patient_values : np.ndarray – raw (unscaled) feature vector (all features)
    p1             : result from run_problem1
    p4             : result from run_problem4 (use tuned model if provided)
    threshold      : decision threshold (0.5 default; use P2 optimal if available)

    Returns
    -------
    dict with risk_score, diagnosis, confidence, explanation
    """
    from sklearn.preprocessing import StandardScaler
    from utils.explainability import get_explanation

    model    = p4["tuned_model"] if p4 else p1["model"]
    selected = p1["selected_indices"]
    feat_names = p1["feature_names"]

    # Scale using training-set statistics (re-fit scaler on X_train)
    scaler = StandardScaler()
    scaler.fit(p1["X_train"])

    patient_scaled   = scaler.transform(patient_values.reshape(1, -1))
    patient_selected = patient_scaled[:, selected]

    prob      = float(model.predict_proba(patient_selected)[0, 1])
    pred      = int(prob >= threshold)
    risk_score = round(prob * 100, 1)

    class_names = p1["class_names"]
    diagnosis   = class_names[pred]

    if risk_score >= 70:
        warning = "⚠️ HIGH RISK — Immediate clinical attention recommended"
    elif risk_score >= 40:
        warning = "🔶 MODERATE RISK — Schedule follow-up tests"
    else:
        warning = "✅ LOW RISK — Routine monitoring advised"

    selected_feat_names = [feat_names[i] for i in selected]
    explanation = get_explanation(
        model,
        p1["X_train"][:, selected],
        patient_selected[0],
        selected_feat_names,
    )

    return {
        "risk_score":   risk_score,
        "diagnosis":    diagnosis,
        "confidence":   round(prob * 100, 1),
        "warning":      warning,
        "threshold":    threshold,
        "explanation":  explanation,
        "selected_features": selected_feat_names,
    }