"""
utils/visualization.py
-----------------------
All matplotlib figure generators for the MEDOPT Streamlit UI.
No seaborn required — pure matplotlib.

Functions
─────────
plot_heatmap              – feature correlation heatmap
plot_confusion_matrix     – styled CM with cell annotations
plot_convergence          – multi-algorithm convergence curves
plot_accuracy_bar         – horizontal accuracy bar chart
plot_roc_curve            – ROC with AUC + optimal threshold marker
plot_risk_gauge           – colour-gradient risk score bar
plot_shap_bar             – SHAP / importance horizontal bar chart
plot_threshold_analysis   – sensitivity & specificity vs threshold
plot_hyperparam_comparison – before/after hyperparameter table chart
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

_PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


# ── 1. Correlation Heatmap ────────────────────────────────────────────────────
def plot_heatmap(X_df, figsize=(14, 10)):
    corr = X_df.corr()
    n    = len(corr.columns)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=9)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(corr.values[i,j]) > 0.6 else "black")
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout(); return fig


# ── 2. Confusion Matrix ───────────────────────────────────────────────────────
def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", figsize=(5, 4)):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#333333")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout(); return fig


# ── 3. Convergence Curve ──────────────────────────────────────────────────────
def plot_convergence(curves: dict, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    for idx, (label, curve) in enumerate(curves.items()):
        ax.plot(range(1, len(curve) + 1), curve,
                label=label, color=_PALETTE[idx % len(_PALETTE)],
                linewidth=2.2, marker="o", markersize=3)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Fitness (Accuracy)", fontsize=12)
    ax.set_title("Optimiser Convergence", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim([0, 1.05]); ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout(); return fig


# ── 4. Accuracy Bar Chart ─────────────────────────────────────────────────────
def plot_accuracy_bar(labels: list, values: list, figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=_PALETTE[:len(labels)],
                   edgecolor="none", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=11, fontweight="bold")
    ax.set_xlim([0, 1.12])
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.invert_yaxis(); plt.tight_layout(); return fig


# ── 5. ROC Curve ──────────────────────────────────────────────────────────────
def plot_roc_curve(y_test, y_prob, optimal_threshold=None,
                   label="Model", figsize=(6, 5)):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=_PALETTE[0], lw=2,
            label=f"{label}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1,
            label="Random classifier")
    if optimal_threshold is not None:
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        ax.scatter(fpr[idx], tpr[idx], color="red", s=180, zorder=5,
                   label=f"PSO threshold = {optimal_threshold:.2f}",
                   edgecolors="black", linewidths=1.2)
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout(); return fig


# ── 6. Risk Score Gauge ───────────────────────────────────────────────────────
def plot_risk_gauge(risk_score: float, figsize=(8, 2)):
    fig, ax = plt.subplots(figsize=figsize)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="RdYlGn_r", extent=[0, 100, 0, 1])
    ax.axvline(x=risk_score, color="black", lw=3)
    ax.scatter([risk_score], [0.5], color="black", s=250, zorder=5)
    zone_style = dict(ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(16, 0.5, "LOW\nRISK",      color="white", **zone_style)
    ax.text(50, 0.5, "MODERATE\nRISK", color="black", **zone_style)
    ax.text(84, 0.5, "HIGH\nRISK",     color="white", **zone_style)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Risk Score: {risk_score:.1f} / 100",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(); return fig


# ── 7. SHAP / Importance Bar ──────────────────────────────────────────────────
def plot_shap_bar(values: np.ndarray, feature_names: list,
                  title="Feature Importance (SHAP)", figsize=(8, 5)):
    order = np.argsort(np.abs(values))
    sorted_vals  = values[order]
    sorted_names = [feature_names[i] for i in order]
    colors = ["#F44336" if v > 0 else "#2196F3" for v in sorted_vals]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(sorted_names, sorted_vals, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP Value → pushes toward disease (+) or healthy (–)",
                  fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout(); return fig


# ── 8. Sensitivity & Specificity vs Threshold ─────────────────────────────────
def plot_threshold_analysis(y_test, y_prob, optimal_threshold=None, figsize=(7, 4)):
    from sklearn.metrics import confusion_matrix
    thresholds  = np.linspace(0.05, 0.95, 50)
    sens_list, spec_list = [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        else:
            sens_list.append(0); spec_list.append(0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, sens_list, color=_PALETTE[1], lw=2, label="Sensitivity (Recall)")
    ax.plot(thresholds, spec_list, color=_PALETTE[0], lw=2, label="Specificity")
    if optimal_threshold is not None:
        ax.axvline(optimal_threshold, color="red", linestyle="--", lw=1.8,
                   label=f"PSO optimal = {optimal_threshold:.2f}")
    ax.axvline(0.5, color="grey", linestyle=":", lw=1.2, label="Default = 0.50")
    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Sensitivity & Specificity vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout(); return fig


# ── 9. Hyperparameter Comparison ──────────────────────────────────────────────
def plot_hyperparam_comparison(default_params: dict, tuned_params: dict,
                                default_acc: float, tuned_acc: float,
                                figsize=(8, 4)):
    labels = list(default_params.keys()) + ["CV Accuracy"]
    def_vals   = list(default_params.values()) + [default_acc * 100]
    tuned_vals = list(tuned_params.values())   + [tuned_acc  * 100]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w/2, def_vals,   width=w, label="Default RF",  color=_PALETTE[0], alpha=0.85)
    ax.bar(x + w/2, tuned_vals, width=w, label="PSO-Tuned RF", color=_PALETTE[1], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_title("Default vs PSO-Tuned Hyperparameters", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout(); return fig
