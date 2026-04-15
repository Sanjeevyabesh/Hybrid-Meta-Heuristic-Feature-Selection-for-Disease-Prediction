"""
app.py  —  MEDOPT Clinical Screening System
=============================================
Multi-page Streamlit UI with sidebar navigation.

Pages
─────
🏠 Overview            – System introduction + per-disease stats
🔬 Problem 1           – Feature Selection  (GWO / PSO / WOA)
🎯 Problem 2           – Clinical Threshold Optimisation (PSO)
⚙️  Problem 4           – Hyperparameter Tuning (PSO)
🩺 Patient Diagnosis   – Single-patient risk score + SHAP
📊 Comparison          – Cross-disease/algorithm summary
"""

import os
import numpy as np
import streamlit as st

from data_configs    import DISEASE_CONFIGS
from main            import run_problem1, run_problem2, run_problem4, diagnose_patient
from utils.preprocessing  import load_data
from utils.visualization  import (
    plot_heatmap, plot_confusion_matrix, plot_convergence,
    plot_accuracy_bar, plot_roc_curve, plot_risk_gauge,
    plot_shap_bar, plot_threshold_analysis, plot_hyperparam_comparison,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MEDOPT — Clinical Screening System",
    page_icon="🏥", layout="wide"
)

st.markdown("""
<style>
  .block-container { padding-top: 1.4rem; }
  .stMetric label  { font-size: 0.78rem; }
  div[data-testid="metric-container"] { background:#f0f2f6;
      border-radius:8px; padding:10px 14px; }
</style>""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.title("MEDOPT")
    st.caption("Clinical Screening System")
    st.divider()

    page = st.radio("📌 Navigate", [
        "🏠 Overview",
        "🔬 Problem 1 · Feature Selection",
        "🎯 Problem 2 · Threshold Optimisation",
        "⚙️  Problem 4 · Hyperparameter Tuning",
        "🩺 Patient Diagnosis",
        "📊 Cross-Disease Comparison",
    ])

    st.divider()
    disease = st.selectbox(
        "🦠 Disease",
        list(DISEASE_CONFIGS.keys()),
        format_func=lambda d: f"{DISEASE_CONFIGS[d]['emoji']}  {d}",
    )
    algorithm = st.selectbox("🤖 Algorithm (Problem 1)", ["GWO", "PSO", "WOA"])
    fast_mode = st.checkbox("⚡ Fast Mode (fewer iterations)", value=True)
    st.caption("Fast mode: 5 agents × 10 iters. Uncheck for full 10×20.")

# ─── Session-state cache ──────────────────────────────────────────────────────
for key in ("p1", "p2", "p4"):
    if key not in st.session_state:
        st.session_state[key] = {}

cache_key = f"{disease}_{algorithm}"


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏥 MEDOPT — Clinical Intelligent Screening System")
    st.markdown(
        "**Meta-Heuristic Optimisation (MHO) solves 3 real clinical problems "
        "across 3 disease domains using GWO, PSO & WOA.**"
    )
    st.divider()

    # ── Dataset status cards (compact) ────────────────────────────────────
    cols = st.columns(3)
    for col, (dname, dcfg) in zip(cols, DISEASE_CONFIGS.items()):
        file_ok = os.path.isfile(dcfg["file"])
        with col:
            with st.container():
                badge = "🟢" if file_ok else "🔴"
                st.markdown(f"#### {dcfg['emoji']} {dname}")
                st.caption(dcfg["description"])
                if file_ok:
                    X_df, y, stats = load_data(dcfg["file"], cfg := dcfg)
                    st.markdown(
                        f"{badge} **Dataset ready** &nbsp;·&nbsp; "
                        f"`{stats['total_samples']}` samples &nbsp;·&nbsp; "
                        f"`{stats['total_features']}` features",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"{badge} *Dataset missing*")

    st.divider()
    st.subheader("🔍 What Each Problem Solves")
    st.markdown("""
| Problem | MHO Method | Clinical Impact |
|---------|------------|-----------------|
| **P1 · Feature Selection** | GWO / PSO / WOA | Reduces tests needed → faster screening |
| **P2 · Threshold Optimisation** | PSO | Maximises sensitivity → fewer missed diagnoses |
| **P4 · Hyperparameter Tuning** | PSO | Auto-tunes Random Forest → higher AUC |
""")

    st.divider()
    st.info("💡 Select a disease and algorithm in the sidebar, then navigate to each Problem page.")

    # Optional: heatmap for selected disease
    cfg = DISEASE_CONFIGS[disease]
    if os.path.isfile(cfg["file"]):
        with st.expander(f"🔥 {disease} — Feature Correlation Heatmap"):
            X_df, _, _ = load_data(cfg["file"], cfg)
            st.pyplot(plot_heatmap(X_df))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Problem 1: Feature Selection
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Problem 1 · Feature Selection":
    st.title(f"🔬 Problem 1 — Feature Selection")
    st.markdown(f"**Disease:** {DISEASE_CONFIGS[disease]['emoji']} {disease} &nbsp;|&nbsp; **Algorithm:** {algorithm}")

    cfg = DISEASE_CONFIGS[disease]
    if not os.path.isfile(cfg["file"]):
        st.error(f"Dataset not found: `{cfg['file']}`  \nPlease place the CSV in the `data/` folder.")
        st.stop()

    if st.button(f"🚀 Run {algorithm} Feature Selection", use_container_width=True):
        with st.spinner(f"Running {algorithm} on {disease}… please wait."):
            result = run_problem1(disease, algorithm, fast_mode)
        st.session_state["p1"][cache_key] = result
        st.success("✅ Done!")

    p1 = st.session_state["p1"].get(cache_key)
    if p1 is None:
        st.info("Press the button above to run."); st.stop()

    # ── Summary metrics ────────────────────────────────────────────────────
    st.subheader("📊 Results")
    sel_idx   = p1["selected_indices"]
    feat_names = p1["feature_names"]
    n_total    = len(feat_names)
    n_selected = len(sel_idx)
    sel_names  = [feat_names[i] for i in sel_idx]
    red_pct    = round((n_total - n_selected) / n_total * 100, 1) if n_total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Accuracy",  f"{p1['baseline_acc']:.4f}")
    c2.metric("Optimised Accuracy", f"{p1['optimized_acc']:.4f}",
              delta=f"{p1['optimized_acc'] - p1['baseline_acc']:+.4f}")
    c3.metric("Tests Required",
              f"{n_selected} / {n_total}",
              delta=f"–{n_total - n_selected} tests", delta_color="inverse")
    c4.metric("Test Reduction", f"{red_pct}%")

    st.markdown(f"**✅ Minimum Required Test Panel:**  \n`{'  •  '.join(sel_names)}`")

    # ── Tabs ──────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(
        ["📉 Convergence", "🔲 Confusion Matrix", "📋 Report", "🔥 Heatmap"]
    )
    with t1:
        st.pyplot(plot_convergence({algorithm: p1["fitness_curve"]}))
    with t2:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Baseline (all features)**")
            st.pyplot(plot_confusion_matrix(
                p1["baseline_cm"], p1["class_names"], "Baseline"))
        with cb:
            st.markdown(f"**{algorithm} Optimised**")
            st.pyplot(plot_confusion_matrix(
                p1["confusion_matrix"], p1["class_names"], f"{algorithm} Optimised"))
    with t3:
        st.code(p1["baseline_report"],          language="text")
        st.code(p1["classification_report"],    language="text")
    with t4:
        st.pyplot(plot_heatmap(p1["X_df"]))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Problem 2: Threshold Optimisation
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Problem 2 · Threshold Optimisation":
    st.title("🎯 Problem 2 — Clinical Threshold Optimisation (PSO)")
    st.markdown(
        "PSO finds the decision threshold that **maximises Sensitivity** "
        "(fewer missed diagnoses) while maintaining Specificity."
    )

    p1 = st.session_state["p1"].get(cache_key)
    if p1 is None:
        st.warning("⚠️ Run Problem 1 first for this disease/algorithm."); st.stop()

    alpha = st.slider("Sensitivity weight (α)", 0.5, 1.0, 0.75, 0.05,
                      help="α=1 = maximise Sensitivity | α=0.5 = balanced")

    if st.button("🚀 Run PSO Threshold Optimisation", use_container_width=True):
        with st.spinner("PSO searching for optimal threshold…"):
            result = run_problem2(p1, alpha=alpha)
        st.session_state["p2"][cache_key] = result
        st.success("✅ Done!")

    p2 = st.session_state["p2"].get(cache_key)
    if p2 is None:
        st.info("Press the button above to run."); st.stop()

    # ── Impact metrics ─────────────────────────────────────────────────────
    st.subheader("📊 Clinical Impact")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Default Threshold",  "0.50")
    c2.metric("PSO Optimal Threshold", f"{p2['optimal_threshold']:.3f}")
    c3.metric("Sensitivity (Default → Optimal)",
              f"{p2['opt_sensitivity']:.3f}",
              delta=f"{p2['opt_sensitivity'] - p2['default_sensitivity']:+.3f}")
    c4.metric("Missed Patients (False Negatives)",
              f"{p2['opt_fn']}",
              delta=f"{p2['opt_fn'] - p2['default_fn']:+d}", delta_color="inverse")

    # ── Tabs ──────────────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["📈 ROC Curve", "📉 Threshold Analysis", "🔲 Confusion Matrices"])
    with t1:
        st.pyplot(plot_roc_curve(
            p2["y_test"], p2["y_prob"],
            optimal_threshold=p2["optimal_threshold"]
        ))
    with t2:
        st.pyplot(plot_threshold_analysis(
            p2["y_test"], p2["y_prob"], p2["optimal_threshold"]
        ))
    with t3:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Default Threshold (0.50)**")
            st.pyplot(plot_confusion_matrix(
                p2["cm_default"], p1["class_names"], "Default Threshold"))
        with cb:
            st.markdown(f"**PSO Optimal ({p2['optimal_threshold']:.3f})**")
            st.pyplot(plot_confusion_matrix(
                p2["cm_optimal"], p1["class_names"], "PSO Optimal Threshold"))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Problem 4: Hyperparameter Tuning
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Problem 4 · Hyperparameter Tuning":
    st.title("⚙️ Problem 4 — Hyperparameter Tuning (PSO)")
    st.markdown(
        "PSO auto-tunes RandomForest hyperparameters on the **selected features** "
        "from Problem 1 using 5-fold cross-validation as the fitness function."
    )

    p1 = st.session_state["p1"].get(cache_key)
    if p1 is None:
        st.warning("⚠️ Run Problem 1 first for this disease/algorithm."); st.stop()

    if st.button("🚀 Run PSO Hyperparameter Tuning", use_container_width=True):
        with st.spinner("PSO searching hyperparameter space…"):
            result = run_problem4(p1, fast_mode=fast_mode)
        st.session_state["p4"][cache_key] = result
        st.success("✅ Done!")

    p4 = st.session_state["p4"].get(cache_key)
    if p4 is None:
        st.info("Press the button above to run."); st.stop()

    # ── Metrics ────────────────────────────────────────────────────────────
    st.subheader("📊 Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Default RF Accuracy",  f"{p4['default_acc']:.4f}")
    c2.metric("PSO-Tuned Accuracy",   f"{p4['tuned_acc']:.4f}",
              delta=f"{p4['tuned_acc'] - p4['default_acc']:+.4f}")
    c3.metric("Best CV Score", f"{p4['best_cv_score']:.4f}")

    st.subheader("🔧 Tuned Hyperparameters")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Default RF**")
        st.json(p4["default_params"])
    with col_b:
        st.markdown("**PSO-Tuned RF**")
        st.json(p4["best_params"])

    t1, t2, t3 = st.tabs(["📉 Tuning Convergence", "📊 Parameter Chart", "🔲 Confusion Matrix"])
    with t1:
        st.pyplot(plot_convergence({"PSO Hyperopt": p4["hp_curve"]}))
    with t2:
        default_vis = {k: v if v is not None else 0 for k, v in p4["default_params"].items()}
        st.pyplot(plot_hyperparam_comparison(
            default_vis, p4["best_params"],
            p4["default_acc"], p4["tuned_acc"]
        ))
    with t3:
        st.pyplot(plot_confusion_matrix(
            p4["tuned_cm"], p1["class_names"], "PSO-Tuned RF"))
        st.code(p4["tuned_report"], language="text")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Patient Diagnosis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Patient Diagnosis":
    st.title("🩺 Patient Diagnosis — Clinical Decision Support")
    st.markdown(
        "Enter a patient's test values to receive **risk score, diagnosis, "
        "confidence, and an AI explanation** of which features drove the prediction."
    )

    p1 = st.session_state["p1"].get(cache_key)
    p4 = st.session_state["p4"].get(cache_key)
    p2 = st.session_state["p2"].get(cache_key)

    if p1 is None:
        st.warning("⚠️ Run Problem 1 first for this disease."); st.stop()

    cfg          = DISEASE_CONFIGS[disease]
    feature_names = p1["feature_names"]

    st.subheader("📋 Patient Data Entry")
    mode = st.radio("Input mode", ["Select from test set", "Manual entry"], horizontal=True)

    if mode == "Select from test set":
        idx = st.slider("Patient index (from test set)",
                        0, len(p1["X_test"]) - 1, 0)
        # X_test is scaled; we need unscaled → re-generate from X_df
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(p1["X_train"])
        patient_values = scaler.inverse_transform(p1["X_test"])[idx]
        true_label = int(np.array(p1["y_test"])[idx])
        st.caption(f"True label: **{cfg['class_names'][true_label]}**")
    else:
        X_df = p1["X_df"]
        patient_values = np.array([
            st.number_input(f"{f}", value=float(X_df[f].median()), key=f"inp_{f}")
            for f in feature_names
        ])

    threshold = p2["optimal_threshold"] if p2 else 0.5
    st.caption(f"Decision threshold: **{threshold:.3f}** "
               f"({'PSO-optimised' if p2 else 'default 0.50'})")

    if st.button("🔍 Diagnose Patient", use_container_width=True):
        with st.spinner("Analysing…"):
            result = diagnose_patient(patient_values, p1, p4, threshold)

        # ── Risk gauge ─────────────────────────────────────────────────────
        st.subheader("🎯 Clinical Output")
        st.pyplot(plot_risk_gauge(result["risk_score"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("Diagnosis",   result["diagnosis"])
        c2.metric("Risk Score",  f"{result['risk_score']} / 100")
        c3.metric("Confidence",  f"{result['confidence']:.1f}%")

        st.markdown(f"### {result['warning']}")
        st.divider()

        # ── SHAP / Importance Explanation ─────────────────────────────────
        st.divider()
        exp = result["explanation"]
        shap_title = ("SHAP Feature Explanation" if exp["available"]
                      else "Feature Importance (SHAP not installed — using RF importance)")
        st.subheader(f"🧠 {shap_title}")
        if not exp["available"]:
            st.caption("Install SHAP for better explanations: `pip install shap`")
        st.pyplot(plot_shap_bar(exp["values"], exp["feature_names"]))


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — Cross-Disease Comparison
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Cross-Disease Comparison":
    st.title("📊 Cross-Disease Results Comparison")
    st.markdown("Run **Problem 1** for each disease first, then view this summary.")

    rows = []
    for dname, dcfg in DISEASE_CONFIGS.items():
        for alg in ["GWO", "PSO", "WOA"]:
            k   = f"{dname}_{alg}"
            p1r = st.session_state["p1"].get(k)
            if p1r:
                _fn  = p1r["feature_names"]
                _si  = p1r["selected_indices"]
                _nt  = len(_fn)
                _ns  = len(_si)
                _rp  = round((_nt - _ns) / _nt * 100, 1) if _nt > 0 else 0.0
                rows.append({
                    "Disease":       f"{dcfg['emoji']} {dname}",
                    "Algorithm":     alg,
                    "Baseline Acc":  f"{p1r['baseline_acc']:.4f}",
                    "Optimised Acc": f"{p1r['optimized_acc']:.4f}",
                    "Δ Acc":         f"{p1r['optimized_acc'] - p1r['baseline_acc']:+.4f}",
                    "Tests →":       f"{_nt} → {_ns}",
                    "Test Red. %":   f"{_rp}%",
                })

    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Accuracy bar for current selection
        all_labels, all_vals = [], []
        for r in rows:
            all_labels.append(f"{r['Disease']} / {r['Algorithm']}")
            all_vals.append(float(r["Optimised Acc"]))
        st.pyplot(plot_accuracy_bar(all_labels, all_vals, figsize=(9, max(4, len(rows) * 0.5))))
    else:
        st.info("No results yet. Run Problem 1 for at least one disease + algorithm combination.")