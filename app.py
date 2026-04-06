import streamlit as st
from main import run_pipeline, run_both
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="MHO Disease Prediction", layout="centered")

st.title("🧠 MHO Disease Prediction System")

# Mode selection
mode = st.radio("Select Mode", ["Single Algorithm", "GWO vs PSO Comparison"])

# Algorithm selection (only for single mode)
if mode == "Single Algorithm":
    method = st.selectbox("Choose Algorithm", ["GWO", "PSO"])

st.divider()

# Dataset selection
st.subheader("📂 Dataset Selection")
use_default = st.checkbox("Use Default Parkinson Dataset")
uploaded_file = st.file_uploader("Or Upload Dataset (CSV)", type=["csv"])

data_path = None

# Set dataset path
if use_default:
    data_path = "data/parkinsons.csv"
    st.success("Using default dataset")

elif uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    data_path = "data/temp.csv"

    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Custom dataset uploaded")

else:
    st.warning("Please select or upload a dataset")

st.divider()

# -------------------------------
# 🚀 SINGLE ALGORITHM MODE
# -------------------------------
if data_path and mode == "Single Algorithm":
    if st.button("🚀 Run Optimization"):

        with st.spinner("Running optimization..."):
            results = run_pipeline(data_path, method)

        st.subheader("📊 Results")

        col1, col2 = st.columns(2)
        col1.metric("Baseline Accuracy", f"{results['baseline']:.4f}")
        col2.metric("Optimized Accuracy", f"{results['optimized']:.4f}")

        st.metric("Selected Features", results["features"])

        # Fitness curve
        st.subheader("📈 Fitness Curve")

        fig, ax = plt.subplots()
        ax.plot(results["curve"])
        ax.set_title(f"{method} Convergence")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Accuracy")

        st.pyplot(fig)

        st.success("✅ Done")

# -------------------------------
# 🚀 COMPARISON MODE
# -------------------------------
if data_path and mode == "GWO vs PSO Comparison":
    if st.button("🚀 Run Full Comparison"):

        with st.spinner("Running GWO & PSO..."):
            results = run_both(data_path)

        st.subheader("📊 Accuracy Comparison")

        st.write(f"Baseline: {results['baseline']:.4f}")
        st.write(f"GWO Accuracy: {results['gwo_acc']:.4f}")
        st.write(f"PSO Accuracy: {results['pso_acc']:.4f}")

        st.subheader("📉 Feature Reduction")

        st.write(f"GWO Features: {results['gwo_features']}")
        st.write(f"PSO Features: {results['pso_features']}")

        # Convergence graph
        st.subheader("📈 Convergence Comparison")

        fig, ax = plt.subplots()
        ax.plot(results["gwo_curve"], label="GWO")
        ax.plot(results["pso_curve"], label="PSO")

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Accuracy")
        ax.set_title("GWO vs PSO Convergence")
        ax.legend()

        st.pyplot(fig)

        st.success("✅ Comparison Completed")

        # -------------------------------
        # 📄 REPORT GENERATION
        # -------------------------------
        def generate_report(results):
            return f"""
MHO PROJECT REPORT

Title:
Hybrid Meta-Heuristic Feature Selection for Disease Prediction

Results:
Baseline Accuracy: {results['baseline']:.4f}
GWO Accuracy: {results['gwo_acc']:.4f}
PSO Accuracy: {results['pso_acc']:.4f}

Feature Selection:
GWO Selected Features: {results['gwo_features']}
PSO Selected Features: {results['pso_features']}

Conclusion:
Meta-heuristic algorithms improved model performance by optimizing feature selection.
GWO and PSO enhanced accuracy and reduced feature space.
"""

        report_text = generate_report(results)

        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name="MHO_Report.txt",
            mime="text/plain"
        )