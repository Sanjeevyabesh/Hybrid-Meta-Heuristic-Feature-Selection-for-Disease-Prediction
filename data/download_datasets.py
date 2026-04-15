"""
data/download_datasets.py
--------------------------
Run this ONCE to download the Diabetes and Heart Disease datasets
into the data/ folder.

Usage:
    python data/download_datasets.py
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download_diabetes():
    print("Downloading PIMA Indians Diabetes dataset...")
    ds = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    df = ds.frame.copy()
    # Rename target column
    df["Outcome"] = (df["class"] == "tested_positive").astype(int)
    df = df.drop(columns=["class"])
    path = os.path.join(DATA_DIR, "diabetes.csv")
    df.to_csv(path, index=False)
    print(f"  ✅ Saved {len(df)} rows → {path}")


def download_heart():
    print("Downloading Cleveland Heart Disease dataset...")
    ds = fetch_openml("heart-c", version=1, as_frame=True, parser="auto")
    df = ds.frame.copy()
    # Binarise: 0 = no disease, 1 = disease (original values 1-4)
    df["target"] = (df["num"].astype(str).str.strip() != "0").astype(int)
    df = df.drop(columns=["num"])
    # Convert object columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    # Drop any remaining non-numeric columns
    df = df.select_dtypes(include=["number"])
    path = os.path.join(DATA_DIR, "heart.csv")
    df.to_csv(path, index=False)
    print(f"  ✅ Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    download_diabetes()
    download_heart()
    print("\n✅ All datasets downloaded. You can now run:  streamlit run app.py")
