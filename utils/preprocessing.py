import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)

    # ✅ Drop non-useful column
    if 'name' in df.columns:
        df = df.drop(columns=['name'])

    # ✅ Explicitly select target
    if 'status' in df.columns:
        y = df['status']
        X = df.drop(columns=['status'])
    else:
        # fallback (last column)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    return X, y


def preprocess(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )