from utils.preprocessing import load_data, preprocess
from optimization.gwo import GWO
from optimization.pso import PSO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def baseline_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def run_pipeline(data_path, method="GWO"):
    # Load & preprocess
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Baseline
    baseline_acc = baseline_model(X_train, X_test, y_train, y_test)

    # Choose optimizer
    if method == "GWO":
        optimizer = GWO(n_wolves=10, n_features=X.shape[1], max_iter=20)
    else:
        optimizer = PSO(n_particles=10, n_features=X.shape[1], max_iter=20)

    # Optimization
    best_solution, fitness_curve = optimizer.optimize(
        X_train, X_test, y_train, y_test
    )

    # Feature selection
    selected_features = np.where(best_solution > 0.5)[0]

    # Train optimized model
    model = RandomForestClassifier()
    model.fit(X_train[:, selected_features], y_train)
    preds = model.predict(X_test[:, selected_features])

    optimized_acc = accuracy_score(y_test, preds)

    return {
        "baseline": baseline_acc,
        "optimized": optimized_acc,
        "features": len(selected_features),
        "curve": fitness_curve,
    }

def run_both(data_path):
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess(X, y)

    baseline_acc = baseline_model(X_train, X_test, y_train, y_test)

    # GWO
    gwo = GWO(n_wolves=10, n_features=X.shape[1], max_iter=20)
    gwo_sol, gwo_curve = gwo.optimize(X_train, X_test, y_train, y_test)
    gwo_features = np.where(gwo_sol > 0.5)[0]

    model = RandomForestClassifier()
    model.fit(X_train[:, gwo_features], y_train)
    gwo_acc = accuracy_score(y_test, model.predict(X_test[:, gwo_features]))

    # PSO
    pso = PSO(n_particles=10, n_features=X.shape[1], max_iter=20)
    pso_sol, pso_curve = pso.optimize(X_train, X_test, y_train, y_test)
    pso_features = np.where(pso_sol > 0.5)[0]

    model.fit(X_train[:, pso_features], y_train)
    pso_acc = accuracy_score(y_test, model.predict(X_test[:, pso_features]))

    return {
        "baseline": baseline_acc,
        "gwo_acc": gwo_acc,
        "pso_acc": pso_acc,
        "gwo_curve": gwo_curve,
        "pso_curve": pso_curve,
        "gwo_features": len(gwo_features),
        "pso_features": len(pso_features),
    }