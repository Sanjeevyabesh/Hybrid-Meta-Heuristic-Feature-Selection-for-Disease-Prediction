"""
optimization/gwo.py
--------------------
Grey Wolf Optimiser (GWO) for binary feature selection.

Fixes applied
─────────────
✔ fitness_curve reset at the start of every optimize() call  (was accumulating)
✔ np.random.default_rng(seed) used for full reproducibility  (was unseeded)
✔ wolves[i].copy() stored for α/β/δ leaders                  (was storing references)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class GWO:
    def __init__(self, n_wolves: int, n_features: int, max_iter: int, random_state: int = 42):
        """
        Parameters
        ----------
        n_wolves     : number of wolves (population size)
        n_features   : dimensionality of the feature space
        max_iter     : maximum optimisation iterations
        random_state : seed for NumPy RNG (reproducibility)
        """
        self.n_wolves     = n_wolves
        self.n_features   = n_features
        self.max_iter     = max_iter
        self.random_state = random_state
        self.fitness_curve: list = []

    # ── Fitness function ──────────────────────────────────────────────────────
    def _fitness(self, X_train, X_test, y_train, y_test, solution: np.ndarray) -> float:
        """Evaluate a binary feature mask using SVC accuracy."""
        selected = np.where(solution > 0.5)[0]
        if len(selected) == 0:
            return 0.0

        model = SVC(kernel="rbf")
        model.fit(X_train[:, selected], y_train)
        preds = model.predict(X_test[:, selected])
        return accuracy_score(y_test, preds)

    # ── Main optimisation loop ────────────────────────────────────────────────
    def optimize(self, X_train, X_test, y_train, y_test):
        """
        Run GWO and return (best_solution, fitness_curve).

        Returns
        -------
        alpha          : np.ndarray – best binary feature mask found
        fitness_curve  : list[float] – best fitness per iteration
        """
        # ✅ Reset curve so re-running the same object doesn't accumulate values
        self.fitness_curve = []

        # ✅ Seeded RNG for reproducibility
        rng = np.random.default_rng(self.random_state)

        # Initialise wolves in [0, 1]^n_features
        wolves = rng.random((self.n_wolves, self.n_features))

        # Leader positions (initialise to zeros so position math is always valid)
        alpha = np.zeros(self.n_features)
        beta  = np.zeros(self.n_features)
        delta = np.zeros(self.n_features)
        alpha_score = beta_score = delta_score = -1.0

        for t in range(self.max_iter):

            # ── Evaluate each wolf and update leaders ─────────────────────────
            for i in range(self.n_wolves):
                fit = self._fitness(X_train, X_test, y_train, y_test, wolves[i])

                if fit > alpha_score:
                    delta_score, delta = beta_score,  beta.copy()
                    beta_score,  beta  = alpha_score, alpha.copy()
                    alpha_score, alpha = fit,          wolves[i].copy()  # ✅ copy
                elif fit > beta_score:
                    delta_score, delta = beta_score, beta.copy()
                    beta_score,  beta  = fit,        wolves[i].copy()
                elif fit > delta_score:
                    delta_score, delta = fit, wolves[i].copy()

            # ── Linearly decrease a from 2 → 0 ───────────────────────────────
            a = 2.0 - t * (2.0 / self.max_iter)

            # ── Position update step ──────────────────────────────────────────
            for i in range(self.n_wolves):
                for j in range(self.n_features):
                    r1, r2 = rng.random(), rng.random()
                    A1, C1  = 2 * a * r1 - a, 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                    X1      = alpha[j] - A1 * D_alpha

                    r1, r2 = rng.random(), rng.random()
                    A2, C2  = 2 * a * r1 - a, 2 * r2
                    D_beta  = abs(C2 * beta[j]  - wolves[i][j])
                    X2      = beta[j]  - A2 * D_beta

                    r1, r2 = rng.random(), rng.random()
                    A3, C3  = 2 * a * r1 - a, 2 * r2
                    D_delta = abs(C3 * delta[j] - wolves[i][j])
                    X3      = delta[j] - A3 * D_delta

                    wolves[i][j] = (X1 + X2 + X3) / 3.0

            # Clip to valid range
            wolves = np.clip(wolves, 0.0, 1.0)

            self.fitness_curve.append(alpha_score)
            print(f"GWO  iter {t + 1:>2}/{self.max_iter}  best fitness = {alpha_score:.4f}")

        return alpha, self.fitness_curve