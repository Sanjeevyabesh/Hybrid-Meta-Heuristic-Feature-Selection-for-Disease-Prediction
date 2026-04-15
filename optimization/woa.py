"""
optimization/woa.py
--------------------
Whale Optimization Algorithm (WOA) — Mirjalili & Lewis (2016).
Binary feature selection wrapper using SVC fitness.

Fixes vs naive implementations
────────────────────────────────
✔ Seeded RNG (np.random.default_rng)
✔ fitness_curve reset every optimize() call
✔ Defensive copy for best whale position
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class WOA:
    def __init__(self, n_whales: int, n_features: int, max_iter: int,
                 random_state: int = 42, b: float = 1.0):
        """
        Parameters
        ----------
        n_whales     : population size
        n_features   : search-space dimensionality
        max_iter     : number of iterations
        random_state : RNG seed
        b            : spiral constant (typically 1.0)
        """
        self.n_whales     = n_whales
        self.n_features   = n_features
        self.max_iter     = max_iter
        self.random_state = random_state
        self.b            = b
        self.fitness_curve: list = []

    def _fitness(self, X_train, X_test, y_train, y_test, solution: np.ndarray) -> float:
        selected = np.where(solution > 0.5)[0]
        if len(selected) == 0:
            return 0.0
        model = SVC(kernel="rbf")
        model.fit(X_train[:, selected], y_train)
        return accuracy_score(y_test, model.predict(X_test[:, selected]))

    def optimize(self, X_train, X_test, y_train, y_test):
        """
        Returns
        -------
        best_whale    : np.ndarray – best binary feature mask
        fitness_curve : list[float]
        """
        self.fitness_curve = []
        rng = np.random.default_rng(self.random_state)

        whales     = rng.random((self.n_whales, self.n_features))
        best       = np.zeros(self.n_features)
        best_score = -1.0

        for t in range(self.max_iter):
            # Update global best
            for i in range(self.n_whales):
                fit = self._fitness(X_train, X_test, y_train, y_test, whales[i])
                if fit > best_score:
                    best_score = fit
                    best = whales[i].copy()

            a = 2.0 - t * (2.0 / self.max_iter)   # linearly decreases 2 → 0

            for i in range(self.n_whales):
                p = rng.random()
                A = 2.0 * a * rng.random(self.n_features) - a
                C = 2.0 * rng.random(self.n_features)
                l = rng.uniform(-1.0, 1.0, self.n_features)

                if p < 0.5:
                    if np.linalg.norm(A) < 1.0:
                        # Shrinking encircling mechanism
                        D = np.abs(C * best - whales[i])
                        whales[i] = best - A * D
                    else:
                        # Exploration: move toward a random whale
                        rand_idx = rng.integers(0, self.n_whales)
                        X_rand   = whales[rand_idx]
                        D        = np.abs(C * X_rand - whales[i])
                        whales[i] = X_rand - A * D
                else:
                    # Spiral bubble-net update
                    D_prime    = np.abs(best - whales[i])
                    whales[i]  = (D_prime * np.exp(self.b * l)
                                  * np.cos(2.0 * np.pi * l) + best)

            whales = np.clip(whales, 0.0, 1.0)
            self.fitness_curve.append(best_score)
            print(f"WOA  iter {t + 1:>2}/{self.max_iter}  best = {best_score:.4f}")

        return best, self.fitness_curve
