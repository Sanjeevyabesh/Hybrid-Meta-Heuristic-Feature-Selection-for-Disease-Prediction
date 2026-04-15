"""
problems/problem4_hyperparam_opt.py
-------------------------------------
Problem 4: Classifier Hyperparameter Optimisation
PSO tunes RandomForest hyperparameters on the SELECTED features
(from Problem 1 output) using 5-fold cross-validation as fitness.

Search space (normalised to [0, 1] per dimension):
  • n_estimators      : 50  – 300
  • max_depth         : 3   – 25
  • min_samples_split : 2   – 20
  • min_samples_leaf  : 1   – 10
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class HyperparamOptimizer:
    """PSO-based hyperparameter tuner for RandomForestClassifier."""

    # (low, high) bounds for each hyperparameter
    PARAM_BOUNDS = {
        "n_estimators":      (50,  300),
        "max_depth":         (3,   25),
        "min_samples_split": (2,   20),
        "min_samples_leaf":  (1,   10),
    }

    def __init__(self, n_particles: int = 15, max_iter: int = 20,
                 cv_folds: int = 5, random_state: int = 42):
        self.n_particles  = n_particles
        self.max_iter     = max_iter
        self.cv_folds     = cv_folds
        self.random_state = random_state
        self.fitness_curve: list = []

    # ── Decode normalised particle → RF kwargs ────────────────────────────
    def _decode(self, particle: np.ndarray) -> dict:
        keys   = list(self.PARAM_BOUNDS.keys())
        bounds = list(self.PARAM_BOUNDS.values())
        return {
            keys[i]: int(round(lo + particle[i] * (hi - lo)))
            for i, (lo, hi) in enumerate(bounds)
        }

    # ── Fitness ───────────────────────────────────────────────────────────
    def _fitness(self, particle, X, y, selected) -> float:
        params = self._decode(particle)
        model  = RandomForestClassifier(random_state=self.random_state, **params)
        scores = cross_val_score(
            model, X[:, selected], y,
            cv=self.cv_folds, scoring="accuracy", n_jobs=-1
        )
        return float(scores.mean())

    # ── PSO loop ──────────────────────────────────────────────────────────
    def optimize(self, X_train, y_train, selected_features):
        """
        Parameters
        ----------
        X_train          : np.ndarray
        y_train          : array-like
        selected_features: array-like – column indices from Problem 1

        Returns
        -------
        best_params   : dict   – decoded RF hyperparameters
        best_cv_score : float  – best cross-validation accuracy
        fitness_curve : list[float]
        """
        self.fitness_curve = []
        rng     = np.random.default_rng(self.random_state)
        n_dim   = len(self.PARAM_BOUNDS)

        particles = rng.random((self.n_particles, n_dim))
        velocity  = np.zeros_like(particles)

        # Initialise personal bests
        pbest        = particles.copy()
        pbest_scores = np.array(
            [self._fitness(p, X_train, y_train, selected_features)
             for p in particles]
        )

        gbest_idx   = int(np.argmax(pbest_scores))
        gbest       = particles[gbest_idx].copy()
        gbest_score = float(pbest_scores[gbest_idx])

        w, c1, c2 = 0.5, 1.5, 1.5

        for t in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = rng.random(n_dim), rng.random(n_dim)
                velocity[i] = (w * velocity[i]
                               + c1 * r1 * (pbest[i]  - particles[i])
                               + c2 * r2 * (gbest     - particles[i]))
                particles[i] = np.clip(particles[i] + velocity[i], 0.0, 1.0)

                fit = self._fitness(
                    particles[i], X_train, y_train, selected_features
                )
                if fit > pbest_scores[i]:
                    pbest_scores[i] = fit
                    pbest[i]        = particles[i].copy()
                if fit > gbest_score:
                    gbest_score = fit
                    gbest       = particles[i].copy()

            self.fitness_curve.append(gbest_score)
            print(f"HyperOpt  iter {t+1:>2}/{self.max_iter}"
                  f"  CV acc={gbest_score:.4f}"
                  f"  params={self._decode(gbest)}")

        return self._decode(gbest), gbest_score, self.fitness_curve
