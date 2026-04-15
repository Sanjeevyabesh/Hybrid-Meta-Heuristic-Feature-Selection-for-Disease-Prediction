"""
optimization/pso.py
--------------------
Particle Swarm Optimiser (PSO) for binary feature selection.

Fixes applied
─────────────
✔ fitness_curve reset at the start of every optimize() call  (was accumulating)
✔ np.random.default_rng(seed) used for full reproducibility  (was unseeded)
✔ particles[i].copy() stored for pbest / gbest               (was storing references)
✔ r1 / r2 drawn per-feature (vectorised) instead of scalars  (more standard PSO)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class PSO:
    def __init__(self, n_particles: int, n_features: int, max_iter: int, random_state: int = 42):
        """
        Parameters
        ----------
        n_particles  : swarm size
        n_features   : dimensionality of the feature space
        max_iter     : maximum optimisation iterations
        random_state : seed for NumPy RNG (reproducibility)
        """
        self.n_particles  = n_particles
        self.n_features   = n_features
        self.max_iter     = max_iter
        self.random_state = random_state
        self.fitness_curve: list = []

    # ── Fitness function ──────────────────────────────────────────────────────
    def _fitness(self, X_train, X_test, y_train, y_test, particle: np.ndarray) -> float:
        """Evaluate a binary feature mask using SVC accuracy."""
        selected = np.where(particle > 0.5)[0]
        if len(selected) == 0:
            return 0.0

        model = SVC(kernel="rbf")
        model.fit(X_train[:, selected], y_train)
        preds = model.predict(X_test[:, selected])
        return accuracy_score(y_test, preds)

    # ── Main optimisation loop ────────────────────────────────────────────────
    def optimize(self, X_train, X_test, y_train, y_test):
        """
        Run PSO and return (best_solution, fitness_curve).

        Returns
        -------
        gbest         : np.ndarray – best binary feature mask found
        fitness_curve : list[float] – best global fitness per iteration
        """
        # ✅ Reset curve to avoid accumulation across re-runs
        self.fitness_curve = []

        # ✅ Seeded RNG for reproducibility
        rng = np.random.default_rng(self.random_state)

        # Initialise particles and velocities
        particles = rng.random((self.n_particles, self.n_features))
        velocity  = np.zeros_like(particles)

        # Personal bests
        pbest        = particles.copy()                     # ✅ copy
        pbest_scores = np.full(self.n_particles, -1.0)

        # Global best
        gbest       = particles[0].copy()                   # ✅ copy
        gbest_score = -1.0

        # PSO hyper-parameters
        w, c1, c2 = 0.5, 1.5, 1.5

        for t in range(self.max_iter):

            # ── Evaluate particles and update bests ───────────────────────────
            for i in range(self.n_particles):
                fit = self._fitness(X_train, X_test, y_train, y_test, particles[i])

                if fit > pbest_scores[i]:
                    pbest_scores[i] = fit
                    pbest[i]        = particles[i].copy()   # ✅ copy

                if fit > gbest_score:
                    gbest_score = fit
                    gbest       = particles[i].copy()       # ✅ copy

            # ── Velocity and position update (vectorised per feature) ─────────
            for i in range(self.n_particles):
                r1 = rng.random(self.n_features)   # ✅ per-feature random vectors
                r2 = rng.random(self.n_features)

                velocity[i] = (
                    w  * velocity[i]
                    + c1 * r1 * (pbest[i]   - particles[i])
                    + c2 * r2 * (gbest      - particles[i])
                )
                particles[i] += velocity[i]

            # Clip to valid range
            particles = np.clip(particles, 0.0, 1.0)

            self.fitness_curve.append(gbest_score)
            print(f"PSO  iter {t + 1:>2}/{self.max_iter}  best fitness = {gbest_score:.4f}")

        return gbest, self.fitness_curve