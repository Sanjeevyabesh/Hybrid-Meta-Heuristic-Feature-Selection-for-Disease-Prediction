"""
problems/problem2_threshold_opt.py
------------------------------------
Problem 2: Clinical Decision Threshold Optimisation
PSO finds the classification threshold that maximises a
clinical fitness balancing Sensitivity and Specificity.

Default threshold (0.5) is NOT medically optimal —
missing a Parkinson's / Diabetes / Heart Disease patient
(False Negative) is far worse than a false alarm.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


class ThresholdOptimizer:
    """
    PSO operating on a 1-D search space: threshold ∈ [0.10, 0.90].

    Fitness = alpha * Sensitivity + (1 - alpha) * Specificity

    alpha controls the clinical priority:
      alpha → 1.0  ⟹ maximise catching sick patients (high Recall)
      alpha → 0.5  ⟹ balanced clinical trade-off
    """

    def __init__(self, n_particles: int = 20, max_iter: int = 30,
                 alpha: float = 0.75, random_state: int = 42):
        self.n_particles  = n_particles
        self.max_iter     = max_iter
        self.alpha        = alpha           # sensitivity weight
        self.random_state = random_state
        self.fitness_curve: list = []

    # ── Fitness ───────────────────────────────────────────────────────────
    def _fitness(self, threshold: float, y_prob, y_true) -> float:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            return 0.0
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return self.alpha * sensitivity + (1.0 - self.alpha) * specificity

    # ── PSO loop ──────────────────────────────────────────────────────────
    def optimize(self, y_prob, y_true):
        """
        Parameters
        ----------
        y_prob : np.ndarray  – predicted probabilities for class 1
        y_true : array-like  – ground-truth labels

        Returns
        -------
        optimal_threshold : float
        fitness_curve     : list[float]
        """
        self.fitness_curve = []
        rng = np.random.default_rng(self.random_state)

        particles = rng.uniform(0.10, 0.90, self.n_particles)
        velocity  = np.zeros(self.n_particles)

        pbest        = particles.copy()
        pbest_scores = np.array(
            [self._fitness(p, y_prob, y_true) for p in particles]
        )

        gbest_idx   = int(np.argmax(pbest_scores))
        gbest       = float(pbest[gbest_idx])
        gbest_score = float(pbest_scores[gbest_idx])

        w, c1, c2 = 0.5, 1.5, 1.5

        for t in range(self.max_iter):
            r1 = rng.random(self.n_particles)
            r2 = rng.random(self.n_particles)

            velocity  = (w * velocity
                         + c1 * r1 * (pbest - particles)
                         + c2 * r2 * (gbest - particles))
            particles = np.clip(particles + velocity, 0.10, 0.90)

            scores = np.array(
                [self._fitness(p, y_prob, y_true) for p in particles]
            )

            improved = scores > pbest_scores
            pbest[improved]        = particles[improved]
            pbest_scores[improved] = scores[improved]

            best_idx = int(np.argmax(pbest_scores))
            if pbest_scores[best_idx] > gbest_score:
                gbest_score = float(pbest_scores[best_idx])
                gbest       = float(pbest[best_idx])

            self.fitness_curve.append(gbest_score)
            print(f"ThreshOpt  iter {t+1:>2}/{self.max_iter}"
                  f"  threshold={gbest:.3f}  fitness={gbest_score:.4f}")

        return gbest, self.fitness_curve
