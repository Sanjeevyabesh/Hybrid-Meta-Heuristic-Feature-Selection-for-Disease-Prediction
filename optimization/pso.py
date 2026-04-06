import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class PSO:
    def __init__(self, n_particles, n_features, max_iter):
        self.n_particles = n_particles
        self.n_features = n_features
        self.max_iter = max_iter
        self.fitness_curve = []

    def fitness(self, X_train, X_test, y_train, y_test, particle):
        selected = np.where(particle > 0.5)[0]

        if len(selected) == 0:
            return 0

        model = SVC()
        model.fit(X_train[:, selected], y_train)
        preds = model.predict(X_test[:, selected])

        return accuracy_score(y_test, preds)

    def optimize(self, X_train, X_test, y_train, y_test):
        particles = np.random.rand(self.n_particles, self.n_features)
        velocity = np.zeros_like(particles)

        pbest = particles.copy()
        pbest_scores = np.zeros(self.n_particles)

        gbest = particles[0]
        gbest_score = -1

        for t in range(self.max_iter):
            for i in range(self.n_particles):
                fitness = self.fitness(X_train, X_test, y_train, y_test, particles[i])

                if fitness > pbest_scores[i]:
                    pbest_scores[i] = fitness
                    pbest[i] = particles[i]

                if fitness > gbest_score:
                    gbest_score = fitness
                    gbest = particles[i]

            w, c1, c2 = 0.5, 1.5, 1.5

            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                velocity[i] = (
                    w * velocity[i]
                    + c1 * r1 * (pbest[i] - particles[i])
                    + c2 * r2 * (gbest - particles[i])
                )

                particles[i] += velocity[i]

            particles = np.clip(particles, 0, 1)
            self.fitness_curve.append(gbest_score)

            print(f"Iteration {t+1}, Best Fitness: {gbest_score}")

        return gbest, self.fitness_curve