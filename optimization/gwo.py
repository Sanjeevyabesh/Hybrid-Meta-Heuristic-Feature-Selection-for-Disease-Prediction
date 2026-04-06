import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class GWO:
    def __init__(self, n_wolves, n_features, max_iter):
        self.n_wolves = n_wolves
        self.n_features = n_features
        self.max_iter = max_iter
        self.fitness_curve = []

    def fitness(self, X_train, X_test, y_train, y_test, solution):
        selected = np.where(solution > 0.5)[0]

        if len(selected) == 0:
            return 0

        model = SVC()
        model.fit(X_train[:, selected], y_train)
        preds = model.predict(X_test[:, selected])

        return accuracy_score(y_test, preds)

    def optimize(self, X_train, X_test, y_train, y_test):
        wolves = np.random.rand(self.n_wolves, self.n_features)

        alpha = beta = delta = None
        alpha_score = beta_score = delta_score = -1

        for t in range(self.max_iter):
            for i in range(self.n_wolves):
                fitness = self.fitness(X_train, X_test, y_train, y_test, wolves[i])

                if fitness > alpha_score:
                    delta_score, delta = beta_score, beta
                    beta_score, beta = alpha_score, alpha
                    alpha_score, alpha = fitness, wolves[i]

                elif fitness > beta_score:
                    delta_score, delta = beta_score, beta
                    beta_score, beta = fitness, wolves[i]

                elif fitness > delta_score:
                    delta_score, delta = fitness, wolves[i]

            a = 2 - t * (2 / self.max_iter)

            for i in range(self.n_wolves):
                for j in range(self.n_features):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * beta[j] - wolves[i][j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * delta[j] - wolves[i][j])
                    X3 = delta[j] - A3 * D_delta

                    wolves[i][j] = (X1 + X2 + X3) / 3

            wolves = np.clip(wolves, 0, 1)
            self.fitness_curve.append(alpha_score)

            print(f"Iteration {t+1}, Best Fitness: {alpha_score}")

        return alpha, self.fitness_curve