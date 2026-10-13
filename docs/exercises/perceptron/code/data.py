"""Geração dos datasets 2D dos Exercícios 1 e 2 (sempre com o mesmo `rng`)."""

import numpy as np


def generate_exercise1_data(rng):
    """Duas classes 2D linearmente separáveis, 1000 pontos cada (Exercise 1, item A)."""
    mean0, cov0 = [1.5, 1.5], [[0.5, 0.0], [0.0, 0.5]]
    mean1, cov1 = [5.0, 5.0], [[0.5, 0.0], [0.0, 0.5]]
    class0 = rng.multivariate_normal(mean0, cov0, size=1000)
    class1 = rng.multivariate_normal(mean1, cov1, size=1000)
    X = np.vstack([class0, class1])
    y = np.array([0] * 1000 + [1] * 1000)
    return X, y


def generate_exercise2_data(rng):
    """Duas classes 2D fortemente sobrepostas, 1000 pontos cada (Exercise 2, item A)."""
    mean0, cov0 = [3.0, 3.0], [[1.5, 0.0], [0.0, 1.5]]
    mean1, cov1 = [4.0, 4.0], [[1.5, 0.0], [0.0, 1.5]]
    class0 = rng.multivariate_normal(mean0, cov0, size=1000)
    class1 = rng.multivariate_normal(mean1, cov1, size=1000)
    X = np.vstack([class0, class1])
    y = np.array([0] * 1000 + [1] * 1000)
    return X, y
