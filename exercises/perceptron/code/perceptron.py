"""Perceptron implementado do zero — regra de atualização com rótulos {0, 1}.
"""

import numpy as np


def step(z):
    """Ativação do perceptron: 1 se z >= 0, senão 0."""
    return (z >= 0).astype(int)


def predict(X, w, b):
    """Rótulos previstos (0/1) para as entradas X, dados os pesos w e o bias b."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    """Fração de pontos de X classificados corretamente por (w, b)."""
    return np.mean(predict(X, w, b) == y)


def train_perceptron(X, y, eta, w_init, b_init, max_epochs=100, track_pocket=False):
    w = np.array(w_init, dtype=float).copy()
    b = float(b_init)

    epoch_accuracies = []
    updates_per_epoch = []
    pocket_w, pocket_b = w.copy(), b
    pocket_acc = accuracy(X, y, w, b)
    pocket_accuracies = []
    pocket_epoch = 0

    epoch = 0
    for epoch in range(1, max_epochs + 1):
        n_updates = 0
        for xi, yi in zip(X, y):
            y_hat = step(np.array([xi @ w + b]))[0]
            if y_hat != yi:
                w = w + eta * (yi - y_hat) * xi
                b = b + eta * (yi - y_hat)
                n_updates += 1
                if track_pocket:
                    acc = accuracy(X, y, w, b)
                    if acc > pocket_acc:
                        pocket_acc = acc
                        pocket_w, pocket_b = w.copy(), b
                        pocket_epoch = epoch

        epoch_acc = accuracy(X, y, w, b)
        epoch_accuracies.append(epoch_acc)
        updates_per_epoch.append(n_updates)
        if track_pocket:
            pocket_accuracies.append(pocket_acc)

        if n_updates == 0:
            break

    result = {
        "w": w,
        "b": b,
        "epoch_accuracies": np.array(epoch_accuracies),
        "updates_per_epoch": np.array(updates_per_epoch),
        "n_epochs": epoch,
    }
    if track_pocket:
        result.update(
            {
                "pocket_w": pocket_w,
                "pocket_b": pocket_b,
                "pocket_accuracies": np.array(pocket_accuracies),
                "pocket_epoch": pocket_epoch,
            }
        )
    return result
