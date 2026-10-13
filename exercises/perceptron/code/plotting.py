"""Funções de plot compartilhadas pelos Exercícios 1 e 2."""

import numpy as np

from perceptron import predict


def plot_scatter(ax, X, y, class_names=("Classe 0", "Classe 1")):
    for k, name in enumerate(class_names):
        mask = y == k
        ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5, s=15, label=name)


def plot_boundary(ax, X, w, b, color="black", linestyle="-", label="fronteira"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    if abs(w[1]) > 1e-12:
        xs = np.linspace(x_min, x_max, 200)
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=2, label=label)
    else:
        x0 = -b / w[0] if abs(w[0]) > 1e-12 else 0.0
        ax.axvline(x0, color=color, linestyle=linestyle, linewidth=2, label=label)


def mark_misclassified(ax, X, y, w, b, color="red"):
    pred = predict(X, w, b)
    wrong = pred != y
    ax.scatter(
        X[wrong, 0],
        X[wrong, 1],
        facecolors="none",
        edgecolors=color,
        s=70,
        linewidths=1.3,
        label=f"erro ({wrong.sum()})",
    )
