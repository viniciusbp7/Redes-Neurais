"""Funções de plot compartilhadas pelos Exercícios 2, 3 e 4."""

import numpy as np


def plot_scatter(ax, X, y, class_names):
    for k, name in enumerate(class_names):
        mask = y == k
        ax.scatter(X[mask, 0], X[mask, 1], alpha=0.5, s=15, label=name)


def plot_loss_curve(ax, losses, label=None):
    ax.plot(np.arange(1, len(losses) + 1), losses, label=label)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (treino)")


def plot_decision_boundary(ax, net, X, y, class_names, standardize_fn=None, grid_n=300):
    """Avalia a rede numa grade 2D e sombreia as regiões previstas."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_n), np.linspace(y_min, y_max, grid_n)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    if standardize_fn is not None:
        grid = standardize_fn(grid)
    preds = net.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, preds, levels=np.arange(-0.5, len(class_names) + 0.5), alpha=0.25)
    plot_scatter(ax, X, y, class_names)
