"""Exercise 2 — Overlapping data (Pocket Algorithm).

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import generate_exercise2_data
from perceptron import train_perceptron
from plotting import mark_misclassified, plot_boundary, plot_scatter

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# --- A: Generate the data ----------------------------------------------------
X, y = generate_exercise2_data(rng)

fig, ax = plt.subplots(figsize=(7, 6))
plot_scatter(ax, X, y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 4 — Dados fortemente sobrepostos")
ax.legend()
fig.savefig(FIGURES_DIR / "fig04-dados-sobrepostos.svg", bbox_inches="tight")
plt.close(fig)

# --- B: Train with the Exercise 1 perceptron, unchanged, + pocket tracking --
w0 = rng.normal(0, 0.01, size=2)
b0 = 0.0

run = train_perceptron(X, y, eta=0.01, w_init=w0, b_init=b0, max_epochs=100, track_pocket=True)

w_final, b_final = run["w"], run["b"]
acc_final = run["epoch_accuracies"][-1]
w_pocket, b_pocket = run["pocket_w"], run["pocket_b"]
acc_pocket = run["pocket_accuracies"][-1]
pocket_epoch = run["pocket_epoch"]

print(f"[final]  w = {w_final}, b = {b_final:.4f}, acurácia = {acc_final:.4f}")
print(f"[pocket] w = {w_pocket}, b = {b_pocket:.4f}, acurácia = {acc_pocket:.4f}")
print(f"[pocket] melhor valor atingido na época {pocket_epoch}")
print(f"épocas executadas = {run['n_epochs']} (sempre 100 — dados não são separáveis)")

fig, ax = plt.subplots(figsize=(7, 6))
plot_scatter(ax, X, y)
plot_boundary(ax, X, w_final, b_final, color="black", linestyle="--", label="fronteira final")
plot_boundary(ax, X, w_pocket, b_pocket, color="darkgreen", linestyle="-", label="fronteira pocket")
mark_misclassified(ax, X, y, w_pocket, b_pocket)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 5 — Fronteiras final e pocket")
ax.legend(fontsize=9)
fig.savefig(FIGURES_DIR / "fig05-fronteiras-final-pocket.svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, run["n_epochs"] + 1), run["epoch_accuracies"], label="pesos atuais (final)")
ax.plot(range(1, run["n_epochs"] + 1), run["pocket_accuracies"], label="melhor-até-agora (pocket)")
ax.set_xlabel("Época")
ax.set_ylabel("Acurácia")
ax.set_title("Figura 6 — Acurácia por época: final vs. pocket")
ax.set_ylim(0.0, 1.0)
ax.legend()
fig.savefig(FIGURES_DIR / "fig06-acuracia-epoca-final-pocket.svg", bbox_inches="tight")
plt.close(fig)
