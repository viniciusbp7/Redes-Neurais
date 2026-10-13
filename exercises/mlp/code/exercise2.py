"""Exercise 2 — Binary Classification.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from data import generate_exercise2_data, standardize
from mlp import MLP
from plotting import plot_decision_boundary, plot_loss_curve, plot_scatter

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ("Classe 0", "Classe 1")

# --- A: Generate the data -----------------------------------------------------
X_train, X_test, y_train, y_test = generate_exercise2_data(random_state=42)
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

fig, ax = plt.subplots(figsize=(7, 6))
plot_scatter(ax, X_all, y_all, CLASS_NAMES)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 1 — Classe 1 cerca a Classe 0 por lados opostos")
ax.legend()
fig.savefig(FIGURES_DIR / "fig01-dados.svg", bbox_inches="tight")
plt.close(fig)

Xtr_s, Xte_s = standardize(X_train, X_test)

# --- B: Linear baseline ---------------------------------------------------------
baseline = LogisticRegression()
baseline.fit(Xtr_s, y_train)
baseline_acc = baseline.score(Xte_s, y_test)
print(f"[baseline] acurácia de teste (regressão logística) = {baseline_acc:.4f}")

# --- C: MLP from scratch ---------------------------------------------------------
HIDDEN_UNITS = 8
LR = 0.5
EPOCHS = 2000

rng = np.random.default_rng(42)
net = MLP([2, HIDDEN_UNITS, 1], output_activation="sigmoid", rng=rng)
losses = net.fit(Xtr_s, y_train, lr=LR, epochs=EPOCHS)
mlp_acc = net.accuracy(Xte_s, y_test)

print(f"[MLP] arquitetura = [2, {HIDDEN_UNITS}, 1], ativação oculta = tanh, saída = sigmoid")
print(f"[MLP] eta = {LR}, épocas = {EPOCHS}")
print(f"[MLP] loss final (treino) = {losses[-1]:.4f}")
print(f"[MLP] acurácia de teste = {mlp_acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
plot_loss_curve(ax, losses)
ax.set_title("Figura 2 — Loss de treino por época (MLP, Exercício 2)")
fig.savefig(FIGURES_DIR / "fig02-loss-epoca.svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
plot_decision_boundary(
    ax, net, X_test, y_test, CLASS_NAMES,
    standardize_fn=lambda g: (g - X_train.mean(axis=0)) / X_train.std(axis=0),
)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 3 — Fronteira de decisão da MLP (pontos de teste)")
ax.legend()
fig.savefig(FIGURES_DIR / "fig03-fronteira.svg", bbox_inches="tight")
plt.close(fig)
