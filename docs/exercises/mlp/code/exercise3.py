"""Exercise 3 — Multi-Class.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from data import generate_exercise34_data, standardize
from mlp import MLP
from plotting import plot_loss_curve

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ("Classe 0", "Classe 1", "Classe 2")

# --- A: Generate the data -----------------------------------------------------
X_train, X_test, y_train, y_test = generate_exercise34_data(random_state=42)
Xtr_s, Xte_s = standardize(X_train, X_test)

baseline = LogisticRegression(max_iter=1000)
baseline.fit(Xtr_s, y_train)
baseline_acc = baseline.score(Xte_s, y_test)
print(f"[baseline] acurácia de teste (regressão logística) = {baseline_acc:.4f}")

# --- B: Extend to 3 classes (softmax + categorical cross-entropy) -----------
HIDDEN_UNITS = 16
LR = 0.3
EPOCHS = 3000

rng = np.random.default_rng(42)
net = MLP([4, HIDDEN_UNITS, 3], output_activation="softmax", rng=rng)
losses = net.fit(Xtr_s, y_train, lr=LR, epochs=EPOCHS)
mlp_acc = net.accuracy(Xte_s, y_test)

print(f"[MLP] arquitetura = [4, {HIDDEN_UNITS}, 3], ativação oculta = tanh, saída = softmax")
print(f"[MLP] eta = {LR}, épocas = {EPOCHS}, parâmetros = {net.n_params()}")
print(f"[MLP] loss final (treino) = {losses[-1]:.4f}")
print(f"[MLP] acurácia de teste = {mlp_acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
plot_loss_curve(ax, losses)
ax.set_title("Figura 4 — Loss de treino por época (MLP multiclasse, Exercício 3)")
fig.savefig(FIGURES_DIR / "fig04-loss-epoca.svg", bbox_inches="tight")
plt.close(fig)

# --- C: Confusion matrix ------------------------------------------------------
y_pred = net.predict(Xte_s)
cm = confusion_matrix(y_test, y_pred)
print(f"[MLP] matriz de confusão (linhas = real, colunas = previsto):\n{cm}")

off_diag = cm.copy()
np.fill_diagonal(off_diag, 0)
i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
print(f"[MLP] maior cela fora da diagonal: real={i} previsto={j} ({off_diag[i, j]} casos)")

pair_totals = {}
for a in range(3):
    for b in range(a + 1, 3):
        pair_totals[(a, b)] = int(cm[a, b] + cm[b, a])
best_pair = max(pair_totals, key=pair_totals.get)
print(f"[MLP] totais por par (ambas as direções): {pair_totals}")
print(f"[MLP] par mais confundido (não ordenado): {best_pair}, {pair_totals[best_pair]} casos no total")

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Classe prevista")
ax.set_ylabel("Classe real")
ax.set_title("Figura 5 — Matriz de confusão (teste)")
for r in range(3):
    for c in range(3):
        ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                 color="white" if cm[r, c] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, label="nº de amostras")
fig.savefig(FIGURES_DIR / "fig05-matriz-confusao.svg", bbox_inches="tight")
plt.close(fig)
