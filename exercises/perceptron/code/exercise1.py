"""Exercise 1 — Separable data.

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import generate_exercise1_data
from perceptron import train_perceptron
from plotting import mark_misclassified, plot_boundary, plot_scatter

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# --- A: Generate the data ----------------------------------------------------
X, y = generate_exercise1_data(rng)

fig, ax = plt.subplots(figsize=(7, 6))
plot_scatter(ax, X, y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 1 — Dados linearmente separáveis")
ax.legend()
fig.savefig(FIGURES_DIR / "fig01-dados-separaveis.svg", bbox_inches="tight")
plt.close(fig)

# --- B/C: Train with eta = 0.01 ----------------------------------------------
w0 = rng.normal(0, 0.01, size=2)
b0 = 0.0

run_eta001 = train_perceptron(X, y, eta=0.01, w_init=w0, b_init=b0)
w1, b1 = run_eta001["w"], run_eta001["b"]
epochs1 = run_eta001["n_epochs"]
acc1 = run_eta001["epoch_accuracies"][-1]

print(f"[eta=0.01] w inicial = {w0}, b inicial = {b0}")
print(f"[eta=0.01] w final   = {w1}")
print(f"[eta=0.01] b final   = {b1:.4f}")
print(f"[eta=0.01] épocas até convergência = {epochs1}")
print(f"[eta=0.01] acurácia final = {acc1:.4f}")
print(f"[eta=0.01] atualizações por época = {list(run_eta001['updates_per_epoch'])}")

fig, ax = plt.subplots(figsize=(7, 6))
plot_scatter(ax, X, y)
plot_boundary(ax, X, w1, b1, label="fronteira final")
mark_misclassified(ax, X, y, w1, b1)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 2 — Fronteira de decisão (η = 0.01)")
ax.legend()
fig.savefig(FIGURES_DIR / "fig02-fronteira-eta001.svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, epochs1 + 1), run_eta001["epoch_accuracies"], marker="o", markersize=4)
ax.set_xlabel("Época")
ax.set_ylabel("Acurácia")
ax.set_title("Figura 3 — Acurácia por época (η = 0.01)")
ax.set_ylim(0.0, 1.05)
fig.savefig(FIGURES_DIR / "fig03-acuracia-epoca-eta001.svg", bbox_inches="tight")
plt.close(fig)

run_eta100 = train_perceptron(X, y, eta=1.0, w_init=w0, b_init=b0)
w2, b2 = run_eta100["w"], run_eta100["b"]
epochs2 = run_eta100["n_epochs"]
acc2 = run_eta100["epoch_accuracies"][-1]

print(f"\n[eta=1.0] w final = {w2}")
print(f"[eta=1.0] b final = {b2:.4f}")
print(f"[eta=1.0] épocas até convergência = {epochs2}")
print(f"[eta=1.0] acurácia final = {acc2:.4f}")

dir1 = w1 / np.linalg.norm(w1)
dir2 = w2 / np.linalg.norm(w2)
cos_sim = float(np.dot(dir1, dir2))
print(f"[eta=1.0] cos(ângulo) entre w/||w|| dos dois runs = {cos_sim:.6f}")

zero_eta001 = train_perceptron(X, y, eta=0.01, w_init=[0.0, 0.0], b_init=0.0)
zero_eta100 = train_perceptron(X, y, eta=1.0, w_init=[0.0, 0.0], b_init=0.0)

ratio_w = zero_eta100["w"] / zero_eta001["w"]
ratio_b = zero_eta100["b"] / zero_eta001["b"] if zero_eta001["b"] != 0 else float("nan")

print(f"\n[w0=0] épocas (eta=0.01) = {zero_eta001['n_epochs']}, "
      f"épocas (eta=1.0) = {zero_eta100['n_epochs']}")
print(f"[w0=0] w(eta=1.0) / w(eta=0.01) = {ratio_w}  (esperado: 100 = eta2/eta1)")
print(f"[w0=0] b(eta=1.0) / b(eta=0.01) = {ratio_b:.4f}  (esperado: 100)")
