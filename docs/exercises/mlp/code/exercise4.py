"""Exercise 4 — Depth Comparison.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import generate_exercise34_data, standardize
from mlp import MLP
from plotting import plot_loss_curve

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

LR = 0.3
EPOCHS = 3000
SEEDS = [42, 43, 44]

X_train, X_test, y_train, y_test = generate_exercise34_data(random_state=42)
Xtr_s, Xte_s = standardize(X_train, X_test)

architectures = {
    "Exercício 3 (1 camada oculta)": [4, 16, 3],
    "Exercício 4 (2 camadas ocultas)": [4, 16, 16, 3],
}

results = {name: {"final_loss": [], "test_acc": [], "losses_seed42": None, "n_params": None}
           for name in architectures}

for name, layer_sizes in architectures.items():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        net = MLP(layer_sizes, output_activation="softmax", rng=rng)
        losses = net.fit(Xtr_s, y_train, lr=LR, epochs=EPOCHS)
        acc = net.accuracy(Xte_s, y_test)
        results[name]["final_loss"].append(losses[-1])
        results[name]["test_acc"].append(acc)
        results[name]["n_params"] = net.n_params()
        if seed == 42:
            results[name]["losses_seed42"] = losses

print(f"Hiperparâmetros iguais aos do Exercício 3: eta = {LR}, épocas = {EPOCHS}")
print(f"Seeds usadas para a inicialização dos pesos: {SEEDS}\n")

for name, r in results.items():
    n_layers = len(architectures[name]) - 2  
    mean_loss = np.mean(r["final_loss"])
    mean_acc = np.mean(r["test_acc"])
    print(f"[{name}] camadas ocultas = {n_layers}, parâmetros = {r['n_params']}")
    print(f"  loss final (treino) por seed = {[f'{v:.4f}' for v in r['final_loss']]}")
    print(f"  acurácia de teste por seed   = {[f'{v:.4f}' for v in r['test_acc']]}")
    print(f"  média loss final = {mean_loss:.4f}, média acurácia = {mean_acc:.4f}\n")

acc3 = np.mean(results["Exercício 3 (1 camada oculta)"]["test_acc"])
acc4 = np.mean(results["Exercício 4 (2 camadas ocultas)"]["test_acc"])
print(f"Diferença de acurácia (Ex.4 - Ex.3), média sobre {len(SEEDS)} seeds: {acc4 - acc3:+.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
plot_loss_curve(ax, results["Exercício 3 (1 camada oculta)"]["losses_seed42"], label="Exercício 3 (1 oculta)")
plot_loss_curve(ax, results["Exercício 4 (2 camadas ocultas)"]["losses_seed42"], label="Exercício 4 (2 ocultas)")
ax.set_title("Figura 6 — Loss de treino por época: 1 vs. 2 camadas ocultas (seed 42)")
ax.legend()
fig.savefig(FIGURES_DIR / "fig06-loss-profundidade.svg", bbox_inches="tight")
plt.close(fig)
