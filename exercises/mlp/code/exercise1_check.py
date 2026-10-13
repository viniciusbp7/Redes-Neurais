"""Exercise 1 — Backpropagation by Hand: verificação numérica.
"""

import numpy as np

x = np.array([0.5, -0.2])
y = 1.0

W1 = np.array([[0.3, -0.1], [0.2, 0.4]])  # W^(1), 2x2
b1 = np.array([0.1, -0.2])  # b^(1)
W2 = np.array([0.5, -0.3])  # W^(2)
b2 = 0.2  # b^(2)

eta = 0.3

# --- A: Forward pass ---------------------------------------------------------
z1 = x @ W1 + b1  
h1 = np.tanh(z1)  
u2 = h1 @ W2 + b2  
y_hat = np.tanh(u2)  
L = (y - y_hat) ** 2  

print("--- Forward ---")
print(f"z1 = {z1}")
print(f"h1 = {h1}")
print(f"u2 = {u2:.6f}")
print(f"y_hat = {y_hat:.6f}")
print(f"L = {L:.6f}")

# --- B: Backward pass ----------------------------------------------------------
dL_dyhat = -2 * (y - y_hat)
dyhat_du2 = 1 - y_hat**2
dL_du2 = dL_dyhat * dyhat_du2

dL_dW2 = dL_du2 * h1  # dL/dW^(2)
dL_db2 = dL_du2  # dL/db^(2)

dL_dh1 = dL_du2 * W2  # dL/dh^(1), vetor (2,)
dh1_dz1 = 1 - h1**2
dL_dz1 = dL_dh1 * dh1_dz1  # dL/dz^(1), vetor (2,)

dL_dW1 = np.outer(x, dL_dz1)  # dL/dW^(1), 2x2 (x_i * dL/dz1_j)
dL_db1 = dL_dz1  # dL/db^(1)

print("\n--- Backward ---")
print(f"dL/dy_hat = {dL_dyhat:.6f}")
print(f"dy_hat/du2 (1 - y_hat^2) = {dyhat_du2:.6f}")
print(f"dL/du2 = {dL_du2:.6f}")
print(f"dL/dW2 = {dL_dW2}")
print(f"dL/db2 = {dL_db2:.6f}")
print(f"dL/dh1 = {dL_dh1}")
print(f"dh1/dz1 (1 - h1^2) = {dh1_dz1}")
print(f"dL/dz1 = {dL_dz1}")
print(f"dL/dW1 =\n{dL_dW1}")
print(f"dL/db1 = {dL_db1}")

# --- C: Parameter update --------------------------------------------------------
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2

print("\n--- Update (eta = 0.3) ---")
print(f"W1_new =\n{W1_new}")
print(f"b1_new = {b1_new}")
print(f"W2_new = {W2_new}")
print(f"b2_new = {b2_new:.6f}")

z1_2 = x @ W1_new + b1_new
h1_2 = np.tanh(z1_2)
u2_2 = h1_2 @ W2_new + b2_new
y_hat_2 = np.tanh(u2_2)
L_2 = (y - y_hat_2) ** 2
print(f"\nSegundo forward: y_hat = {y_hat_2:.6f}, L = {L_2:.6f} (L original = {L:.6f})")
print(f"A loss {'caiu' if L_2 < L else 'subiu'}.")
