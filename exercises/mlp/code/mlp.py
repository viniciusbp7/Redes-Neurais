"""MLP totalmente conectado, implementado do zero (forward, loss, backward, update).
"""

import numpy as np


def tanh(z):
    return np.tanh(z)


def tanh_grad_from_activation(a):
    """dz/d(pré-ativação), dado a = tanh(z): tanh'(z) = 1 - a^2."""
    return 1.0 - a**2


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True) 
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """MLP com `tanh` nas camadas ocultas e saída sigmoid+BCE ou softmax+CCE.
    """

    def __init__(self, layer_sizes, output_activation, rng):
        assert output_activation in ("sigmoid", "softmax")
        self.layer_sizes = list(layer_sizes)
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))
            b = np.zeros(n_out)
            self.weights.append(w)
            self.biases.append(b)

    def n_params(self):
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def forward(self, X):
        """Retorna a lista de ativações de cada camada (ativações[0] = X)."""
        activations = [X]
        a = X
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if i < n_layers - 1:
                a = tanh(z)
            else:
                a = sigmoid(z) if self.output_activation == "sigmoid" else softmax(z)
            activations.append(a)
        return activations

    def predict_proba(self, X):
        return self.forward(X)[-1]

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.output_activation == "sigmoid":
            return (probs.ravel() >= 0.5).astype(int)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def loss(self, X, y, eps=1e-12):
        probs = self.predict_proba(X)
        if self.output_activation == "sigmoid":
            p = probs.ravel()
            return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        n = X.shape[0]
        return float(-np.mean(np.log(probs[np.arange(n), y] + eps)))

    def backward(self, activations, y):
        """Backprop manual. Retorna (dWs, dbs), na mesma ordem de self.weights."""
        n_layers = len(self.weights)
        n = activations[0].shape[0]
        dWs = [None] * n_layers
        dbs = [None] * n_layers

        a_out = activations[-1]
        if self.output_activation == "sigmoid":
            y_col = y.reshape(-1, 1).astype(float)
            delta = (a_out - y_col) / n  
        else:
            y_onehot = np.zeros_like(a_out)
            y_onehot[np.arange(n), y] = 1.0
            delta = (a_out - y_onehot) / n 

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            dWs[i] = a_prev.T @ delta
            dbs[i] = delta.sum(axis=0)
            if i > 0:
                da_prev = delta @ self.weights[i].T
                delta = da_prev * tanh_grad_from_activation(activations[i])

        return dWs, dbs

    def step(self, X, y, lr):
        activations = self.forward(X)
        dWs, dbs = self.backward(activations, y)
        for i in range(len(self.weights)):
            self.weights[i] -= lr * dWs[i]
            self.biases[i] -= lr * dbs[i]

    def fit(self, X, y, lr, epochs):
        """Gradient descent em lote completo (full-batch). Retorna a loss por época."""
        losses = np.empty(epochs)
        for epoch in range(epochs):
            self.step(X, y, lr)
            losses[epoch] = self.loss(X, y)
        return losses
