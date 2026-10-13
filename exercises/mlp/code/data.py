import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split


def generate_exercise2_data(random_state=42):
    """Classe 1 cerca a Classe 0 por lados opostos — nenhuma reta separa as duas."""
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
    X, c = make_blobs(
        n_samples=[500, 250, 250],
        centers=centers,
        cluster_std=1.2,
        random_state=random_state,
    )
    y = (c > 0).astype(int)  # c=0 (centro) -> Classe 0; c=1 ou 2 (opostos) -> Classe 1
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


def generate_exercise34_data(random_state=42):
    """Dataset multiclasse (3 classes) compartilhado pelos Exercícios 3 e 4."""
    X, y = make_classification(
        n_samples=1500,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


def standardize(X_train, X_test):
    """Padroniza (média 0, desvio 1) com estatísticas tiradas só do treino."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std
