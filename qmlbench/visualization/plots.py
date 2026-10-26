
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel


def plot_decision_boundary(model, X, y, save_path):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    predictions = model.predict(grid)
    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, predictions, alpha=0.3, cmap="coolwarm")

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="black"
    )

    plt.title("Classical SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_kernel_matrix(X, save_path, gamma=0.5):
    K = rbf_kernel(X, gamma=gamma)

    plt.figure(figsize=(7, 6))

    plt.imshow(K, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Similarity")

    plt.title("Classical RBF Kernel Matrix")
    plt.xlabel("Training Sample Index")
    plt.ylabel("Training Sample Index")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
