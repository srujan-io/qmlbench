
import json
from pathlib import Path
from qmlbench.visualization.plots import (
    plot_decision_boundary,
    plot_kernel_matrix
)

from qmlbench.datasets.synthetic import load_two_moons
from qmlbench.models.classical_svm import train_classical_svm
from qmlbench.evaluation.metrics import evaluate_classifier


def main():
    X_train, X_test, y_train, y_test = load_two_moons()

    model = train_classical_svm(
        X_train,
        y_train
    )

    metrics = evaluate_classifier(
        model,
        X_test,
        y_test
    )

    print("\nQMLBench — Experiment 001")
    print("Model: Classical RBF-SVM")
    print("Dataset: Two Moons\n")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "exp001_classical.json"

    with open(output_file, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nResults saved to {output_file}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    plot_decision_boundary(
        model,
        X_train,
        y_train,
        output_dir / "exp001_decision_boundary.png"
    )

    plot_kernel_matrix(
        X_train,
        output_dir / "exp001_kernel_matrix.png",
        gamma=model._gamma
    )

print("Visualizations saved successfully!")


if __name__ == "__main__":
    main()
