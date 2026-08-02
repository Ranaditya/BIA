"""Train and evaluate linear SVM and KNN classifiers."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


IMAGE_DIR = Path(__file__).parent / "images"


def evaluate_model(name, model, X_test, y_test, class_names):
    """Predict, print classification metrics, and save a confusion matrix."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print(f"\n{name} accuracy: {accuracy:.4f}")
    print(f"\n{name} classification report:")
    print(classification_report(y_test, predictions, target_names=class_names, zero_division=0))
    print(f"{name} confusion matrix:\n{matrix}")

    IMAGE_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"{name.lower()}_confusion_matrix.png", dpi=150)
    plt.show()
    plt.close()
    return predictions, accuracy


def run_models(X_train, X_test, y_train, y_test, class_names, k: int = 3):
    """Train and evaluate a linear SVM and k-nearest-neighbors model."""
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train, y_train)
    _, svm_accuracy = evaluate_model("SVM", svm_model, X_test, y_test, class_names)

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    _, knn_accuracy = evaluate_model("KNN", knn_model, X_test, y_test, class_names)

    return svm_model, knn_model, {"SVM": svm_accuracy, "KNN": knn_accuracy}
