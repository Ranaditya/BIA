"""
model.py
========
Builds and evaluates classification models for Iris species prediction:
  - DecisionTreeClassifier
  - RandomForestClassifier (optional)
  - Accuracy and classification reports
  - Decision tree visualization
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree


IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def train_decision_tree(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	random_state: int = 42,
) -> DecisionTreeClassifier:
	"""
	Train a Decision Tree classifier.

	Args:
		X_train (pd.DataFrame): Training features.
		y_train (pd.Series): Training labels.
		random_state (int): Random seed.

	Returns:
		DecisionTreeClassifier: Trained classifier.
	"""
	model = DecisionTreeClassifier(random_state=random_state)
	model.fit(X_train, y_train)

	print("\n" + "=" * 60)
	print("DECISION TREE MODEL TRAINED")
	print("=" * 60)
	print(f"Tree depth       : {model.get_depth()}")
	print(f"Number of leaves : {model.get_n_leaves()}")
	print("=" * 60)

	return model


def visualize_decision_tree(
	model: DecisionTreeClassifier,
	feature_names: list[str],
	class_names: list[str],
) -> None:
	"""
	Visualize and save the trained decision tree.

	Args:
		model (DecisionTreeClassifier): Trained tree model.
		feature_names (list[str]): Predictor names.
		class_names (list[str]): Class labels as a list.
	"""
	plt.figure(figsize=(18, 10))
	plot_tree(
		model,
		feature_names=feature_names,
		class_names=class_names,
		filled=True,
		rounded=True,
		impurity=True,
		fontsize=9,
	)
	plt.title("Decision Tree - Iris Classification", fontsize=14, fontweight="bold")
	plt.tight_layout()
	plt.savefig(IMAGES_DIR / "model_decision_tree.png", dpi=150, bbox_inches="tight")
	plt.show()
	plt.close()
	print("Decision Tree plot saved as: images/model_decision_tree.png")


def train_random_forest(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	random_state: int = 42,
) -> RandomForestClassifier:
	"""
	Train a Random Forest classifier.

	Args:
		X_train (pd.DataFrame): Training features.
		y_train (pd.Series): Training labels.
		random_state (int): Random seed.

	Returns:
		RandomForestClassifier: Trained classifier.
	"""
	model = RandomForestClassifier(n_estimators=200, random_state=random_state)
	model.fit(X_train, y_train)
	print("\nRandom Forest model trained.")
	return model


def evaluate_model(
	model_name: str,
	model,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	class_names: list[str],
) -> dict:
	"""
	Evaluate classifier using accuracy and per-class report.

	Args:
		model_name (str): Display name of the model.
		model: Trained classifier with predict().
		X_test (pd.DataFrame): Test features.
		y_test (pd.Series): Test labels.
		class_names (list[str]): Sorted class names for report formatting.

	Returns:
		dict: Model metrics and report.
	"""
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, labels=class_names, target_names=class_names)

	print("\n" + "=" * 60)
	print(f"{model_name.upper()} - EVALUATION")
	print("=" * 60)
	print(f"Accuracy: {accuracy:.4f}")
	print("\nClassification Report:")
	print(report)

	return {
		"model": model_name,
		"accuracy": accuracy,
		"classification_report": report,
	}


def run_models(
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	y_test: pd.Series,
	feature_names: list[str],
	class_names: list[str],
	use_random_forest: bool = True,
) -> dict:
	"""
	Train and evaluate Decision Tree and optional Random Forest.

	Args:
		X_train (pd.DataFrame): Training features.
		X_test (pd.DataFrame): Test features.
		y_train (pd.Series): Training labels.
		y_test (pd.Series): Test labels.
		feature_names (list[str]): Predictor names.
		class_names (list[str]): Class label names as list.
		use_random_forest (bool): Whether to train Random Forest too.

	Returns:
		dict: Results for all trained models.
	"""
	results = {}

	tree_model = train_decision_tree(X_train, y_train)
	visualize_decision_tree(tree_model, feature_names, class_names)
	results["decision_tree"] = evaluate_model(
		"Decision Tree", tree_model, X_test, y_test, class_names
	)

	if use_random_forest:
		rf_model = train_random_forest(X_train, y_train)
		results["random_forest"] = evaluate_model(
			"Random Forest", rf_model, X_test, y_test, class_names
		)

		print("=" * 60)
		print("MODEL COMPARISON (ACCURACY)")
		print("=" * 60)
		print(f"Decision Tree : {results['decision_tree']['accuracy']:.4f}")
		print(f"Random Forest : {results['random_forest']['accuracy']:.4f}")
		print("=" * 60)

	return results


if __name__ == "__main__":
	from data_loader import load_data
	from eda import run_eda
	from preprocessing import prepare_data

	iris_df = load_data(source="csv")
	clean_df = run_eda(iris_df)
	X_train, X_test, y_train, y_test, feature_names, class_names = prepare_data(clean_df)
	run_models(X_train, X_test, y_train, y_test, feature_names, class_names)
