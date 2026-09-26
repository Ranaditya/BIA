"""Evaluation helpers for segment-classification models."""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(y_true, y_pred):
	"""Evaluate segment predictions with standard classification metrics.

	Args:
		y_true: Actual segment labels.
		y_pred: Predicted segment labels.

	Returns:
		Dictionary containing accuracy, classification report, and confusion
		matrix values.
	"""
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"classification_report": classification_report(
			y_true,
			y_pred,
			output_dict=True,
			zero_division=0,
		),
		"confusion_matrix": confusion_matrix(y_true, y_pred),
	}


def get_feature_importance(model, feature_names):
	"""Return model feature importances sorted from highest to lowest.

	Args:
		model: Fitted tree-based model exposing ``feature_importances_``.
		feature_names: Names corresponding to the model input columns.

	Returns:
		DataFrame with ``feature`` and ``importance`` columns, sorted descending.

	Raises:
		AttributeError: If the model has no feature importances.
		ValueError: If the number of names does not match the importances.
	"""
	if not hasattr(model, "feature_importances_"):
		raise AttributeError("model must expose feature_importances_")

	importance = model.feature_importances_
	if len(feature_names) != len(importance):
		raise ValueError("feature_names must match the number of model features")

	return (
		pd.DataFrame({"feature": list(feature_names), "importance": importance})
		.sort_values("importance", ascending=False)
		.reset_index(drop=True)
	)
