"""Train classifiers that predict customer segment membership."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def train_segment_classifier(X, y, test_size=0.2, random_state=42):
	"""Train a Random Forest classifier to predict segment membership.

	The split is stratified because segment sizes are typically unequal.

	Args:
		X: RFM feature table or array-like feature matrix.
		y: Segment labels.
		test_size: Fraction of samples reserved for testing.
		random_state: Seed for reproducible splitting and model training.

	Returns:
		Tuple of fitted model, test features, test labels, and a metrics dict.
		The metrics dict contains accuracy, weighted precision, weighted recall,
		weighted F1, and the full per-class classification report.
	"""
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y,
	)

	model = RandomForestClassifier(random_state=random_state, n_estimators=200)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	metrics = {
		"accuracy": accuracy_score(y_test, y_pred),
		"precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
		"recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
		"f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
		"classification_report": classification_report(
			y_test,
			y_pred,
			output_dict=True,
			zero_division=0,
		),
	}
	return model, X_test, y_test, metrics
