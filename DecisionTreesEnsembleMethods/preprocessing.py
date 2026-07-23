"""
preprocessing.py
================
Handles preprocessing for Iris classification:
  - Select features and target variable
  - Convert class names to a list
  - Split data into training and testing sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def _get_species_column(df: pd.DataFrame) -> str:
	"""
	Find species target column with case-insensitive matching.

	Args:
		df (pd.DataFrame): Iris DataFrame.

	Returns:
		str: Target column name.
	"""
	for col in df.columns:
		if col.lower() == "species":
			return col
	raise ValueError("Species column not found. Expected 'species' or 'Species'.")


def select_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
	"""
	Split dataframe into independent features (X) and target (y).

	Args:
		df (pd.DataFrame): Iris DataFrame.

	Returns:
		tuple:
			- X (pd.DataFrame): Numeric feature matrix.
			- y (pd.Series): Species labels.
			- feature_names (list[str]): Feature column names.
	"""
	target_col = _get_species_column(df)
	X = df.drop(columns=[target_col])
	y = df[target_col]
	feature_names = X.columns.tolist()

	print("\n" + "=" * 60)
	print("FEATURES AND TARGET SELECTION")
	print("=" * 60)
	print(f"Features ({len(feature_names)}): {feature_names}")
	print(f"Target: {target_col}")
	print("=" * 60)

	return X, y, feature_names


def get_class_names(y: pd.Series) -> list[str]:
	"""
	Convert unique class labels to a sorted list.

	Args:
		y (pd.Series): Target labels.

	Returns:
		list[str]: Sorted class names.
	"""
	class_names = sorted(y.astype(str).unique().tolist())

	print("\nClass names list:", class_names)
	return class_names


def split_data(
	X: pd.DataFrame,
	y: pd.Series,
	test_size: float = 0.20,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""
	Split features and target into train/test sets.

	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target labels.
		test_size (float): Ratio for test set.
		random_state (int): Random seed.

	Returns:
		tuple: X_train, X_test, y_train, y_test
	"""
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=y,
	)

	print("\n" + "=" * 60)
	print("TRAIN / TEST SPLIT")
	print("=" * 60)
	print(f"Total samples : {len(X)}")
	print(f"Train samples : {len(X_train)}")
	print(f"Test samples  : {len(X_test)}")
	print("=" * 60)

	return X_train, X_test, y_train, y_test


def prepare_data(
	df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
	"""
	Run preprocessing tasks end-to-end.

	Args:
		df (pd.DataFrame): Clean Iris DataFrame.

	Returns:
		tuple:
			- X_train
			- X_test
			- y_train
			- y_test
			- feature_names
			- class_names
	"""
	X, y, feature_names = select_features_target(df)
	class_names = get_class_names(y)
	X_train, X_test, y_train, y_test = split_data(X, y)
	return X_train, X_test, y_train, y_test, feature_names, class_names


if __name__ == "__main__":
	from data_loader import load_data
	from eda import run_eda

	iris_df = load_data(source="csv")
	clean_df = run_eda(iris_df)
	prepare_data(clean_df)
