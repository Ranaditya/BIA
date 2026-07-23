"""
eda.py
======
Performs Exploratory Data Analysis (EDA) for the Iris dataset:
  - Pairplot visualization for feature relationships
  - Boxplots by species for feature distributions
  - Outlier handling using the IQR method
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def _get_species_column(df: pd.DataFrame) -> str:
	"""
	Find the target column for species with case-insensitive matching.

	Args:
		df (pd.DataFrame): Iris DataFrame.

	Returns:
		str: Name of the species column in the DataFrame.

	Raises:
		ValueError: If no species column is found.
	"""
	for col in df.columns:
		if col.lower() == "species":
			return col
	raise ValueError("Species column not found. Expected a column named 'species' or 'Species'.")


def plot_pairplot(df: pd.DataFrame) -> None:
	"""
	Create and save a pairplot to visualize relationships across features.

	Args:
		df (pd.DataFrame): Iris DataFrame.
	"""
	species_col = _get_species_column(df)

	pair = sns.pairplot(
		data=df,
		hue=species_col,
		diag_kind="hist",
		corner=False,
		plot_kws={"alpha": 0.8, "s": 35},
	)
	pair.fig.suptitle("Iris Pairplot by Species", y=1.02, fontsize=14, fontweight="bold")
	pair.savefig(IMAGES_DIR / "eda_pairplot.png", dpi=150, bbox_inches="tight")
	plt.show()
	plt.close(pair.fig)
	print("Pairplot saved as: images/eda_pairplot.png")


def plot_boxplots_by_species(df: pd.DataFrame) -> None:
	"""
	Create and save boxplots for each numeric feature grouped by species.

	Args:
		df (pd.DataFrame): Iris DataFrame.
	"""
	species_col = _get_species_column(df)
	feature_cols = [c for c in df.select_dtypes(include="number").columns if c != species_col]

	fig, axes = plt.subplots(2, 2, figsize=(14, 9))
	axes = axes.flatten()

	for i, col in enumerate(feature_cols):
		sns.boxplot(
			data=df,
			x=species_col,
			y=col,
			hue=species_col,
			legend=False,
			ax=axes[i],
			palette="Set2",
		)
		axes[i].set_title(f"{col} by {species_col}")
		axes[i].set_xlabel("Species")
		axes[i].set_ylabel(col)

	plt.suptitle("Feature Distributions by Species (Boxplots)", fontsize=14, fontweight="bold", y=1.01)
	plt.tight_layout()
	plt.savefig(IMAGES_DIR / "eda_boxplots_by_species.png", dpi=150, bbox_inches="tight")
	plt.show()
	plt.close(fig)
	print("Boxplots saved as: images/eda_boxplots_by_species.png")


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Remove outliers using the IQR method feature-by-feature.

	Applies filtering iteratively on each numeric feature, where values outside
	[Q1 - 1.5*IQR, Q3 + 1.5*IQR] are removed.

	Args:
		df (pd.DataFrame): Iris DataFrame.

	Returns:
		pd.DataFrame: Cleaned DataFrame after outlier removal.
	"""
	clean_df = df.copy()
	species_col = _get_species_column(clean_df)
	feature_cols = [c for c in clean_df.select_dtypes(include="number").columns if c != species_col]

	print("\n" + "=" * 60)
	print("OUTLIER HANDLING - IQR METHOD")
	print("=" * 60)

	for col in feature_cols:
		q1 = clean_df[col].quantile(0.25)
		q3 = clean_df[col].quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr

		before = len(clean_df)
		clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
		removed = before - len(clean_df)

		print(
			f"{col:15s} | Bounds: [{lower:.3f}, {upper:.3f}] | "
			f"Removed: {removed:2d} | Remaining: {len(clean_df)}"
		)

	total_removed = len(df) - len(clean_df)
	print("=" * 60)
	print(f"Total rows removed: {total_removed}")
	print(f"Final dataset size : {len(clean_df)}")
	print("=" * 60)

	return clean_df


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Run all requested EDA tasks and return the outlier-cleaned dataset.

	Args:
		df (pd.DataFrame): Iris DataFrame.

	Returns:
		pd.DataFrame: Cleaned DataFrame after IQR outlier handling.
	"""
	print("\nGenerating pairplot...")
	plot_pairplot(df)

	print("\nGenerating species-wise boxplots...")
	plot_boxplots_by_species(df)

	clean_df = remove_outliers_iqr(df)
	return clean_df


if __name__ == "__main__":
	from data_loader import load_data

	iris_df = load_data(source="csv")
	cleaned_df = run_eda(iris_df)
	print(f"\nCleaned dataset shape: {cleaned_df.shape}")
