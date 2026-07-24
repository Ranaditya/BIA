# Wholesale Customer Clustering

## Project Overview

This project implements an unsupervised learning workflow for the wholesale customer dataset using:
- K-Means clustering
- Agglomerative hierarchical clustering
- Elbow method for selecting the number of clusters
- Silhouette score for cluster quality evaluation

## Dataset

The dataset is stored in the `data/` folder and contains wholesale customer spending features such as:
- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

## Project Structure

```text
UnsupervisedLearning/
├── data/
├── images/
├── data_loader.py
├── eda.py
├── preprocessing.py
├── model.py
├── main.py
└── README.md
```

## How to Run

```bash
cd UnsupervisedLearning
python main.py
```

## What the Pipeline Does

1. Loads the wholesale customer dataset
2. Performs exploratory data analysis
3. Drops non-numeric identifier-like columns and standardizes the feature values
4. Uses the elbow method to choose K
5. Runs K-Means clustering
6. Runs hierarchical clustering
7. Adds cluster labels to the data and evaluates them using silhouette score

## Outputs

The script saves plots to the `images/` folder, including:
- `feature_correlation_heatmap.png`
- `elbow_method.png`
