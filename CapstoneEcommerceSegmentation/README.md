# E-commerce Customer Segmentation

This project analyzes transaction history to group customers by purchasing behavior and support more focused marketing, retention, and inventory decisions. It builds customer-level Recency, Frequency, and Monetary (RFM) features, compares clustering methods, profiles the resulting groups, and trains a classifier to reproduce KMeans segment labels from RFM features.

> **Important interpretation:** the classifier predicts labels created by clustering the same RFM feature set. Its accuracy measures how consistently a Random Forest can reproduce those labels; it does not establish that future purchases can be predicted. A separate time-based target and holdout period are needed for that claim.

## Project Contents

| File | Purpose |
| --- | --- |
| `data/data.csv` | Raw transaction input. |
| `data_loader.py` | Loads transactions, parses dates, and summarizes source data. |
| `preprocessing.py` | Removes incomplete, cancelled, invalid-price, non-numeric stock code, and duplicate rows; adds line-item revenue. |
| `features.py` | Aggregates customer-level RFM and log-scales/standardizes the features. |
| `eda.py` | Creates exploratory charts for spending, country, sales timing, RFM relationships, and products. |
| `clustering.py` | Evaluates KMeans values of k, fits KMeans/hierarchical/DBSCAN models, and profiles segments. |
| `classifier.py` | Trains a stratified Random Forest to predict cluster membership. |
| `evaluation.py` | Calculates classifier metrics and feature importance. |
| `visualization.py` | Creates clustering, segment-profile, and classifier-result charts. |
| `main.py` | Runs the end-to-end analysis and writes a run signature. |
| `outputs/figures/` | Generated EDA, clustering, profile, and classifier charts. |
| `outputs/run_signature.json` | Compact record of the latest run's key counts, versions, and metrics. |
| `outputs/analysis_results.json` | Cleaning-step counts, algorithm comparison, segment profiles for the selected k and k=4, cluster stability, classifier metrics, and feature importance. |
| `outputs/dashboard_data.json` | Data consumed by the interactive dashboard in `dashboards/`. |
| `reports/build_deck.py` | Builds `reports/Capstone_Ecommerce_Segmentation.pptx` from the BIA template and the `outputs/` JSON files (requires `python-pptx`). Run after `main.py`. |

## Data

The loader expects `data/data.csv` with the following columns:

`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

`InvoiceDate` is parsed as a datetime, `CustomerID` is loaded as a nullable string, and the CSV is read using Latin-1 encoding. Relative input paths are resolved from this project directory, so the commands below work from either the repository root or this project folder.

The row count (541,909) and columns match the public **Online Retail** dataset from the UCI Machine Learning Repository: transactions of a UK-based online gift retailer between 1 Dec 2010 and 9 Dec 2011 (Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33; licensed CC BY 4.0). Confirm that `data/data.csv` was obtained from that source before citing it.

## Setup and Run

Use a Python environment with the packages in the repository-level `requirements.txt` installed. From the repository root:

```powershell
python -m pip install -r requirements.txt
python CapstoneEcommerceSegmentation/main.py
```

If using the repository's existing Windows virtual environment:

```powershell
.venv-1\Scripts\python.exe CapstoneEcommerceSegmentation\main.py
```

The run prints data-quality counts, EDA interpretations, K-selection and algorithm-comparison tables, segment profiles, classifier metrics, feature importance, and the JSON run signature. Figures and the signature are written beneath `CapstoneEcommerceSegmentation/outputs/`.

## Workflow

1. **Inspect:** load raw records, parse dates, and report shape, missing values, customer/invoice counts, and country count.
2. **Clean:** drop missing CustomerIDs and cancellation invoices; retain prices of at least 0.01; keep numeric StockCodes as product lines; remove exact duplicate rows; calculate `TotalPrice = Quantity * UnitPrice`.
3. **Explore:** chart customer monetary value, country transaction counts, monthly sales, RFM correlations, and top products.
4. **Engineer RFM:** calculate recency in days from the latest transaction plus one day, frequency as unique invoices, and monetary value as summed line-item total.
5. **Scale:** apply `log1p` and then `StandardScaler` to Recency, Frequency, and Monetary.
6. **Cluster:** evaluate KMeans for k values 2 through 10 using inertia and silhouette score; compare KMeans, agglomerative hierarchical clustering, and DBSCAN.
7. **Profile:** calculate segment sizes, mean RFM values, and revenue share. Segment names combine a value tier (mean of the log-scaled Frequency and Monetary z-scores) with an activity status (inverse log-scaled Recency z-score), for example `High-Value Active` or `Low-Value Lapsing`. A k=4 profile is also saved as an alternative, finer-grained view.
8. **Check stability:** refit KMeans across 20 single-initialization seeds and 20 bootstrap subsamples (80%) and report the adjusted Rand index against the full-data labels.
9. **Evaluate classifier:** use a stratified 80/20 split to train a 200-tree Random Forest to reproduce KMeans labels; report accuracy, weighted precision/recall/F1, classification report, confusion matrix, and feature importance.

## Current Recorded Run

These values are from `outputs/run_signature.json` and will vary if the data, dependencies, preprocessing, or randomization changes.

| Measure | Recorded value |
| --- | ---: |
| scikit-learn version | 1.9.0 |
| Raw transaction rows | 541,909 |
| Clean transaction rows | 358,277 |
| Customers in RFM table | 4,314 |
| Snapshot date | 2011-12-10 12:50:00 |
| KMeans clusters | 2 |
| Cluster sizes, cluster IDs 0 and 1 | 1,665; 2,649 |
| Segment names, cluster IDs 0 and 1 | High-Value Active (85.0% of revenue); Low-Value Lapsing (15.0%) |
| KMeans silhouette | 0.4312 |
| KMeans Davies-Bouldin | 0.894 |
| Stability ARI, seeds / bootstrap (mean) | 0.994 / 0.989 |
| Random Forest cluster-label accuracy | 0.9942 |
| Highest classifier feature importance | Frequency |

The high classifier accuracy should be interpreted as segment-label reproducibility, not future-purchase performance. The model is trained on labels generated by KMeans from the same RFM variables. Feature importance is the Random Forest's impurity-based importance for reproducing those labels, not a causal explanation of customer behavior.

## Generated Figures

The main run generates these figures:

- `rfm_distributions.png`
- `country_breakdown.png`
- `sales_over_time.png`
- `rfm_correlations.png`
- `top_products.png`
- `k_selection.png`
- `customer_clusters.png`
- `segment_profiles.png`
- `confusion_matrix.png`

Every chart is saved to `outputs/figures/` and is also displayed when run with an interactive Matplotlib backend.

## Methodological Limitations and Next Steps

- The current workflow is a retrospective segmentation analysis, not a validated forecast of future transactions.
- KMeans labels are created before the classifier's train/test split. The reported classifier score therefore evaluates reproduction of an already-created partition, not an independent business outcome.
- Compare cluster quality, stability, usefulness to stakeholders, and segment sizes in addition to silhouette score. DBSCAN can identify noise as label `-1`; its silhouette may be unavailable if it finds fewer than two non-noise clusters.
- The segment names are heuristic summaries of relative RFM profiles. Validate them with business stakeholders before launching campaigns.
- For a future-purchase model, define a prediction cutoff, build RFM only from the past, define an outcome in a later observation window, and evaluate on a later time period to prevent temporal leakage.
- The source code currently prints results and saves figures/signature; it does not serialize the fitted scaler or classifier. Persist versioned model artifacts before using them in a deployment or scoring workflow.

## Business Use

Use the printed segment profile and charts to decide where to test retention, loyalty, reactivation, and product campaigns. The profile table reports the measured mean Recency, Frequency, Monetary value, segment size, and revenue share for each cluster. Treat campaign choices as testable hypotheses, and compare outcomes against a control group before scaling them.
