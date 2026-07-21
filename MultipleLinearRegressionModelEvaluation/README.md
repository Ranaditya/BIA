# Multiple Linear Regression Model Evaluation

This project builds a **Multiple Linear Regression** model to predict car mileage (`mpg`) from the car dataset in `data/car_mileage_dataset.csv`.

## Goal

Predict `mpg` using the available vehicle features and evaluate the model with standard regression metrics.

## Project Architecture

The project is organized into focused modules with `main.py` as the orchestrator:

- `data_loader.py`
  - Loads the CSV dataset
  - Prints dataset inspection details (shape, dtypes, missing values, duplicates)
- `eda.py`
  - Prints descriptive statistics
  - Reports outlier counts with IQR logic
  - Generates EDA plots in `images/`
- `preprocessing.py`
  - Standardizes column names
  - Handles duplicates and empty strings
  - Applies IQR-based outlier capping on numeric columns
- `model.py`
  - Builds the Multiple Linear Regression pipeline
  - Performs train/test split
  - Trains and evaluates with MAE, MSE, and R2
- `main.py`
  - Coordinates all steps in order
  - Prints step headers and a final summary

## Workflow

The orchestration in `main.py` runs the workflow in this sequence:

1. Load and inspect dataset
2. Exploratory data analysis and visualization
3. Data preprocessing and outlier capping
4. Model training and evaluation
5. Final metric summary

## Dataset Notes

The dataset includes the following columns:

- `mpg` - target variable
- `cylinders`
- `displacement`
- `horsepower`
- `weight`
- `acceleration`
- `model_year`
- `origin`
- `name`

Important observations:

- `horsepower` contains missing values in the raw dataset.
- `origin` is categorical and is encoded during preprocessing.
- `name` is treated as a categorical field and is encoded by the pipeline.

## Outputs

When the script runs, it generates:

- Console output showing step-by-step pipeline execution and evaluation metrics.
- Visualizations saved in the `images/` folder:
  - `countplot_origin.png`
  - `distplot_mpg.png`
  - `correlation_heatmap.png`

## How to Run

From the repository root:

```bash
cd MultipleLinearRegressionModelEvaluation
..\.venv-1\Scripts\python.exe main.py
```

If you already activated your virtual environment, you can run:

```bash
python main.py
```

Plots are displayed with `plt.show()` and also saved as image files by default.

## Evaluation Metrics

The script reports the following regression metrics:

- **MAE** - Mean Absolute Error
- **MSE** - Mean Squared Error
- **R2** - Coefficient of determination

## Notes on Preprocessing

The preprocessing pipeline uses:

- Median imputation for numeric columns
- Most-frequent imputation for categorical columns
- One-hot encoding for categorical variables
- IQR-based capping for numeric outliers

This keeps the workflow simple, reproducible, and appropriate for a Multiple Linear Regression model.

## Folder Structure

```text
MultipleLinearRegressionModelEvaluation/
├── README.md
├── main.py
├── data_loader.py
├── eda.py
├── preprocessing.py
├── model.py
├── data/
│   └── car_mileage_dataset.csv
└── images/
  ├── countplot_origin.png
  ├── distplot_mpg.png
  └── correlation_heatmap.png
```

## Future Improvements

Possible next steps for this project:

- Outlier handling with a reproducible rule
- Residual analysis after training
- Comparison with Ridge and Lasso regression
- Saving the trained model for reuse
