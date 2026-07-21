# Multiple Linear Regression Model Evaluation

This project builds a **Multiple Linear Regression** model to predict car mileage (`mpg`) from the car dataset in `data/3304db2c078848f8ad85537da4d87645_car_(1).csv`.

## Goal

Predict `mpg` using the available vehicle features and evaluate the model with standard regression metrics.

## Workflow

The script in `main.py` follows the workflow below step by step:

1. Import the required libraries.
2. Load the dataset with Pandas.
3. Clean the data by standardizing column names, removing duplicates, and handling missing values.
4. Detect and cap numeric outliers using the IQR rule.
5. Perform basic visual analysis with plots.
6. Split the dataset into training and testing sets.
7. Train a Multiple Linear Regression model.
8. Evaluate the model with MAE, MSE, and R2.

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

- Console output showing dataset shape, missing values, duplicate rows, and evaluation metrics.
- Visualizations saved in the `images/` folder:
  - `countplot_origin.png`
  - `distplot_mpg.png`
  - `correlation_heatmap.png`

## How to Run

From the repository root:

```bash
cd MultipleLinearRegressionModelEvaluation
python main.py
```

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

## Future Improvements

Possible next steps for this project:

- Outlier handling with a reproducible rule
- Residual analysis after training
- Comparison with Ridge and Lasso regression
- Saving the trained model for reuse
