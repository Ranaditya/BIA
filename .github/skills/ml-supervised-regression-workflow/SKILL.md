---
name: ml-supervised-regression-workflow
user-invocable: true
description: "Build and document supervised regression workflows from CSV datasets, including simple linear regression and multiple linear regression. Use when users ask for EDA, preprocessing, train-test split, model training, MAE MSE R2 evaluation, modular Python structure, and step-by-step approval gates. Trigger phrases include: build linear regression model, multiple linear regression from CSV, predict numeric target, evaluate with MAE MSE R2, and modular regression pipeline with data_loader eda preprocessing model main."
---

# ML Supervised Regression Workflow

Use this skill to implement and document supervised regression projects.

## Use When

- User asks to build a regression model from CSV data.
- User requests linear regression or multiple linear regression workflow.
- User expects EDA, preprocessing, training, evaluation, and documentation.

## Do Not Use When

- Target is categorical and user asks for classification.
- User asks for clustering, PCA, or unsupervised learning.
- User asks only for deployment and not model-building.

## Required Inputs

1. Dataset path
2. Target column
3. Feature columns (or instruction to infer candidates)

## Optional Inputs with Defaults

- test_size: 0.2
- random_state: 42
- missing numeric strategy: median imputation
- missing categorical strategy: most frequent imputation
- outlier strategy: IQR capping
- baseline metrics: MAE, MSE, R2

## Recommended Module Layout

- data_loader.py
- eda.py
- preprocessing.py
- model.py
- main.py
- README.md

## Workflow Steps

1. Validate dataset availability and schema.
2. Inspect data types, missing values, duplicates, and target quality.
3. Run EDA visuals and summary stats.
4. Apply preprocessing choices (missing data, encoding, outliers).
5. Split train and test sets.
6. Build and train regression model.
7. Evaluate with MAE, MSE, and R2.
8. Document outputs and interpretation.

## Execution Rules

1. Propose one recommended step at a time.
2. Wait for approval before code edits or execution.
3. Create a feature branch for new work.
4. Keep workflow reproducible and easy to run.
5. Validate by running the project with the selected environment.

## Reporting Standard

- Show final metric values.
- Explain tradeoffs and quality briefly.
- List generated artifacts (plots, files, docs).
- Prepare PR-ready summary with validation evidence.

## Notes for This Repository

- Align style with projects in MachineLearningIntroduction and MultipleLinearRegressionModelEvaluation.
- Prefer modular architecture and explicit orchestration from main.py.

## Trigger Examples

- "Build a multiple linear regression model from this CSV and document each step."
- "Create a linear regression pipeline with EDA, preprocessing, train/test split, and MAE MSE R2."
- "Modularize my regression project into data_loader, eda, preprocessing, model, and main."
