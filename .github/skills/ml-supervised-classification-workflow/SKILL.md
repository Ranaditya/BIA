---
name: ml-supervised-classification-workflow
user-invocable: true
description: "Build and document supervised classification workflows from CSV datasets, including logistic regression baselines. Use when users ask for EDA, preprocessing, train-test split, classification training, accuracy/precision/recall/f1 evaluation, modular Python structure, and step-by-step approval gates."
---

# ML Supervised Classification Workflow

Use this skill to implement and document supervised classification projects.

## Use When

- User asks to build a classification model from CSV data.
- User requests logistic regression classification workflow.
- User expects EDA, preprocessing, training, evaluation, and documentation.

## Do Not Use When

- Target is numeric and user asks for regression metrics (MAE, MSE, R2).
- User asks for clustering, PCA, or unsupervised learning.
- User asks only for deployment and not model-building.

## Required Inputs

1. Dataset path
2. Target column
3. Feature columns (or instruction to infer candidates)
4. Workflow type intent (classification)
5. Instruction set (if provided by user)

## Optional Inputs with Defaults

- test_size: 0.2
- random_state: 42
- missing numeric strategy: median imputation
- missing categorical strategy: most frequent imputation
- outlier strategy: IQR capping
- baseline metrics: accuracy, precision, recall, f1

## Input Contract

- dataset_path
- workflow_type=classification
- target_column
- feature_columns
- test_size (default 0.2)
- random_state (default 42)
- instruction_set (optional)

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
6. Build and train classification model.
7. Evaluate with accuracy, precision, recall, and f1.
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

## Trigger Examples

- "Build a logistic regression classification model from this CSV and document each step."
- "Create a classification pipeline with EDA, preprocessing, train/test split, and precision-recall evaluation."
- "Modularize my classification project into data_loader, eda, preprocessing, model, and main."
