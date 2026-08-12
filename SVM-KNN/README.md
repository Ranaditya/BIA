# Iris Classification with SVM and KNN

This project loads the Iris CSV from `data/`, explores the dataset, prepares a
stratified train/test split, and compares a linear Support Vector Machine with a
3-neighbor K-Nearest Neighbors classifier.

## Structure

- `data_loader.py` - finds, loads, validates, and inspects the CSV
- `eda.py` - summary statistics and exploratory visualizations
- `preprocessing.py` - label encoding, stratified splitting, and standardization
- `model.py` - SVM/KNN training, prediction, metrics, and confusion matrices
- `main.py` - end-to-end orchestration of all 17 requested steps
- `data/` - Iris dataset
- `images/` - generated EDA and model evaluation plots

## Run

```powershell
cd C:\Users\vaidi\BIA
.\.venv-1\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd SVM-KNN
python main.py
```

The split is reproducible (`random_state=189`) and stratified. The scaler is fit
only on the training features and then applied to the test features, preventing
test-data leakage.
