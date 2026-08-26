# IMDb Sentiment Classification

This project follows the modular structure used by `MachineLearningIntroduction` and classifies IMDb movie reviews as positive or negative with a TensorFlow/Keras neural network.

## Dataset

The bundled file is `data/IMDB.csv` with 50,000 labeled reviews and two columns:

- `review`: raw movie-review text, including HTML break tags in some records
- `sentiment`: `positive` or `negative`

## Project Structure

```text
NLP/
├── data/IMDB.csv          # IMDb reviews and labels
├── data_loader.py         # Load and inspect the CSV
├── eda.py                 # Class balance and review-length analysis
├── preprocessing.py       # Cleaning, split, tokenization, and padding
├── model.py               # Embedding + Flatten + Dense classifier
├── main.py                # End-to-end entry point
├── images/                # Generated EDA and training plots
└── README.md
```

## Setup and Run

From the repository root:

```powershell
.\.venv-1\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Set-Location NLP
python main.py
```

The default configuration uses an 80/20 stratified train/test split, a 20,000-word vocabulary, sequences padded or truncated to 200 tokens, five training epochs, and a batch size of 128. The tokenizer is fit only on training reviews to avoid test-set vocabulary leakage.

## Model and Evaluation

The model uses an integer token sequence, a 64-dimensional `Embedding` layer, `Flatten`, and a sigmoid `Dense` output for binary classification. It is compiled with Adam, binary cross-entropy, and accuracy. Evaluation includes test loss, accuracy, thresholded predictions, a classification report, and a confusion matrix.

Generated plots include:

- `images/sentiment_distribution.png`
- `images/review_lengths.png`
- `images/training_history.png`

Training and validation curves help identify underfitting or overfitting. A widening gap where training accuracy rises while validation accuracy stalls indicates that regularization, a smaller model, or early stopping may improve generalization. A recurrent model such as LSTM or a convolutional text model is a natural next experiment for stronger sequence modeling.
