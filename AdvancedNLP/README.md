# Advanced NLP: IMDB Sentiment Analysis

This project trains a binary LSTM sentiment classifier on the local IMDB review dataset using local 50-dimensional GloVe embeddings.

## Project Structure

- `data_loader.py`: Validates and loads `data/IMDB.csv` and `data/glove.txt`.
- `eda.py`: Reports label balance and builds unigram/bigram features with `CountVectorizer`.
- `preprocessing.py`: Cleans reviews, maps negative/positive labels to `0`/`1`, tokenizes, pads sequences, and creates the GloVe embedding matrix.
- `model.py`: Defines, trains, evaluates, and predicts with the LSTM model.
- `visualization.py`: Saves training accuracy and loss charts.
- `main.py`: Orchestrates the full pipeline and accepts custom reviews.

## Run

Install the workspace dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Run the full dataset workflow:

```powershell
Set-Location AdvancedNLP
python main.py --epochs 5 --review "A moving, well-acted movie with a satisfying ending."
```

For a fast smoke test, limit the dataset and epochs:

```powershell
python main.py --max-samples 1000 --epochs 1 --batch-size 64 --review "The story was dull and the acting was poor."
```

The training-history chart is saved at `images/training_history.png`. The test output includes accuracy, precision, recall, F1 score, and a per-class classification report.