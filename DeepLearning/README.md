# CIFAR-10 Image Classification with a CNN

This educational project trains a Convolutional Neural Network (CNN) to classify
32 x 32 color images from the CIFAR-10 dataset into ten categories.

## Project structure

```text
DeepLearning/
|-- main.py               # Orchestrates the complete 10-step pipeline
|-- data_loader.py        # Loads and inspects CIFAR-10
|-- preprocessing.py      # Normalizes image pixels
|-- model.py              # Builds, compiles, trains, evaluates, and predicts
|-- visualization.py      # Creates training and prediction plots
|-- images/               # Generated plots (created when run)
`-- models/               # Best saved model (created when run)
```

## Setup

This repository uses one Python 3.13 environment and one dependency file at its
root. From the repository root, run:

```powershell
py -3.13 -m venv .venv-1
.\.venv-1\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
cd DeepLearning
```

Confirm that the activated environment is correct with `python --version`; it
should report Python 3.13.x, not 3.14.x.

## Run

```powershell
python main.py
```

The first run downloads CIFAR-10. By default, training runs for at most 15 epochs;
early stopping may finish sooner. Outputs are saved even when plot windows are not
shown.

Useful options:

```powershell
python main.py --epochs 2                 # Quick smoke run
python main.py --epochs 20 --batch-size 128
python main.py --predictions 15 --show-plots
```

## Pipeline

1. Import TensorFlow/Keras, NumPy, and Matplotlib.
2. Load CIFAR-10 through Keras.
3. Convert pixels from 0-255 integers to 0-1 floats.
4. Build a CNN with convolution, pooling, normalization, dropout, and dense layers.
5. Compile it with Adam, sparse categorical cross-entropy, and accuracy.
6. Train it while retaining the Keras training history and best checkpoint.
7. Evaluate loss and accuracy on the held-out test set.
8. Save loss and accuracy curves to `images/training_history.png`.
9. Predict labels and confidence scores for test images.
10. Save an annotated image grid to `images/test_predictions.png`.
