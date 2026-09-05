# Neural Network Experiments with PyTorch

A collection of university exercises that use multilayer perceptrons (MLPs) for classification and function approximation.

## Included Tasks

### CTG Medical Data Classification

`main.py` trains and compares three MLP architectures on cardiotocography (CTG) data.

The task classifies records into three classes:

- Normal
- Suspect
- Pathologic

The experiment includes:

- Three MLP architectures: M1, M2, and M3
- Five runs for each architecture
- Training, validation, and test splits
- Z-score feature normalization
- Early stopping
- TensorBoard loss logging
- Confusion matrix
- Sensitivity and specificity for each class

### 3D Point Classification

`umint_5a.py` trains an MLP to classify points with `x`, `y`, and `z` coordinates.

The script:

- Uses an 80/20 training and test split
- Normalizes input features
- Displays a training-loss chart
- Displays a confusion matrix
- Predicts classes for five test points

### Nonlinear Function Approximation

`umint_5b.py` trains a deep MLP to approximate a nonlinear function.

The script:

- Uses a predefined training and test split
- Normalizes input and target values
- Uses three hidden layers with Tanh activation
- Evaluates SSE, MSE, and MAE
- Displays loss curves and predicted-function plots

## Requirements

- Python 3.10 or newer
- NumPy
- Pandas
- Matplotlib
- Plotly
- PyTorch
- scikit-learn
- TensorBoard

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib plotly torch scikit-learn tensorboard
```

## Run

Run each task separately:

```bash
python main.py
python umint_5a.py
python umint_5b.py
```

## Datasets

The repository includes these CSV files:

```text
CTGdata.csv       # Cardiotocography classification data
databody.csv      # 3D point classification data
datafun.csv       # Nonlinear function values
datafunindx.csv   # Training and test split labels
```

The scripts can also load course datasets from the STU FEI OUI GitHub data repository when an internet connection is available.

## Project Structure

```text
.
|-- main.py          # CTG classification with MLP models
|-- umint_5a.py      # 3D point classification
|-- umint_5b.py      # Nonlinear function approximation
|-- CTGdata.csv      # CTG dataset
|-- databody.csv     # 3D classification dataset
|-- datafun.csv      # Function data
`-- datafunindx.csv  # Train/test split labels
```

## Notes

This repository contains academic machine-learning experiments. The CTG classification task is for educational use only and must not be used for medical diagnosis or clinical decisions.
