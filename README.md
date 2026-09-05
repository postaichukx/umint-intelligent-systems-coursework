# CTG Classification with MLP Neural Networks

A university coursework project focused on classifying fetal cardiotocography (CTG) records using multilayer perceptron (MLP) neural networks implemented with PyTorch.

The project compares several neural-network architectures and evaluates their performance on a three-class classification task:

- Normal
- Suspect
- Pathologic

## Features

- Data preprocessing and normalization
- Stratified train/test split
- Three MLP architectures:
  - M1: one hidden layer with 20 neurons
  - M2: one hidden layer with 60 neurons
  - M3: two hidden layers with 80 and 40 neurons
- Multiple runs with different random seeds
- Training-loss and accuracy visualizations
- Confusion matrices
- CSV files with experiment metrics and model comparisons

## Project Structure

```text
UMINT_cv06/
├── task6_ctg_mlp.py             # Main training and evaluation script
├── CTG_task6_summary.pdf        # Coursework report
├── CTGdata.csv                  # Required dataset (add manually)
├── task6_results/               # Generated plots, logs, and CSV metrics
│   ├── M1_history.png
│   ├── M1_confusion.png
│   ├── M2_history.png
│   ├── M2_confusion.png
│   ├── M3_history.png
│   ├── M3_confusion.png
│   ├── best_runs_comparison.png
│   ├── all_runs.csv
│   └── summary.csv
└── README.md
```

## Dataset

This project requires the `CTGdata.csv` dataset containing cardiotocography measurements and class labels.

Place the dataset in one of these locations:

```text
UMINT_cv06/CTGdata.csv
```

or

```text
UMINT_cv06/data/CTGdata.csv
```

The dataset is not included in this repository unless redistribution is permitted by the course or dataset source.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- PyTorch

Install the dependencies:

```bash
pip install numpy pandas matplotlib torch
```

## Run

```bash
python task6_ctg_mlp.py
```

After execution, the program saves plots, logs, confusion matrices, and evaluation results in the `task6_results/` directory.

## Results

The models are evaluated using test accuracy and loss. The project saves the best run for every architecture and provides a final comparison of all tested MLP models.

## Note

This project was created for educational purposes. It is not intended for clinical use, medical diagnosis, or treatment decisions.
