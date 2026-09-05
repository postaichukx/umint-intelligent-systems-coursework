# MNIST Neural Network Comparison

A university project that compares multilayer perceptrons (MLPs) and convolutional neural networks (CNNs) for handwritten digit classification.

The program trains two MLP models and three CNN models on the MNIST dataset. It also compares different dropout values for one CNN model.

## Features

- MNIST digit classification from 0 to 9
- MLP and CNN model comparison
- Training, validation, and test evaluation
- Accuracy and loss charts
- Confusion matrices
- Example predictions for the best MLP and CNN models
- Dropout experiment for CNN2

## Requirements

- Python 3.10 or newer
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib

Install dependencies:

```bash
python -m pip install torch torchvision numpy pandas matplotlib
MNIST Dataset
This project requires the MNIST handwritten digit dataset.
You do not need to download it manually. When you run the program for the first time, PyTorch downloads the dataset automatically into:
data/MNIST/
The source code uses download=True when loading the dataset.
If the automatic download does not work, download the four MNIST files manually from the official page:
https://yann.lecun.org/exdb/mnist/index.html
Place the extracted files in:
data/MNIST/raw/
Run
python main.py
The program saves charts and CSV results in:
task7_results/
Project Structure
.
|-- main.py            # Training and evaluation code
|-- data/              # MNIST dataset, not included in the repository
`-- task7_results/     # Generated charts and result tables
