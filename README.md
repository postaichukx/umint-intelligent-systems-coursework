# Food-101 Image Classification with Transfer Learning

A university coursework project that compares convolutional neural-network architectures for food image classification using PyTorch and the Food-101 dataset.

## Classification Task

The project uses 10 selected Food-101 classes:

- Apple pie
- Caesar salad
- Clam chowder
- Edamame
- French fries
- Hamburger
- Hot dog
- Ice cream
- Sushi
- Waffles

## Models

The project evaluates three architectures:

- `M1` — AlexNet
- `M2` — ResNet18
- `M3` — MobileNetV2

Each architecture can be trained:

- from scratch
- with ImageNet pretrained weights using transfer learning

The full experiment also tests data augmentation and runs models with multiple random seeds.

## Dataset

The program uses the Food-101 dataset through TorchVision.

You do not need to upload the dataset to GitHub. On the first run, the script downloads it automatically into:

```text
data/food-101/
```

An internet connection and sufficient disk space are required.

Dataset source: [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

## Requirements

- Python 3.10 or newer
- PyTorch
- TorchVision
- NumPy
- Matplotlib

Install the dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

## Run

```bash
python main.py
```

The default configuration uses the `quick_demo` profile:

- MobileNetV2
- transfer learning
- one random seed
- five training epochs

To run the complete experiment, change this line in `main.py`:

```python
RUN_PROFILE = "quick_demo"
```

to:

```python
RUN_PROFILE = "full"
```

## Results

The script saves training histories and prediction visualizations in:

```text
task8_results_quick/
```

The full profile saves results in:

```text
task8_results/
```

## Project Structure

```text
UMINT_cv08/
├── main.py
├── rebuild_console_plots.py
├── task8_results_quick/
├── task8_results/
└── README.md
```

## Notes

This project was created for educational purposes as part of university coursework.
