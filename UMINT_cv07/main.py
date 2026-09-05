from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "task7_results"

SEED = 42
VAL_RATIO = 0.10
EPOCHS = 5 #10
DROPOUT_EPOCHS = 8 #12
LEARNING_RATE = 0.001
BATCH_SIZE = 256
SHOW_PLOTS = True


def write_log(text: str) -> None:
    print(text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_split_indices(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        class_indices = rng.permutation(class_indices)
        n_val = int(len(class_indices) * val_ratio)
        val_idx.extend(class_indices[:n_val].tolist())
        train_idx.extend(class_indices[n_val:].tolist())

    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)
    return train_idx, val_idx


class MLP1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN2(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout_value = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        if self.dropout_value > 0.0:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true = []
    y_pred = []
    probs_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * len(x)
            total_correct += (pred == y).sum().item()
            total_count += len(x)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            probs_all.append(probs.cpu().numpy())

    return (
        total_loss / total_count,
        total_correct / total_count,
        np.array(y_true, dtype=int),
        np.array(y_pred, dtype=int),
        np.concatenate(probs_all, axis=0),
    )


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((10, 10), dtype=int)
    for real_class, pred_class in zip(y_true, y_pred):
        cm[real_class, pred_class] += 1
    return cm


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    model_name: str,
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += len(x)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if epoch == 1 or epoch == epochs or epoch == epochs // 2:
            write_log(
                f"{model_name} | epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.5f} | train_acc={train_acc * 100:.2f}% | "
                f"val_loss={val_loss:.5f} | val_acc={val_acc * 100:.2f}%"
            )

    train_time = time.perf_counter() - start

    train_loss, train_acc, _, _, _ = evaluate(model, train_loader, criterion, device)
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, y_true, y_pred, probs = evaluate(model, test_loader, criterion, device)
    cm = make_confusion_matrix(y_true, y_pred)

    return {
        "model": model,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "confusion_matrix": cm,
        "train_loss_history": train_loss_history,
        "train_acc_history": train_acc_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
        "train_time_sec": train_time,
    }


def save_history_plot(name: str, result: dict[str, object]) -> None:
    epochs = np.arange(1, len(result["train_loss_history"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, result["train_loss_history"], label="train")
    axes[0].plot(epochs, result["val_loss_history"], label="val")
    axes[0].set_title(f"{name} - loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, np.array(result["train_acc_history"]) * 100.0, label="train")
    axes[1].plot(epochs, np.array(result["val_acc_history"]) * 100.0, label="val")
    axes[1].set_title(f"{name} - accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy [%]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_history.png", dpi=180, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def save_confusion_plot(name: str, cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} - confusion matrix")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_confusion.png", dpi=180, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def save_examples_plot(name: str, model: nn.Module, test_dataset_raw, test_dataset_norm, device: torch.device) -> pd.DataFrame:
    model.eval()
    targets = test_dataset_raw.targets.numpy()
    rows = []

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.ravel()

    with torch.no_grad():
        for digit in range(10):
            index = int(np.where(targets == digit)[0][0])
            raw_image, true_class = test_dataset_raw[index]
            norm_image, _ = test_dataset_norm[index]

            logits = model(norm_image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

            axes[digit].imshow(raw_image.squeeze(0).numpy(), cmap="gray")
            axes[digit].set_title(f"T={true_class}, P={pred_class}")
            axes[digit].axis("off")

            row = {
                "index": index,
                "true_class": int(true_class),
                "predicted_class": pred_class,
            }
            for i in range(10):
                row[f"prob_{i}"] = float(probs[i])
            rows.append(row)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_examples.png", dpi=180, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = get_device()

    write_log(f"seed={SEED}")
    write_log(f"device={device}")
    write_log("")

    transform_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_raw = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform_norm)
    test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform_norm)
    test_dataset_raw = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform_raw)

    y_train = train_dataset.targets.numpy()
    train_idx, val_idx = make_split_indices(y_train, VAL_RATIO, SEED)

    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(train_dataset, val_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    write_log(f"train samples={len(train_subset)}")
    write_log(f"val samples={len(val_subset)}")
    write_log(f"test samples={len(test_dataset)}")
    write_log("")

    results_rows = []

    mlp_models = {
        "MLP1": MLP1(),
        "MLP2": MLP2(),
    }

    best_mlp_name = ""
    best_mlp_result = None
    best_mlp_score = -1.0

    for name, model in mlp_models.items():
        write_log(f"=== {name} ===")
        model = model.to(device)
        result = train_model(model, train_loader, val_loader, test_loader, device, EPOCHS, name)
        save_history_plot(name, result)
        save_confusion_plot(name, result["confusion_matrix"])

        results_rows.append(
            {
                "group": "MLP",
                "model": name,
                "train_acc_pct": 100.0 * result["train_acc"],
                "val_acc_pct": 100.0 * result["val_acc"],
                "test_acc_pct": 100.0 * result["test_acc"],
                "train_loss": result["train_loss"],
                "val_loss": result["val_loss"],
                "test_loss": result["test_loss"],
                "train_time_sec": result["train_time_sec"],
            }
        )

        write_log(
            f"{name} final | train_acc={100.0 * result['train_acc']:.2f}% | "
            f"val_acc={100.0 * result['val_acc']:.2f}% | test_acc={100.0 * result['test_acc']:.2f}%"
        )
        write_log("")

        if result["test_acc"] > best_mlp_score:
            best_mlp_score = result["test_acc"]
            best_mlp_name = name
            best_mlp_result = result

    cnn_models = {
        "CNN1": CNN1(),
        "CNN2": CNN2(),
        "CNN3": CNN3(),
    }

    best_cnn_name = ""
    best_cnn_result = None
    best_cnn_score = -1.0

    for name, model in cnn_models.items():
        write_log(f"=== {name} ===")
        model = model.to(device)
        result = train_model(model, train_loader, val_loader, test_loader, device, EPOCHS, name)
        save_history_plot(name, result)
        save_confusion_plot(name, result["confusion_matrix"])

        results_rows.append(
            {
                "group": "CNN",
                "model": name,
                "train_acc_pct": 100.0 * result["train_acc"],
                "val_acc_pct": 100.0 * result["val_acc"],
                "test_acc_pct": 100.0 * result["test_acc"],
                "train_loss": result["train_loss"],
                "val_loss": result["val_loss"],
                "test_loss": result["test_loss"],
                "train_time_sec": result["train_time_sec"],
            }
        )

        write_log(
            f"{name} final | train_acc={100.0 * result['train_acc']:.2f}% | "
            f"val_acc={100.0 * result['val_acc']:.2f}% | test_acc={100.0 * result['test_acc']:.2f}%"
        )
        write_log("")

        if result["test_acc"] > best_cnn_score:
            best_cnn_score = result["test_acc"]
            best_cnn_name = name
            best_cnn_result = result

    dropout_results = []

    for dropout_value in [0.0, 0.3, 0.5]:
        name = f"dropout_{dropout_value}"
        write_log(f"=== {name} ===")
        model = CNN2(dropout_value).to(device)
        result = train_model(model, train_loader, val_loader, test_loader, device, DROPOUT_EPOCHS, name)

        dropout_results.append(
            {
                "dropout": dropout_value,
                "train_acc_pct": 100.0 * result["train_acc"],
                "val_acc_pct": 100.0 * result["val_acc"],
                "test_acc_pct": 100.0 * result["test_acc"],
                "train_loss": result["train_loss"],
                "val_loss": result["val_loss"],
                "test_loss": result["test_loss"],
            }
        )

        write_log(
            f"{name} final | train_acc={100.0 * result['train_acc']:.2f}% | "
            f"val_acc={100.0 * result['val_acc']:.2f}% | test_acc={100.0 * result['test_acc']:.2f}%"
        )
        write_log("")

    assert best_mlp_result is not None
    assert best_cnn_result is not None

    best_mlp_examples = save_examples_plot(best_mlp_name, best_mlp_result["model"], test_dataset_raw, test_dataset, device)
    best_cnn_examples = save_examples_plot(best_cnn_name, best_cnn_result["model"], test_dataset_raw, test_dataset, device)

    results_df = pd.DataFrame(results_rows)
    dropout_df = pd.DataFrame(dropout_results)

    results_df.to_csv(RESULTS_DIR / "results.csv", index=False)
    dropout_df.to_csv(RESULTS_DIR / "dropout_results.csv", index=False)
    best_mlp_examples.to_csv(RESULTS_DIR / "best_mlp_examples.csv", index=False)
    best_cnn_examples.to_csv(RESULTS_DIR / "best_cnn_examples.csv", index=False)

    write_log("=== SUMMARY ===")
    write_log(results_df.to_string(index=False))
    write_log("")
    write_log("=== DROPOUT SUMMARY ===")
    write_log(dropout_df.to_string(index=False))
    write_log("")
    write_log(f"Best MLP: {best_mlp_name} | test_acc={100.0 * best_mlp_result['test_acc']:.2f}%")
    write_log(f"Best CNN: {best_cnn_name} | test_acc={100.0 * best_cnn_result['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
