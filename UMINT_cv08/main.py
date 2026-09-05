from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "umint_matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FULL_RESULTS_DIR = ROOT / "task8_results"

CLASS_NAMES = [
    "apple_pie",
    "caesar_salad",
    "clam_chowder",
    "edamame",
    "french_fries",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "sushi",
    "waffles",
]

MODEL_SPECS = [
    {
        "id": "M1",
        "arch": "AlexNet",
        "key": "alexnet",
        "builder": models.alexnet,
        "weights": models.AlexNet_Weights.IMAGENET1K_V1,
    },
    {
        "id": "M2",
        "arch": "ResNet18",
        "key": "resnet18",
        "builder": models.resnet18,
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
    },
    {
        "id": "M3",
        "arch": "MobileNetV2",
        "key": "mobilenet_v2",
        "builder": models.mobilenet_v2,
        "weights": models.MobileNet_V2_Weights.IMAGENET1K_V2,
    },
]


IMAGE_SIZE = 224
VAL_RATIO = 0.20
SPLIT_SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 20
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001
NUM_WORKERS = 2
RUN_SEEDS = [40, 41, 42]
RUN_AUGMENTED_EXPERIMENT = True
SAVE_VISUALS = True
SHOW_PLOTS = True

# Simple profile switch:
# - "quick_demo" fits a short real run on MacBook
# - "full" restores the complete assignment experiment
RUN_PROFILE = "quick_demo"

if RUN_PROFILE == "quick_demo":
    RESULTS_DIR = ROOT / "task8_results_quick"
    ACTIVE_MODEL_IDS = {"M3"}
    ACTIVE_MODES = ["tl"]
    ACTIVE_RUN_SEEDS = [41]
    ACTIVE_MAX_EPOCHS = 5
    ACTIVE_RUN_AUGMENTED_EXPERIMENT = False
    SHOW_PLOTS = True
else:
    RESULTS_DIR = FULL_RESULTS_DIR
    ACTIVE_MODEL_IDS = {model_spec["id"] for model_spec in MODEL_SPECS}
    ACTIVE_MODES = ["scratch", "tl"]
    ACTIVE_RUN_SEEDS = RUN_SEEDS
    ACTIVE_MAX_EPOCHS = MAX_EPOCHS
    ACTIVE_RUN_AUGMENTED_EXPERIMENT = RUN_AUGMENTED_EXPERIMENT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FilteredFoodDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices,
        label_map,
        transform=None,
    ):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_map = label_map
        self.transform = transform
        self.targets = np.array([label_map[int(base_dataset._labels[int(idx)])] for idx in self.indices], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = int(self.indices[idx])
        image, old_label = self.base_dataset[base_idx]
        label = self.label_map[int(old_label)]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_raw_item(self, idx):
        base_idx = int(self.indices[idx])
        image, old_label = self.base_dataset[base_idx]
        return image, self.label_map[int(old_label)]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_filtered_indices(base_dataset):
    for class_name in CLASS_NAMES:
        if class_name not in base_dataset.class_to_idx:
            raise ValueError(f"class {class_name!r} not found in Food101")

    label_map = {base_dataset.class_to_idx[class_name]: idx for idx, class_name in enumerate(CLASS_NAMES)}
    indices = np.array(
        [idx for idx, label in enumerate(base_dataset._labels) if int(label) in label_map],
        dtype=np.int64,
    )
    return indices, label_map


def split_train_val(base_dataset, filtered_indices, label_map, val_ratio, seed):
    labels = np.array([label_map[int(base_dataset._labels[int(idx)])] for idx in filtered_indices], dtype=np.int64)
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in range(len(CLASS_NAMES)):
        class_positions = np.where(labels == class_id)[0]
        class_positions = rng.permutation(class_positions)
        n_val = max(1, int(len(class_positions) * val_ratio))

        val_indices.extend(filtered_indices[class_positions[:n_val]].tolist())
        train_indices.extend(filtered_indices[class_positions[n_val:]].tolist())

    return (
        np.array(sorted(train_indices), dtype=np.int64),
        np.array(sorted(val_indices), dtype=np.int64)
    )


def make_transforms(use_augmentation):
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if not use_augmentation:
        return eval_transform, eval_transform

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.00)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def make_dataloaders(train_base, test_base, train_indices, val_indices, test_indices, label_map, use_augmentation, seed):
    train_transform, eval_transform = make_transforms(use_augmentation)

    train_dataset = FilteredFoodDataset(train_base, train_indices, label_map, transform=train_transform)
    val_dataset = FilteredFoodDataset(train_base, val_indices, label_map, transform=eval_transform)
    test_dataset = FilteredFoodDataset(test_base, test_indices, label_map, transform=eval_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS}
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, eval_transform


def build_model(model_spec, mode):
    weights = model_spec["weights"] if mode == "tl" else None
    model = model_spec["builder"](weights=weights)
    model_key = model_spec["key"]

    if model_key == "alexnet":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, len(CLASS_NAMES))
        if mode == "tl":
            for param in model.features.parameters():
                param.requires_grad = False

    elif model_key == "resnet18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(CLASS_NAMES))
        if mode == "tl":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

    elif model_key == "mobilenet_v2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
        if mode == "tl":
            for param in model.features.parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"unsupported model key: {model_key}")

    return model


def keep_frozen_backbone_in_eval(model, model_key, mode):
    if mode != "tl":
        return

    if model_key == "alexnet":
        model.features.eval()
    elif model_key == "resnet18":
        for name, module in model.named_children():
            if name != "fc":
                module.eval()
    elif model_key == "mobilenet_v2":
        model.features.eval()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * len(x_batch)
            total_correct += (preds == y_batch).sum().item()
            total_count += len(x_batch)
    return total_loss / total_count, total_correct / total_count


def train_one_run(model, model_spec, mode, seed, train_loader, val_loader, test_loader, device, use_augmentation):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=LEARNING_RATE)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    epochs_without_improvement = 0
    best_train_loss = 0.0
    best_train_acc = 0.0

    run_name = f"{model_spec['id']} {model_spec['arch']} | {mode} | seed={seed}"
    if use_augmentation:
        run_name += " | augmented"

    start_time = time.perf_counter()

    for epoch in range(1, ACTIVE_MAX_EPOCHS + 1):
        model.train()
        keep_frozen_backbone_in_eval(model, model_spec["key"], mode)

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x_batch)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_count += len(x_batch)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(
            f"{run_name} | epoch {epoch:02d}/{ACTIVE_MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} (CE) | train_acc={train_acc * 100.0:.2f}% | "
            f"val_loss={val_loss:.4f} (CE) | val_acc={val_acc * 100.0:.2f}%"
        )

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            best_train_loss = train_loss
            best_train_acc = train_acc
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"{run_name} | early stopping at epoch {epoch:02d} | "
                f"best_epoch={best_epoch:02d} | best_val_loss={best_val_loss:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    epochs_trained = len(train_loss_history)
    if best_state is None:
        best_train_loss = train_loss_history[0]
        best_train_acc = train_acc_history[0]

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    elapsed = time.perf_counter() - start_time

    return {
        "seed": seed,
        "train_loss": float(best_train_loss),
        "train_acc": float(best_train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "train_loss_history": train_loss_history,
        "train_acc_history": train_acc_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
        "train_time_sec": elapsed,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "model": model,
    }


def save_history_plot(name, result):
    epochs = np.arange(1, len(result["train_loss_history"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, result["train_loss_history"], label="train", color="tab:blue")
    axes[0].plot(epochs, result["val_loss_history"], label="val", color="tab:orange")
    axes[0].axvline(result["best_epoch"], color="tab:gray", linestyle="--", linewidth=1, label="best epoch")
    axes[0].set_title(f"{name} - loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, np.array(result["train_acc_history"]) * 100.0, label="train", color="tab:green")
    axes[1].plot(epochs, np.array(result["val_acc_history"]) * 100.0, label="val", color="tab:red")
    axes[1].axvline(result["best_epoch"], color="tab:gray", linestyle="--", linewidth=1, label="best epoch")
    axes[1].set_title(f"{name} - accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_history.png", dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_predictions_plot(name, model, raw_test_dataset, eval_transform, device):
    model.eval()
    figure_indices = []

    for class_id in range(len(CLASS_NAMES)):
        matches = np.where(raw_test_dataset.targets == class_id)[0]
        if len(matches) > 0:
            figure_indices.append(int(matches[0]))

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.ravel()

    with torch.no_grad():
        for ax, sample_idx in zip(axes, figure_indices):
            raw_image, true_label = raw_test_dataset.get_raw_item(sample_idx)
            input_tensor = eval_transform(raw_image).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_label = int(probs.argmax(dim=1).item())
            pred_confidence = float(probs.max(dim=1).values.item()) * 100.0

            ax.imshow(raw_image)
            ax.set_title(
                f"T: {CLASS_NAMES[true_label].replace('_', ' ')}\n"
                f"P: {CLASS_NAMES[pred_label].replace('_', ' ')} ({pred_confidence:.1f}%)",
                fontsize=9,
            )
            ax.axis("off")

    for ax in axes[len(figure_indices):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_predictions.png", dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device={device}")
    print(f"data_dir={DATA_DIR}")
    print(f"results_dir={RESULTS_DIR}")
    print(
        "config="
        f"profile={RUN_PROFILE}, max_epochs={ACTIVE_MAX_EPOCHS}, optimizer=Adam, lr={LEARNING_RATE}, "
        f"batch_size={BATCH_SIZE}, early_stopping_patience={EARLY_STOPPING_PATIENCE}, "
        f"run_seeds={ACTIVE_RUN_SEEDS}, augmented={ACTIVE_RUN_AUGMENTED_EXPERIMENT}"
    )
    print("")

    train_base = datasets.Food101(root=str(DATA_DIR), split="train", download=True)
    test_base = datasets.Food101(root=str(DATA_DIR), split="test", download=True)

    train_indices_all, label_map = build_filtered_indices(train_base)
    test_indices, _ = build_filtered_indices(test_base)
    train_indices, val_indices = split_train_val(train_base, train_indices_all, label_map, VAL_RATIO, SPLIT_SEED)

    raw_test_dataset = FilteredFoodDataset(test_base, test_indices, label_map, transform=None)

    train_counts = np.bincount(
        [label_map[int(train_base._labels[int(idx)])] for idx in train_indices],
        minlength=len(CLASS_NAMES),
    )
    val_counts = np.bincount(
        [label_map[int(train_base._labels[int(idx)])] for idx in val_indices],
        minlength=len(CLASS_NAMES),
    )
    test_counts = np.bincount(
        [label_map[int(test_base._labels[int(idx)])] for idx in test_indices],
        minlength=len(CLASS_NAMES),
    )

    print(f"selected classes={', '.join(CLASS_NAMES)}")
    print(f"train samples={len(train_indices)}")
    print(f"val samples={len(val_indices)}")
    print(f"test samples={len(test_indices)}")
    print("")
    print("train class counts:")
    print(dict(zip(CLASS_NAMES, train_counts.tolist())))
    print("val class counts:")
    print(dict(zip(CLASS_NAMES, val_counts.tolist())))
    print("test class counts:")
    print(dict(zip(CLASS_NAMES, test_counts.tolist())))
    print("")

    active_model_specs = [model_spec for model_spec in MODEL_SPECS if model_spec["id"] in ACTIVE_MODEL_IDS]

    best_model_spec = None
    best_mode = ""
    best_baseline_acc = -1.0
    best_baseline_loss = float("inf")

    for model_spec in active_model_specs:
        for mode in ACTIVE_MODES:
            print(f"=== {model_spec['id']} | {model_spec['arch']} | {mode} ===")
            run_results = []
            eval_transform = None

            for seed in ACTIVE_RUN_SEEDS:
                set_seed(seed)
                train_loader, val_loader, test_loader, eval_transform = make_dataloaders(
                    train_base, test_base, train_indices, val_indices, test_indices, label_map, False, seed
                )
                model = build_model(model_spec, mode).to(device)

                result = train_one_run(
                    model, model_spec, mode, seed, train_loader, val_loader, test_loader, device, False
                )
                run_results.append(result)
                print(
                    f"final | seed={seed} | train_acc={result['train_acc'] * 100.0:.2f}% | "
                    f"val_acc={result['val_acc'] * 100.0:.2f}% | "
                    f"test_acc={result['test_acc'] * 100.0:.2f}%"
                )

            mean_train_acc = float(np.mean([result["train_acc"] for result in run_results]))
            mean_val_acc = float(np.mean([result["val_acc"] for result in run_results]))
            mean_test_acc = float(np.mean([result["test_acc"] for result in run_results]))
            mean_test_loss = float(np.mean([result["test_loss"] for result in run_results]))
            mean_epochs = float(np.mean([result["epochs_trained"] for result in run_results]))

            print(
                f"mean | train_acc={mean_train_acc * 100.0:.2f}% | "
                f"val_acc={mean_val_acc * 100.0:.2f}% | "
                f"test_acc={mean_test_acc * 100.0:.2f}% | "
                f"test_loss={mean_test_loss:.4f} | epochs={mean_epochs:.2f}"
            )

            if SAVE_VISUALS:
                best_run = max(run_results, key=lambda result: (result["val_acc"], -result["val_loss"]))
                plot_name = f"{model_spec['id']}_{mode}"
                save_history_plot(plot_name, best_run)
                assert eval_transform is not None
                save_predictions_plot(plot_name, best_run["model"], raw_test_dataset, eval_transform, device)
            print("")

            if (
                mean_test_acc > best_baseline_acc
                or (mean_test_acc == best_baseline_acc and mean_test_loss < best_baseline_loss)
            ):
                best_model_spec = model_spec
                best_mode = mode
                best_baseline_acc = mean_test_acc
                best_baseline_loss = mean_test_loss

    assert best_model_spec is not None
    print(
        f"best without augmentation={best_model_spec['id']} | {best_model_spec['arch']} | {best_mode} | "
        f"mean_test_acc={best_baseline_acc * 100.0:.2f}%"
    )
    print("")

    augmented_results = []
    augmented_eval_transform = None
    if ACTIVE_RUN_AUGMENTED_EXPERIMENT:
        print(f"=== AUGMENTED | {best_model_spec['id']} | {best_model_spec['arch']} | {best_mode} ===")

        for seed in ACTIVE_RUN_SEEDS:
            set_seed(seed)
            train_loader, val_loader, test_loader, augmented_eval_transform = make_dataloaders(
                train_base, test_base, train_indices, val_indices, test_indices, label_map, True, seed
            )
            model = build_model(best_model_spec, best_mode).to(device)

            result = train_one_run(
                model, best_model_spec, best_mode, seed, train_loader, val_loader, test_loader, device, True
            )
            augmented_results.append(result)
            print(
                f"aug final | seed={seed} | train_acc={result['train_acc'] * 100.0:.2f}% | "
                f"val_acc={result['val_acc'] * 100.0:.2f}% | test_acc={result['test_acc'] * 100.0:.2f}%"
            )

        mean_aug_val_acc = float(np.mean([result["val_acc"] for result in augmented_results]))
        mean_aug_test_acc = float(np.mean([result["test_acc"] for result in augmented_results]))
        mean_aug_test_loss = float(np.mean([result["test_loss"] for result in augmented_results]))
        mean_aug_epochs = float(np.mean([result["epochs_trained"] for result in augmented_results]))
        print(
            f"aug mean | val_acc={mean_aug_val_acc * 100.0:.2f}% | "
            f"test_acc={mean_aug_test_acc * 100.0:.2f}% | "
            f"test_loss={mean_aug_test_loss:.4f} | epochs={mean_aug_epochs:.2f}"
        )

        if SAVE_VISUALS:
            best_augmented_run = max(augmented_results, key=lambda result: (result["val_acc"], -result["val_loss"]))
            save_history_plot(f"{best_model_spec['id']}_{best_mode}_augmented", best_augmented_run)
            assert augmented_eval_transform is not None
            save_predictions_plot(
                f"{best_model_spec['id']}_{best_mode}_augmented",
                best_augmented_run["model"],
                raw_test_dataset,
                augmented_eval_transform,
                device,
            )
        print("")

    print("done")


if __name__ == "__main__":
    main()
