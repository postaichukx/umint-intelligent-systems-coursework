from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "umint_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "task6_results"
LOG_PATH = RESULTS_DIR / "task6_log.txt"


LOCAL_DATA_CANDIDATES = [
    ROOT / "CTGdata.csv",
    ROOT / "data" / "CTGdata.csv",
    Path("/Users/gnomik7/Downloads/CTGData_cv6/CTGdata.csv"),
]
DOWNLOADS_CTG_PATH = Path("/Users/gnomik7/Downloads/CTGData_cv6/CTG.csv")
DOWNLOADS_CTG_COLUMNS = [
    "b",
    "e",
    "LBE",
    "LB",
    "AC",
    "FM",
    "UC",
    "ASTV",
    "MSTV",
    "ALTV",
    "MLTV",
    "DL",
    "DS",
    "DP",
    "DR",
    "Width",
    "Min",
    "Max",
    "Nmax",
    "Nzeros",
    "Mode",
    "Mean",
    "Median",
    "Variance",
    "Tendency",
    "NSP",
]

RUN_SEEDS = [40, 41, 42, 43, 44]
STRUCTURES = {
    "M1": (20,),
    "M2": (60,),
    "M3": (80, 40),
}

TRAIN_RATIO = 0.60
EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
DEVICE = torch.device("cpu")
SHOW_PLOTS_IN_PYCHARM = True

CLASS_NAMES = {
    0: "normalny (1)",
    1: "podozrivy (2)",
    2: "patologicky (3)",
}


def write_log(log_path: Path, message: str) -> None:
    print(message)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def save_plots(
    output_dir: Path,
    model_id: str,
    run_seed: int,
    loss_history: list[float],
    train_acc_history: list[float],
    confusion_matrix: np.ndarray,
    show_in_pycharm: bool = False,
) -> None:
    epochs = np.arange(1, len(loss_history) + 1)
    open_figures: list[plt.Figure] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, loss_history, color="tab:blue")
    axes[0].set_title(f"{model_id} - train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, np.array(train_acc_history) * 100.0, color="tab:green")
    axes[1].set_title(f"{model_id} - train accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy [%]")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{model_id} best run - seed {run_seed}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_id}_history.png", dpi=180, bbox_inches="tight")
    if show_in_pycharm:
        open_figures.append(fig)
    else:
        plt.close(fig)

    labels = [CLASS_NAMES[idx] for idx in range(confusion_matrix.shape[0])]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_title(f"{model_id} - confusion matrix (seed {run_seed})")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for row_idx in range(confusion_matrix.shape[0]):
        for col_idx in range(confusion_matrix.shape[1]):
            ax.text(col_idx, row_idx, str(confusion_matrix[row_idx, col_idx]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_id}_confusion.png", dpi=180, bbox_inches="tight")
    if show_in_pycharm:
        open_figures.append(fig)
    else:
        plt.close(fig)

    if show_in_pycharm:
        plt.show()
        for figure in open_figures:
            plt.close(figure)


class SimpleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_ctg_dataframe() -> pd.DataFrame:
    for candidate in LOCAL_DATA_CANDIDATES:
        if candidate.exists():
            return pd.read_csv(candidate)

    if DOWNLOADS_CTG_PATH.exists():
        df = pd.read_csv(DOWNLOADS_CTG_PATH)
        df = df[DOWNLOADS_CTG_COLUMNS].dropna().reset_index(drop=True)
        df.columns = [str(i) for i in range(25)] + ["typ_ochorenia"]
        return df

    raise FileNotFoundError("cvs file not found")


def make_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for target, pred in zip(y_true, y_pred):
        cm[target, pred] += 1
    return cm


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict[str, np.ndarray | float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    prob_all: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(X_batch)
            total_correct += (preds == y_batch).sum().item()
            total_count += len(X_batch)
            y_true_all.extend(y_batch.cpu().numpy().tolist())
            y_pred_all.extend(preds.cpu().numpy().tolist())
            prob_all.append(probs.cpu().numpy())

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
        "y_true": np.array(y_true_all, dtype=np.int64),
        "y_pred": np.array(y_pred_all, dtype=np.int64),
        "probs": np.concatenate(prob_all, axis=0),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    write_log(
        LOG_PATH,
        f"Shared hyperparameters: epochs={EPOCHS}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, train_ratio={TRAIN_RATIO}",
    )

    df = load_ctg_dataframe()  # nacitame data z csv
    X_all = df[[str(i) for i in range(25)]].values.astype(np.float32)  # vyberieme 25 vstupnych priznakov
    y_raw = df["typ_ochorenia"].values.astype(int)  # nacitame povodne triedy 1,2,3
    classes = np.sort(np.unique(y_raw))  # zistime ake triedy su v datasete
    class_to_idx = {class_value: idx for idx, class_value in enumerate(classes)}  # mapovanie tried na 0,1,2
    y_all = np.array([class_to_idx[value] for value in y_raw], dtype=np.int64)  # prevedieme triedy na indexy

    write_log(
        LOG_PATH,
        "Class counts: "
        + ", ".join(
            f"{CLASS_NAMES[idx]}={int((y_all == idx).sum())}" for idx in range(len(classes))
        ),
    )

    all_run_rows: list[dict[str, float | int | str]] = []  # sem ulozime vsetky behy
    best_runs: dict[str, dict[str, object]] = {}  # najlepsi beh pre kazdu strukturu
    best_overall: dict[str, object] | None = None  # najlepsi beh zo vsetkych

    for model_id, hidden_layers in STRUCTURES.items():  # porovname M1, M2 a M3
        write_log(LOG_PATH, "")
        write_log(LOG_PATH, f"=== {model_id} | hidden_layers:{hidden_layers} ")
        structure_runs: list[dict[str, object]] = []  # behy jednej struktury

        for run_seed in RUN_SEEDS:  # opakujeme trening pre rozne seedy
            set_seed(run_seed)  # nastavime rovnaku nahodnost pre tento beh

            train_idx: list[int] = []  # indexy train vzoriek
            test_idx: list[int] = []  # indexy test vzoriek
            rng = np.random.default_rng(run_seed)  # generator pre stratified split
            for class_idx in np.unique(y_all):  # delime kazdu triedu zvlast
                class_indices = np.where(y_all == class_idx)[0]  # indexy jednej triedy
                class_indices = np.sort(class_indices)  # zoradime indexy
                class_indices = rng.permutation(class_indices)  # nahodne premiesame triedu
                n_train = int(len(class_indices) * TRAIN_RATIO)  # vezmeme 60 percent do train
                train_idx.extend(class_indices[:n_train])  # pridame train cast
                test_idx.extend(class_indices[n_train:])  # zvysok ide do test

            train_idx = np.array(sorted(train_idx), dtype=int)  # finalne train indexy
            test_idx = np.array(sorted(test_idx), dtype=int)  # finalne test indexy

            X_train_raw = X_all[train_idx]  # train vstupy pred normalizaciou
            X_test_raw = X_all[test_idx]  # test vstupy pred normalizaciou
            y_train = y_all[train_idx]  # train triedy
            y_test = y_all[test_idx]  # test triedy

            x_mean = X_train_raw.mean(axis=0, keepdims=True)  # priemer z train dat
            x_std = X_train_raw.std(axis=0, keepdims=True) + 1e-8  # smerodajna odchylka z train dat
            X_train = ((X_train_raw - x_mean) / x_std).astype(np.float32)  # normalizujeme train
            X_test = ((X_test_raw - x_mean) / x_std).astype(np.float32)  # normalizujeme test podla train
            X_full = ((X_all - x_mean) / x_std).astype(np.float32)  # normalizujeme cely dataset

            train_loader = DataLoader(
                SimpleDataset(X_train, y_train),  # train data pre batch trening
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(run_seed),
            )
            test_loader = DataLoader(SimpleDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)  # test data
            full_loader = DataLoader(SimpleDataset(X_full, y_all), batch_size=BATCH_SIZE, shuffle=False)  # cele data

            model = SimpleMLP(X_train.shape[1], hidden_layers, len(classes)).to(DEVICE)  # vytvorime MLP
            criterion = nn.CrossEntropyLoss()  # funkcia chyby pre 3 triedy
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # optimizer pre upravu vah

            loss_history: list[float] = []  # historia train loss
            train_acc_history: list[float] = []  # historia train accuracy

            write_log(
                LOG_PATH,
                f"[{model_id}] seed={run_seed} | train={len(train_idx)} | test={len(test_idx)}",
            )

            for epoch in range(1, EPOCHS + 1):  # jedna epocha = jeden prechod train datami
                model.train()  # prepneme model do train rezimu
                total_loss = 0.0  # sucet chyb v epoche
                total_correct = 0  # pocet spravnych predikcii
                total_count = 0  # pocet vzoriek v epoche

                for X_batch, y_batch in train_loader:  # spracujeme jeden minibatch
                    X_batch = X_batch.to(DEVICE)  # presun vstupov na device
                    y_batch = y_batch.to(DEVICE)  # presun tried na device
                    optimizer.zero_grad()  # vynulujeme stare gradienty
                    logits = model(X_batch)  # forward pass cez siet
                    loss = criterion(logits, y_batch)  # vypocitame chybu batchu
                    loss.backward()  # spocitame gradienty
                    optimizer.step()  # upravime vahy modelu

                    total_loss += loss.item() * len(X_batch)  # priratame batch loss
                    total_correct += (logits.argmax(dim=1) == y_batch).sum().item()  # priratame spravne triedy
                    total_count += len(X_batch)  # priratame pocet vzoriek

                epoch_loss = total_loss / total_count  # priemerna chyba epohy
                epoch_acc = total_correct / total_count  # uspesnost epohy
                loss_history.append(epoch_loss)  # ulozime loss do historie
                train_acc_history.append(epoch_acc)  # ulozime accuracy do historie

                if epoch in {1, 25, 50, 75, EPOCHS}:
                    write_log(
                        LOG_PATH,
                        f"  epoch {epoch:3d}/{EPOCHS} | train_loss={epoch_loss:.6f} | train_acc={epoch_acc * 100:.2f}%",
                    )

            train_eval = evaluate(model, train_loader, criterion)  # vyhodnotime train data
            test_eval = evaluate(model, test_loader, criterion)  # vyhodnotime test data
            full_eval = evaluate(model, full_loader, criterion)  # vyhodnotime cele data
            cm = make_confusion_matrix(test_eval["y_true"], test_eval["y_pred"], len(classes))  # matica zamen

            run_result = {  # ulozime vysledky jedneho behu
                "model_id": model_id,
                "hidden_layers": hidden_layers,
                "seed": run_seed,
                "model": model,
                "train_acc": float(train_eval["acc"]),
                "test_acc": float(test_eval["acc"]),
                "full_acc": float(full_eval["acc"]),
                "train_loss": float(train_eval["loss"]),
                "test_loss": float(test_eval["loss"]),
                "full_loss": float(full_eval["loss"]),
                "loss_history": loss_history,
                "train_acc_history": train_acc_history,
                "confusion_matrix": cm,
                "test_eval": test_eval,
                "test_indices": test_idx,
            }
            structure_runs.append(run_result)  # pridame beh do danej struktury

            all_run_rows.append(
                {  # riadok pre finalnu tabulku
                    "model_id": model_id,
                    "hidden_layers": ",".join(str(x) for x in hidden_layers),
                    "seed": run_seed,
                    "train_acc_pct": 100.0 * run_result["train_acc"],
                    "test_acc_pct": 100.0 * run_result["test_acc"],
                    "full_acc_pct": 100.0 * run_result["full_acc"],
                    "train_loss": run_result["train_loss"],
                    "test_loss": run_result["test_loss"],
                    "full_loss": run_result["full_loss"],
                }
            )

            if best_overall is None or (  # priebezne hladame najlepsi model
                run_result["test_acc"],
                -run_result["test_loss"],
            ) > (
                best_overall["test_acc"],
                -best_overall["test_loss"],
            ):
                best_overall = run_result

        best_run = max(  # najlepsi beh danej struktury
            structure_runs,
            key=lambda row: (row["test_acc"], -row["test_loss"]),
        )
        best_runs[model_id] = best_run  # ulozime najlepsi beh struktury

        save_plots(
            RESULTS_DIR,
            model_id,
            int(best_run["seed"]),
            list(best_run["loss_history"]),
            list(best_run["train_acc_history"]),
            np.array(best_run["confusion_matrix"]),
        )

    runs_df = pd.DataFrame(all_run_rows)  # vsetky behy v jednej tabulke
    summary_rows: list[dict[str, float | str]] = []  # sem ulozime suhrn struktur
    for model_id, hidden_layers in STRUCTURES.items():  # ratame min max mean pre kazdu siet
        sub = runs_df[runs_df["model_id"] == model_id]  # vyberieme len jednu strukturu
        summary_rows.append(
            {  # sumar metrik pre danu strukturu
                "model_id": model_id,
                "hidden_layers": ",".join(str(x) for x in hidden_layers),
                "train_min_acc_pct": float(sub["train_acc_pct"].min()),
                "train_max_acc_pct": float(sub["train_acc_pct"].max()),
                "train_mean_acc_pct": float(sub["train_acc_pct"].mean()),
                "test_min_acc_pct": float(sub["test_acc_pct"].min()),
                "test_max_acc_pct": float(sub["test_acc_pct"].max()),
                "test_mean_acc_pct": float(sub["test_acc_pct"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)  # finalna sumarizacia struktur

    best_runs_rows = []  # tabulka najlepsich behov
    for model_id in STRUCTURES:
        row = best_runs[model_id]  # najlepsi beh jednej struktury
        best_runs_rows.append(
            {  # ulozime jeho hlavne metriky
                "model_id": model_id,
                "hidden_layers": ",".join(str(x) for x in row["hidden_layers"]),
                "best_seed": int(row["seed"]),
                "train_acc_pct": 100.0 * float(row["train_acc"]),
                "test_acc_pct": 100.0 * float(row["test_acc"]),
                "full_acc_pct": 100.0 * float(row["full_acc"]),
                "train_loss": float(row["train_loss"]),
                "test_loss": float(row["test_loss"]),
                "full_loss": float(row["full_loss"]),
            }
        )
    best_runs_df = pd.DataFrame(best_runs_rows)  # tabulka najlepsich behov

    comparison_fig = plt.figure(figsize=(12, 4.5))  # porovnanie najlepsich behov M1 M2 M3
    comparison_axes = comparison_fig.subplots(1, 2)
    max_epochs = max(len(best_runs[model_id]["loss_history"]) for model_id in STRUCTURES)
    for model_id in STRUCTURES:
        row = best_runs[model_id]
        run_epochs = np.arange(1, len(row["loss_history"]) + 1)
        comparison_axes[0].plot(
            run_epochs,
            row["loss_history"],
            linewidth=2,
            label=f"{model_id} seed={row['seed']}",
        )
        comparison_axes[1].plot(
            run_epochs,
            np.array(row["train_acc_history"]) * 100.0,
            linewidth=2,
            label=f"{model_id} seed={row['seed']}",
        )

    comparison_axes[0].set_title("Best run from 5 - train loss")
    comparison_axes[0].set_xlabel("epoch")
    comparison_axes[0].set_ylabel("loss")
    comparison_axes[0].set_xlim(1, max_epochs)
    comparison_axes[0].grid(True, alpha=0.3)
    comparison_axes[0].legend()

    comparison_axes[1].set_title("Best run from 5 - train accuracy")
    comparison_axes[1].set_xlabel("epoch")
    comparison_axes[1].set_ylabel("accuracy [%]")
    comparison_axes[1].set_xlim(1, max_epochs)
    comparison_axes[1].grid(True, alpha=0.3)
    comparison_axes[1].legend()

    comparison_fig.suptitle("Comparison of best runs for M1, M2, M3")
    comparison_fig.tight_layout()
    comparison_fig.savefig(RESULTS_DIR / "best_runs_comparison.png", dpi=180, bbox_inches="tight")
    if not SHOW_PLOTS_IN_PYCHARM:
        plt.close(comparison_fig)
        comparison_fig = None

    best_structure_id = summary_df.sort_values(
        ["test_mean_acc_pct", "test_max_acc_pct"], ascending=False
    ).iloc[0]["model_id"]  # najlepsia struktura podla priemerneho test acc

    assert best_overall is not None
    best_cm = np.array(best_overall["confusion_matrix"])  # matica zamen najlepsieho modelu
    total = int(best_cm.sum())  # pocet test vzoriek
    metric_rows = []  # sem ulozime sensitivity a specificity
    for class_idx in range(best_cm.shape[0]):  # ratame metriky pre kazdu triedu
        tp = int(best_cm[class_idx, class_idx])  # spravne trafena trieda
        fn = int(best_cm[class_idx, :].sum() - tp)  # trieda bola, ale model ju minul
        fp = int(best_cm[:, class_idx].sum() - tp)  # model oznacil cudziu triedu ako tuto
        tn = int(total - tp - fn - fp)  # vsetko ostatne spravne zamietnute
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # citlivost triedy
        specificity = tn / (tn + fp) if (tn + fp) else 0.0  # specificita triedy
        metric_rows.append(
            {  # ulozime metriky danej triedy
                "class": CLASS_NAMES[class_idx],
                "sensitivity_pct": 100.0 * sensitivity,
                "specificity_pct": 100.0 * specificity,
            }
        )
    metrics_df = pd.DataFrame(metric_rows)  # tabulka citlivosti a specificity

    example_rows = []  # ukazkove predikcie po 1 z kazdej triedy
    best_y_true = best_overall["test_eval"]["y_true"]  # skutocne triedy na teste
    best_y_pred = best_overall["test_eval"]["y_pred"]  # predikovane triedy na teste
    best_probs = best_overall["test_eval"]["probs"]  # pravdepodobnosti tried
    best_test_indices = best_overall["test_indices"]  # indexy test vzoriek
    for class_idx in range(len(classes)):  # vezmeme prvy priklad z kazdej triedy
        pos = int(np.where(best_y_true == class_idx)[0][0])  # najdeme poziciu danej triedy
        # example_rows.append(
        #     {  # ulozime ukazkovu predikciu
        #         "dataset_index": int(best_test_indices[pos]),
        #         "true_class": CLASS_NAMES[class_idx],
        #         "predicted_class": CLASS_NAMES[int(best_y_pred[pos])],
        #         "prob_normalny_pct": 100.0 * float(best_probs[pos, 0]),
        #         "prob_podozrivy_pct": 100.0 * float(best_probs[pos, 1]),
        #         "prob_patologicky_pct": 100.0 * float(best_probs[pos, 2]),
        #     }
        # )
    examples_df = pd.DataFrame(example_rows)  # tabulka ukazkovych predikcii

    write_log(LOG_PATH, "")
    write_log(LOG_PATH, "=== SUMMARY BY STRUCTURE ===")
    write_log(LOG_PATH, summary_df.to_string(index=False))
    write_log(LOG_PATH, "")
    write_log(LOG_PATH, "=== BEST RUN FROM EACH STRUCTURE ===")
    write_log(LOG_PATH, best_runs_df.to_string(index=False))
    write_log(LOG_PATH, "")
    write_log(
        LOG_PATH,
        f"Best structure by mean test accuracy: {best_structure_id}",
    )
    write_log(
        LOG_PATH,
        f"Best trained network: {best_overall['model_id']} seed={best_overall['seed']} | test_acc={100.0 * best_overall['test_acc']:.3f}%",
    )
    write_log(LOG_PATH, "")
    write_log(LOG_PATH, "=== SENSITIVITY / SPECIFICITY FOR BEST TRAINED NETWORK ===")
    write_log(LOG_PATH, metrics_df.to_string(index=False))
    write_log(LOG_PATH, "")
    #write_log(LOG_PATH, "=== SAMPLE PREDICTIONS FOR BEST TRAINED NETWORK ===")
    #write_log(LOG_PATH, examples_df.to_string(index=False))

    if SHOW_PLOTS_IN_PYCHARM:  # otvorime finalne grafy aj v PyCharme
        write_log(
            LOG_PATH,
            "",
        )
        save_plots(
            RESULTS_DIR,
            str(best_overall["model_id"]),
            int(best_overall["seed"]),
            list(best_overall["loss_history"]),
            list(best_overall["train_acc_history"]),
            np.array(best_overall["confusion_matrix"]),
            show_in_pycharm=True,
        )
        if comparison_fig is not None:
            plt.close(comparison_fig)

    runs_df.to_csv(RESULTS_DIR / "all_runs.csv", index=False)  # ulozime vsetky behy
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)  # ulozime sumar struktur
    best_runs_df.to_csv(RESULTS_DIR / "best_runs.csv", index=False)  # ulozime najlepsie behy
    metrics_df.to_csv(RESULTS_DIR / "best_model_metrics.csv", index=False)  # ulozime citlivost a specificitu
    examples_df.to_csv(RESULTS_DIR / "best_model_examples.csv", index=False)  # ulozime ukazkove predikcie


if __name__ == "__main__":
    main()
