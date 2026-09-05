from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "umint_matplotlib"))

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_RE = re.compile(r"^=== (M\d+) \| ([^|]+) \| (scratch|tl) ===$")
AUG_GROUP_RE = re.compile(r"^=== AUGMENTED \| (M\d+) \| ([^|]+) \| (scratch|tl) ===$")
EPOCH_RE = re.compile(
    r"^(M\d+) (.+?) \| (scratch|tl) \| seed=(\d+) \| epoch (\d+)/(\d+) \| "
    r"train_loss=([0-9.]+) \(CE\) \| train_acc=([0-9.]+)% \| "
    r"val_loss=([0-9.]+) \(CE\) \| val_acc=([0-9.]+)%$"
)
AUG_EPOCH_RE = re.compile(
    r"^(M\d+) (.+?) \| (scratch|tl) \| seed=(\d+) \| augmented \| epoch (\d+)/(\d+) \| "
    r"train_loss=([0-9.]+) \(CE\) \| train_acc=([0-9.]+)% \| "
    r"val_loss=([0-9.]+) \(CE\) \| val_acc=([0-9.]+)%$"
)
FINAL_RE = re.compile(
    r"^final \| seed=(\d+) \| train_acc=([0-9.]+)% \| val_acc=([0-9.]+)% \| test_acc=([0-9.]+)%$"
)
AUG_FINAL_RE = re.compile(
    r"^aug final \| seed=(\d+) \| train_acc=([0-9.]+)% \| val_acc=([0-9.]+)% \| test_acc=([0-9.]+)%$"
)


@dataclass
class RunHistory:
    seed: int
    epochs_total: int = 0
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    final_train_acc: float | None = None
    final_val_acc: float | None = None
    final_test_acc: float | None = None


@dataclass
class PlotGroup:
    model_id: str
    arch: str
    mode: str
    augmented: bool
    runs: dict[int, RunHistory] = field(default_factory=dict)

    @property
    def plot_name(self) -> str:
        name = f"{self.model_id}_{self.mode}"
        if self.augmented:
            name += "_augmented"
        return name


def make_group_key(model_id: str, mode: str, augmented: bool) -> str:
    return f"{model_id}|{mode}|{int(augmented)}"


def parse_console_log(log_path: Path) -> list[PlotGroup]:
    groups: dict[str, PlotGroup] = {}
    order: list[str] = []
    current_group_key: str | None = None

    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = GROUP_RE.match(line)
        if match:
            model_id, arch, mode = match.groups()
            current_group_key = make_group_key(model_id, mode, False)
            if current_group_key not in groups:
                groups[current_group_key] = PlotGroup(model_id=model_id, arch=arch, mode=mode, augmented=False)
                order.append(current_group_key)
            continue

        match = AUG_GROUP_RE.match(line)
        if match:
            model_id, arch, mode = match.groups()
            current_group_key = make_group_key(model_id, mode, True)
            if current_group_key not in groups:
                groups[current_group_key] = PlotGroup(model_id=model_id, arch=arch, mode=mode, augmented=True)
                order.append(current_group_key)
            continue

        match = EPOCH_RE.match(line)
        augmented = False
        if match is None:
            match = AUG_EPOCH_RE.match(line)
            augmented = match is not None

        if match is not None:
            model_id, arch, mode, seed_text, epoch_text, epochs_total_text, train_loss, train_acc, val_loss, val_acc = match.groups()
            group_key = make_group_key(model_id, mode, augmented)
            if group_key not in groups:
                groups[group_key] = PlotGroup(model_id=model_id, arch=arch, mode=mode, augmented=augmented)
                order.append(group_key)

            seed = int(seed_text)
            run = groups[group_key].runs.setdefault(seed, RunHistory(seed=seed))
            run.epochs_total = int(epochs_total_text)
            run.epochs.append(int(epoch_text))
            run.train_loss.append(float(train_loss))
            run.train_acc.append(float(train_acc))
            run.val_loss.append(float(val_loss))
            run.val_acc.append(float(val_acc))
            current_group_key = group_key
            continue

        match = FINAL_RE.match(line)
        if match and current_group_key is not None:
            seed_text, train_acc, val_acc, test_acc = match.groups()
            run = groups[current_group_key].runs[int(seed_text)]
            run.final_train_acc = float(train_acc)
            run.final_val_acc = float(val_acc)
            run.final_test_acc = float(test_acc)
            continue

        match = AUG_FINAL_RE.match(line)
        if match and current_group_key is not None:
            seed_text, train_acc, val_acc, test_acc = match.groups()
            run = groups[current_group_key].runs[int(seed_text)]
            run.final_train_acc = float(train_acc)
            run.final_val_acc = float(val_acc)
            run.final_test_acc = float(test_acc)
            continue

    return [groups[key] for key in order]


def select_best_run(group: PlotGroup) -> RunHistory:
    def score(run: RunHistory) -> tuple[float, float]:
        final_val_acc = run.final_val_acc if run.final_val_acc is not None else run.val_acc[-1]
        final_val_loss = run.val_loss[-1]
        return final_val_acc, -final_val_loss

    return max(group.runs.values(), key=score)


def save_history_plot(output_dir: Path, group: PlotGroup, run: RunHistory) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(run.epochs, run.train_loss, label="train", color="tab:blue")
    axes[0].plot(run.epochs, run.val_loss, label="val", color="tab:orange")
    axes[0].set_title(f"{group.plot_name} - loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (CE)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(run.epochs, run.train_acc, label="train", color="tab:green")
    axes[1].plot(run.epochs, run.val_acc, label="val", color="tab:red")
    axes[1].set_title(f"{group.plot_name} - accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy [%]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    test_acc_text = f"{run.final_test_acc:.2f}%" if run.final_test_acc is not None else "n/a"
    fig.suptitle(
        f"{group.model_id} | {group.arch} | {group.mode}"
        + (" | augmented" if group.augmented else "")
        + f" | best seed={run.seed} | test_acc={test_acc_text}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{group.plot_name}_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_readme(output_dir: Path, groups: list[PlotGroup]) -> None:
    lines = [
        "Recreated matplotlib history plots from console_ocakavane_20ep.txt",
        "",
        "Generated files:",
    ]
    for group in groups:
        lines.append(f"- {group.plot_name}_history.png")
    lines.extend(
        [
            "",
            "Note:",
            "The original project code can also draw prediction plots,",
            "but those cannot be reconstructed from the console log alone",
            "because the log does not contain image-level predictions or source images.",
        ]
    )
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to console log txt file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for recreated plot PNG files.")
    args = parser.parse_args()

    groups = parse_console_log(args.input)
    if not groups:
        raise ValueError("No plot groups found in console log.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        best_run = select_best_run(group)
        save_history_plot(args.output_dir, group, best_run)

    save_readme(args.output_dir, groups)

    print(f"created {len(groups)} history plots in {args.output_dir}")


if __name__ == "__main__":
    main()
