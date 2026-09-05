from __future__ import annotations

import csv
from html import escape
from pathlib import Path

from .simulator import SimulationResult


def write_history_csv(path: Path, result: SimulationResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "A1", "A2", "A3", "B1", "B2", "C1", "C2", "total"])
        for step, row in enumerate(result.lane_history, start=1):
            writer.writerow([step, *row, result.total_history[step - 1]])


def write_summary_csv(path: Path, results_by_mode: dict[int, tuple[SimulationResult, SimulationResult]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mode",
                "controller",
                "max_total",
                "final_total",
                "mean_total",
                "max_lane",
                "final_lane",
            ]
        )
        for mode in sorted(results_by_mode):
            fixed_result, fuzzy_result = results_by_mode[mode]
            for result in (fixed_result, fuzzy_result):
                writer.writerow(
                    [
                        result.mode,
                        result.controller,
                        result.max_total,
                        result.final_total,
                        f"{result.mean_total:.3f}",
                        " ".join(str(value) for value in result.max_lane),
                        " ".join(str(value) for value in result.final_lane),
                    ]
                )


def _polyline_points(values: tuple[int, ...], width: int, height: int, x_pad: int, y_pad: int, max_value: int) -> str:
    usable_width = width - 2 * x_pad
    usable_height = height - 2 * y_pad
    points = []
    for index, value in enumerate(values):
        x = x_pad + usable_width * index / max(1, len(values) - 1)
        y = height - y_pad - usable_height * value / max_value
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_compare_svg(path: Path, fixed: SimulationResult, fuzzy: SimulationResult, title: str, fixed_label: str) -> None:
    width = 1200
    height = 520
    x_pad = 80
    y_pad = 55
    max_value = max(max(fixed.total_history), max(fuzzy.total_history), 1)
    grid_lines = []
    for level in range(0, max_value + 1, 5):
        y = height - y_pad - (height - 2 * y_pad) * level / max_value
        grid_lines.append(
            f"<line x1='{x_pad}' y1='{y:.2f}' x2='{width - x_pad}' y2='{y:.2f}' "
            "stroke='#d7d7d7' stroke-width='1' />"
        )
        grid_lines.append(f"<text x='20' y='{y + 5:.2f}' font-size='14' fill='#333'>{level}</text>")

    x_labels = []
    for step in range(0, len(fixed.total_history) + 1, 50):
        x = x_pad + (width - 2 * x_pad) * step / max(1, len(fixed.total_history) - 1)
        x_labels.append(
            f"<text x='{x:.2f}' y='{height - 15}' text-anchor='middle' font-size='14' fill='#333'>{step}</text>"
        )

    fixed_points = _polyline_points(fixed.total_history, width, height, x_pad, y_pad, max_value)
    fuzzy_points = _polyline_points(fuzzy.total_history, width, height, x_pad, y_pad, max_value)
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='{width}' height='{height}' fill='#fffdf8' />
<text x='{x_pad}' y='32' font-size='24' font-family='Arial' fill='#222'>{escape(title)}</text>
{''.join(grid_lines)}
<line x1='{x_pad}' y1='{height - y_pad}' x2='{width - x_pad}' y2='{height - y_pad}' stroke='#444' stroke-width='2' />
<line x1='{x_pad}' y1='{y_pad}' x2='{x_pad}' y2='{height - y_pad}' stroke='#444' stroke-width='2' />
{''.join(x_labels)}
<polyline fill='none' stroke='#b04632' stroke-width='4' points='{fixed_points}' />
<polyline fill='none' stroke='#206a5d' stroke-width='4' points='{fuzzy_points}' />
<rect x='{width - 270}' y='48' width='18' height='18' fill='#b04632' />
<text x='{width - 242}' y='63' font-size='16' fill='#222'>{escape(fixed_label)}</text>
<rect x='{width - 270}' y='78' width='18' height='18' fill='#206a5d' />
<text x='{width - 242}' y='93' font-size='16' fill='#222'>Fuzzy controller</text>
<text x='{width / 2:.2f}' y='{height - 12}' text-anchor='middle' font-size='16' fill='#333'>Scenario step</text>
<text x='24' y='28' font-size='16' fill='#333' transform='rotate(-90 24,28)'>Cars</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_summary_bars(
    path: Path,
    results_by_mode: dict[int, tuple[SimulationResult, SimulationResult]],
) -> None:
    width = 1100
    height = 520
    x_pad = 80
    y_pad = 60
    chart_height = height - 2 * y_pad
    chart_width = width - 2 * x_pad
    metrics = [
        (
            mode,
            results_by_mode[mode][0].max_total,
            results_by_mode[mode][0].final_total,
            results_by_mode[mode][1].max_total,
            results_by_mode[mode][1].final_total,
        )
        for mode in sorted(results_by_mode)
    ]
    max_value = max(max(row[1:]) for row in metrics)
    group_width = chart_width / len(metrics)
    bar_width = 34
    colors = ("#b04632", "#de7c5a", "#206a5d", "#4aa38f")
    labels = ("fixed max", "fixed final", "fuzzy max", "fuzzy final")

    bars = []
    mode_labels = []
    for index, row in enumerate(metrics):
        mode = row[0]
        values = row[1:]
        group_x = x_pad + index * group_width
        for offset, value in enumerate(values):
            x = group_x + 16 + offset * (bar_width + 8)
            bar_height = chart_height * value / max_value
            y = height - y_pad - bar_height
            bars.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width}' height='{bar_height:.2f}' fill='{colors[offset]}' />")
            bars.append(
                f"<text x='{x + bar_width / 2:.2f}' y='{y - 8:.2f}' text-anchor='middle' font-size='13' fill='#222'>{value}</text>"
            )
        mode_labels.append(
            f"<text x='{group_x + group_width / 2:.2f}' y='{height - 20}' text-anchor='middle' font-size='16' fill='#333'>Mode {mode}</text>"
        )

    legend = []
    for index, label in enumerate(labels):
        y = 46 + index * 24
        legend.append(f"<rect x='{width - 220}' y='{y}' width='16' height='16' fill='{colors[index]}' />")
        legend.append(f"<text x='{width - 196}' y='{y + 13}' font-size='15' fill='#222'>{label}</text>")

    grid = []
    for level in range(0, max_value + 1, 5):
        y = height - y_pad - chart_height * level / max_value
        grid.append(
            f"<line x1='{x_pad}' y1='{y:.2f}' x2='{width - x_pad}' y2='{y:.2f}' stroke='#e3e3e3' stroke-width='1' />"
        )
        grid.append(f"<text x='20' y='{y + 5:.2f}' font-size='14' fill='#333'>{level}</text>")

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
<rect width='{width}' height='{height}' fill='#fffdf8' />
<text x='{x_pad}' y='32' font-size='24' font-family='Arial' fill='#222'>Fixed vs fuzzy metrics by mode</text>
{''.join(grid)}
<line x1='{x_pad}' y1='{height - y_pad}' x2='{width - x_pad}' y2='{height - y_pad}' stroke='#444' stroke-width='2' />
<line x1='{x_pad}' y1='{y_pad}' x2='{x_pad}' y2='{height - y_pad}' stroke='#444' stroke-width='2' />
{''.join(bars)}
{''.join(mode_labels)}
{''.join(legend)}
<text x='{width / 2:.2f}' y='{height - 8}' text-anchor='middle' font-size='16' fill='#333'>Modes</text>
<text x='24' y='28' font-size='16' fill='#333' transform='rotate(-90 24,28)'>Cars</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def export_comparison_artifacts(
    output_dir: Path,
    results_by_mode: dict[int, tuple[SimulationResult, SimulationResult]],
    fixed_intervals: tuple[int, int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "summary.csv", results_by_mode)
    write_summary_bars(output_dir / "summary_max_final.svg", results_by_mode)
    fixed_label = f"Fixed {list(fixed_intervals)}"
    for mode in sorted(results_by_mode):
        fixed_result, fuzzy_result = results_by_mode[mode]
        write_history_csv(output_dir / f"mode{mode}_fixed_history.csv", fixed_result)
        write_history_csv(output_dir / f"mode{mode}_fuzzy_history.csv", fuzzy_result)
        write_compare_svg(
            output_dir / f"mode{mode}_total_compare.svg",
            fixed_result,
            fuzzy_result,
            title=f"Mode {mode}: total number of waiting cars",
            fixed_label=fixed_label,
        )
