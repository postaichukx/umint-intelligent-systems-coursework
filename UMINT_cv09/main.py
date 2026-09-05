from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from python_crossroad.data_loader import load_scenario_data
from python_crossroad.fuzzy_controller import SugenoFuzzyController
from python_crossroad.reporting import export_comparison_artifacts
from python_crossroad.simulator import (
    DEFAULT_FIXED_INTERVALS,
    LANE_NAMES,
    SimulationResult,
    simulate,
)


DEFAULT_COMPARE_MODES = (2, 3, 4, 5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Python rewrite of the crossroad-control assignment from the provided MATLAB materials."
    )
    parser.add_argument(
        "--controller",
        choices=("compare", "fixed", "fuzzy"),
        default="compare",
        help="`compare` runs both controllers and exports comparison artifacts.",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=6,
        help="Scenario mode for single-controller runs.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=list(DEFAULT_COMPARE_MODES),
        help="Modes used when controller=compare.",
    )
    parser.add_argument(
        "--fixed-intervals",
        type=int,
        nargs=3,
        metavar=("A", "B", "C"),
        default=list(DEFAULT_FIXED_INTERVALS),
        help="Green durations for the fixed controller.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python_outputs"),
        help="Directory for CSV/SVG outputs.",
    )
    parser.add_argument(
        "--trace-phases",
        action="store_true",
        help="Print every phase change for the executed simulation.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Do not write CSV/SVG files.",
    )
    return parser.parse_args()


def validate_modes(modes: Iterable[int]) -> list[int]:
    normalized = []
    for mode in modes:
        if mode < 1 or mode > 6:
            raise ValueError(f"Mode must be between 1 and 6, got {mode}.")
        normalized.append(mode)
    return normalized


def format_result(result: SimulationResult) -> str:
    max_lane = ", ".join(f"{name}={value}" for name, value in zip(LANE_NAMES, result.max_lane))
    final_lane = ", ".join(f"{name}={value}" for name, value in zip(LANE_NAMES, result.final_lane))
    return (
        f"Mode {result.mode} | {result.controller}\n"
        f"  max_total={result.max_total}, final_total={result.final_total}, mean_total={result.mean_total:.3f}\n"
        f"  max_lane:   {max_lane}\n"
        f"  final_lane: {final_lane}"
    )


def print_phase_trace(result: SimulationResult) -> None:
    for phase in result.phase_log:
        print(
            "  "
            f"step={phase.cycle:>3} "
            f"phase={phase.configuration} "
            f"green={phase.cars_green:>2} "
            f"red={phase.cars_red:>2} "
            f"duration={phase.green_duration:>5.2f} "
            f"cars={list(phase.cars_snapshot)}"
        )


def run_single(args: argparse.Namespace) -> int:
    init_cars, incom_mtx = load_scenario_data()
    controller = args.controller
    fixed_intervals = tuple(args.fixed_intervals)
    fuzzy_controller = SugenoFuzzyController()
    result = simulate(
        mode=args.mode,
        controller=controller,
        init_cars=init_cars,
        incom_mtx=incom_mtx,
        fixed_intervals=fixed_intervals,
        fuzzy_controller=fuzzy_controller,
    )
    print(format_result(result))
    if args.trace_phases:
        print("Phase trace:")
        print_phase_trace(result)
    return 0


def run_compare(args: argparse.Namespace) -> int:
    modes = validate_modes(args.modes)
    init_cars, incom_mtx = load_scenario_data()
    fixed_intervals = tuple(args.fixed_intervals)
    fuzzy_controller = SugenoFuzzyController()

    results_by_mode: dict[int, tuple[SimulationResult, SimulationResult]] = {}
    for mode in modes:
        fixed_result = simulate(
            mode=mode,
            controller="fixed",
            init_cars=init_cars,
            incom_mtx=incom_mtx,
            fixed_intervals=fixed_intervals,
            fuzzy_controller=fuzzy_controller,
        )
        fuzzy_result = simulate(
            mode=mode,
            controller="fuzzy",
            init_cars=init_cars,
            incom_mtx=incom_mtx,
            fixed_intervals=fixed_intervals,
            fuzzy_controller=fuzzy_controller,
        )
        results_by_mode[mode] = (fixed_result, fuzzy_result)

    for mode in modes:
        fixed_result, fuzzy_result = results_by_mode[mode]
        print(format_result(fixed_result))
        print(format_result(fuzzy_result))
        print()

    if args.trace_phases and len(modes) == 1:
        print("Fixed phase trace:")
        print_phase_trace(results_by_mode[modes[0]][0])
        print("Fuzzy phase trace:")
        print_phase_trace(results_by_mode[modes[0]][1])

    if not args.skip_export:
        export_comparison_artifacts(
            output_dir=args.output_dir,
            results_by_mode=results_by_mode,
            fixed_intervals=fixed_intervals,
        )
        print(f"Artifacts written to: {args.output_dir.resolve()}")
    return 0


def main() -> int:
    args = parse_args()
    validate_modes([args.mode])
    if args.controller == "compare":
        return run_compare(args)
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
