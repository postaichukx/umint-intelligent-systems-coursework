from __future__ import annotations

from dataclasses import dataclass

from .fuzzy_controller import SugenoFuzzyController


LANE_NAMES = ("A1", "A2", "A3", "B1", "B2", "C1", "C2")
PHASE_NAMES = ("A", "B", "C")
LIGHTS_SET = (
    (1, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 1, 0, 1),
    (0, 0, 0, 0, 1, 1, 1),
)
DEFAULT_FIXED_INTERVALS = (13, 13, 13)


@dataclass(frozen=True)
class PhaseEvent:
    cycle: int
    configuration: str
    cars_green: int
    cars_red: int
    green_duration: float
    cars_snapshot: tuple[int, ...]


@dataclass(frozen=True)
class SimulationResult:
    mode: int
    controller: str
    lane_history: tuple[tuple[int, ...], ...]
    total_history: tuple[int, ...]
    max_lane: tuple[int, ...]
    final_lane: tuple[int, ...]
    max_total: int
    final_total: int
    mean_total: float
    phase_log: tuple[PhaseEvent, ...]


def _history_max_per_lane(history: list[tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(max(row[index] for row in history) for index in range(len(LANE_NAMES)))


def simulate(
    mode: int,
    controller: str,
    init_cars: list[list[int]],
    incom_mtx: list[list[list[int]]],
    fixed_intervals: tuple[int, int, int] = DEFAULT_FIXED_INTERVALS,
    fuzzy_controller: SugenoFuzzyController | None = None,
) -> SimulationResult:
    if mode < 1 or mode > 6:
        raise ValueError(f"Mode must be between 1 and 6, got {mode}.")
    if controller not in {"fixed", "fuzzy"}:
        raise ValueError(f"Controller must be `fixed` or `fuzzy`, got {controller}.")

    fuzzy_controller = fuzzy_controller or SugenoFuzzyController()
    sim_length = 500 if mode == 6 else 150
    cars = [int(init_cars[lane][mode - 1]) for lane in range(len(LANE_NAMES))]
    leaving = [10] * len(LANE_NAMES)
    green_duration = 0.0
    set_lights = 0
    lights = [0] * len(LANE_NAMES)
    lane_history: list[tuple[int, ...]] = []
    total_history: list[int] = []
    phase_log: list[PhaseEvent] = []

    for cycle in range(sim_length):
        for lane in range(len(LANE_NAMES)):
            cars[lane] += int(incom_mtx[cycle][lane][mode - 1])

        for lane in range(len(LANE_NAMES)):
            if lights[lane] and cars[lane] > 0:
                leaving[lane] -= 1
                if leaving[lane] < 1:
                    cars[lane] -= 1
                    leaving[lane] = 3
            else:
                leaving[lane] = 10

        if green_duration < 1:
            setter = set_lights % len(LIGHTS_SET)
            lights = list(LIGHTS_SET[setter])
            cars_green = sum(light * car for light, car in zip(lights, cars))
            cars_red = sum(cars) - cars_green
            if controller == "fixed":
                green_duration = float(fixed_intervals[setter])
            else:
                green_duration = fuzzy_controller.evaluate(cars_green, cars_red)
            phase_log.append(
                PhaseEvent(
                    cycle=cycle + 1,
                    configuration=PHASE_NAMES[setter],
                    cars_green=cars_green,
                    cars_red=cars_red,
                    green_duration=green_duration,
                    cars_snapshot=tuple(cars),
                )
            )
            set_lights += 1

        green_duration -= 1
        lane_history.append(tuple(cars))
        total_history.append(sum(cars))

    final_lane = lane_history[-1]
    max_lane = _history_max_per_lane(lane_history)
    max_total = max(total_history)
    final_total = total_history[-1]
    mean_total = sum(total_history) / len(total_history)
    return SimulationResult(
        mode=mode,
        controller=controller,
        lane_history=tuple(lane_history),
        total_history=tuple(total_history),
        max_lane=max_lane,
        final_lane=final_lane,
        max_total=max_total,
        final_total=final_total,
        mean_total=mean_total,
        phase_log=tuple(phase_log),
    )
