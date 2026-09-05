from __future__ import annotations

from dataclasses import dataclass


def trimf(x: float, a: float, b: float, c: float) -> float:
    if a == b and x == a:
        return 1.0
    if b == c and x == c:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


@dataclass(frozen=True)
class SugenoFuzzyController:
    green_memberships: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 9.0),
        (3.0, 11.0, 19.0),
        (13.0, 23.0, 23.0),
    )
    red_memberships: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 9.0),
        (3.0, 11.0, 19.0),
        (13.0, 23.0, 23.0),
    )
    rule_outputs: tuple[tuple[float, float, float], ...] = (
        (16.0, 10.0, 10.0),
        (22.0, 19.0, 13.0),
        (22.0, 22.0, 16.0),
    )
    default_duration: float = 16.0
    min_duration: float = 10.0
    max_duration: float = 22.0

    def evaluate(self, cars_green: int, cars_red: int) -> float:
        numerator = 0.0
        denominator = 0.0
        for green_index, green_mf in enumerate(self.green_memberships):
            green_value = trimf(cars_green, *green_mf)
            if green_value <= 0:
                continue
            for red_index, red_mf in enumerate(self.red_memberships):
                red_value = trimf(cars_red, *red_mf)
                if red_value <= 0:
                    continue
                weight = min(green_value, red_value)
                numerator += weight * self.rule_outputs[green_index][red_index]
                denominator += weight
        if denominator == 0:
            return self.default_duration
        return max(self.min_duration, min(self.max_duration, numerator / denominator))
