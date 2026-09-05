from __future__ import annotations

import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_PATH = PACKAGE_ROOT / "scenario_data.json"


def load_scenario_data() -> tuple[list[list[int]], list[list[list[int]]]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    init_cars = data["init_cars"]
    incom_mtx = data["incom_mtx"]
    return init_cars, incom_mtx
