# Crossroad Traffic Control with a Sugeno Fuzzy Controller

A university coursework project that simulates traffic flow at a crossroad and compares fixed-time traffic lights with a Sugeno fuzzy logic controller.

## Features

- Six traffic scenarios
- Fixed-time traffic light controller
- Sugeno fuzzy controller
- Adjustable fixed green-light durations
- Per-lane and total waiting-car statistics
- CSV history files
- SVG comparison charts

## Controllers

### Fixed-Time Controller

The fixed controller uses predefined green-light durations for traffic-light phases A, B, and C.

Default intervals:

```text
13, 13, 13
```

### Fuzzy Controller

The fuzzy controller calculates the green-light duration from:

- the number of cars with a green light
- the number of cars with a red light

It uses triangular membership functions and produces green-light durations between 10 and 22 simulation steps.

## Requirements

- Python 3.10 or newer

The Python implementation uses only the standard library, so no external packages are required.

## Run

Run the default comparison for scenarios 2–6:

```bash
python main.py
```

Run only the fixed-time controller:

```bash
python main.py --controller fixed --mode 6
```

Run only the fuzzy controller:

```bash
python main.py --controller fuzzy --mode 6
```

Run a comparison for selected scenarios:

```bash
python main.py --controller compare --modes 2 3 4 5 6
```

Show traffic-light phase changes:

```bash
python main.py --controller fuzzy --mode 6 --trace-phases
```

## Results

The program saves output files in the `python_outputs/` directory:

```text
python_outputs/
├── summary.csv
├── summary_max_final.svg
├── mode2_fixed_history.csv
├── mode2_fuzzy_history.csv
├── mode2_total_compare.svg
└── ...
```

The CSV files contain the number of cars in each lane at every simulation step. The SVG files compare the total number of waiting cars for fixed and fuzzy controllers.

## Project Structure

```text
├── main.py
├── python_crossroad/
│   ├── data_loader.py
│   ├── fuzzy_controller.py
│   ├── reporting.py
│   ├── scenario_data.json
│   └── simulator.py
├── python_outputs/
└── README.md
```

## MATLAB Materials

The Python version does not require MATLAB or the MATLAB Fuzzy Logic Toolbox.


## Notes

This project was created for educational purposes as part of university coursework.
