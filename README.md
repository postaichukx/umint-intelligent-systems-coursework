# Intelligent Systems Coursework

Selected projects and experiment utilities from an Intelligent Systems course. The repository contains MATLAB and Python work on traffic-light control, genetic algorithms, and neural-network training analysis.

## Included Projects

### Traffic Intersection Simulation

`UI25_uloha_9/` contains a MATLAB simulation of a traffic intersection with seven lanes.

The simulation supports:

- Six traffic scenarios, including directional traffic peaks
- Fixed green-light intervals
- Custom green-light intervals
- Optional fuzzy-logic control
- Animated intersection visualization
- Graphs for lane occupancy and traffic-light states

Run `run_krizovatka.m` in MATLAB to configure and start the simulation.

### Genetic Algorithm Utilities

`zdroje/genetic_all.py` contains reusable building blocks for genetic algorithms:

- Schwefel-based objective function
- Uniform search-space generation
- Elitist, sorted, diversity-based, random, roulette-wheel, and tournament selection
- Mutation operators

### Neural-Network Training Plots

`UMINT_cv08_console_plots_20ep/` rebuilds training-history plots from a console log. The script reads loss and accuracy values from experiments and generates PNG charts for several model configurations.

## Requirements

### MATLAB Projects

- MATLAB
- Computer Vision Toolbox for the traffic visualization
- Fuzzy Logic Toolbox when fuzzy control is enabled
- The provided `scenarios.mat` and `crossroad.jpg` files

### Python Utilities

- Python 3.10 or newer
- NumPy for genetic-algorithm utilities
- Matplotlib for training-history plots

Install the Python packages:

```bash
python -m pip install numpy matplotlib
```

## Running the Traffic Simulation

1. Open MATLAB.
2. Change the current folder to `UI25_uloha_9`.
3. Open `run_krizovatka.m`.
4. Adjust the scenario, green-light intervals, fuzzy-control settings, and visualization flag.
5. Run the script.

## Rebuilding Training-History Plots

```bash
python UMINT_cv08_console_plots_20ep/rebuild_console_plots.py \
  --input ulohy/console_ocakavane_20ep.txt \
  --output-dir UMINT_cv08_console_plots_20ep
```

## Project Structure

```text
.
|-- UI25_uloha_9/                     # MATLAB traffic intersection simulation
|-- UMINT_cv08_console_plots_20ep/    # Training-log plot reconstruction
|-- zdroje/
|   `-- genetic_all.py                # Genetic algorithm utilities
|-- ulohy/                            # Assignment materials and experiment logs
`-- prednasky/                        # Course reference materials
```

## Notes

This repository documents university coursework and learning experiments. Course documents and third-party materials should only be included when their redistribution is permitted.
