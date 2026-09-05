# Genetic Algorithm Penalty Methods

A Python project that uses a genetic algorithm to solve a constrained profit-maximization problem.

The project compares three penalty methods for handling constraint violations:

- Dead penalty
- Step penalty
- Proportional penalty

## Problem

The algorithm searches for five decision variables:

```text
x1, x2, x3, x4, x5
```

The objective is to maximize profit:

```text
profit = 0.04x1 + 0.07x2 + 0.11x3 + 0.06x4 + 0.05x5
```

Subject to these constraints:

```text
x1 + x2 + x3 + x4 + x5 ≤ 10
x1 + x2 ≤ 2.5
x5 ≤ x4
x3 + x4 ≤ 0.5(x1 + x2 + x3 + x4 + x5)
```

## Method

Each experiment uses:

- Population size: 150
- Generations: 400
- Elite selection: 4 individuals
- Tournament selection
- Multi-point crossover
- Random mutation
- 10 independent runs per penalty method

The program prints the best solution, its profit, and constraint checks. It also displays convergence charts for each run and a final comparison chart.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib

Install the required packages:

```bash
python -m pip install numpy matplotlib
```

## Run

```bash
python main.py
```

## Project Structure

```text
.
|-- main.py          # Optimization problem, penalty methods, and experiment setup
`-- genetic_all.py   # Genetic algorithm operators and helper functions
```

## Notes

This is a university project focused on constrained optimization with genetic algorithms.
