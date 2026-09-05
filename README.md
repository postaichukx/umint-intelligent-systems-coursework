# Genetic Algorithm for Constrained Investment Optimization

A university project that uses a genetic algorithm to optimize the allocation of capital across five investment categories.

The objective is to maximize expected profit while respecting capital, stock, portfolio-balance, and non-negativity constraints.

## Objective Function

The expected profit is calculated as:

```text
profit = 0.04x1 + 0.07x2 + 0.11x3 + 0.06x4 + 0.05x5
The total available capital is:
10,000,000
Constraints
x1 + x2 + x3 + x4 + x5 ≤ 10,000,000
x1 + x2 ≤ 2,500,000
x5 ≤ x4
x3 + x4 ≤ 0.5(x1 + x2 + x3 + x4 + x5)
xi ≥ 0
Features
Constrained investment-allocation optimization
Three penalty methods:Dead penalty
Step penalty
Proportional penalty

Elite selection and tournament selection
Intermediate recombination crossover
Additive mutation
Feasible-population initialization
Random immigrant generation after 60 generations without improvement
Five independent runs for every penalty method
Fitness convergence charts
Comparison of the best result from each penalty method
Genetic Algorithm Settings
The default experiment uses:
Population size: 300
Generations: 600
Elite individuals: 6
Crossover alpha: 0.8
Mutation rate: 0.1
Mutation amplitude: 200,000
Independent runs per method: 5
Requirements
Python 3.10 or newer
NumPy
Matplotlib
Install the dependencies:
python -m pip install numpy matplotlib
Run
python main.py
The program prints the best capital allocation, fitness value, and expected profit for each penalty method. It also displays convergence charts for all runs and a chart comparing the best runs.
Project Structure
.
|-- main.py          # Optimization problem, constraints, and GA experiment
`-- genetic_all.py   # Course-provided genetic algorithm helper functions
Toolbox Note
genetic_all.py provides helper functions for population generation, selection, intermediate crossover, and additive mutation.
Notes
This project was created for coursework in intelligent systems and genetic algorithms. The investment values are part of an academic optimization exercise and do not represent financial advice.
