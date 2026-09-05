# Genetic Algorithm for Route Optimization

A university project that uses a genetic algorithm to find a short route through a set of two-dimensional points.

The route starts at a fixed point, visits all intermediate points exactly once, and ends at a fixed destination. This is a Traveling Salesperson Problem (TSP) variant.

## Features

- Optimizes a route through 25 points
- Keeps the start and end points fixed
- Uses permutation-based chromosomes for the 23 intermediate points
- Calculates Euclidean distances with a precomputed distance matrix
- Uses elitism and tournament selection
- Uses Order Crossover (OX)
- Uses inversion mutation and swap mutation
- Runs the experiment 10 times
- Displays route-length convergence charts
- Visualizes the best route found

## Genetic Algorithm Settings

The default experiment uses:

- Population size: `120`
- Generations: `220`
- Elite individuals: `8`
- Crossover probability: `0.95`
- Inversion mutation probability: `0.30`
- Swap mutation probability: `0.12`
- Independent runs: `10`

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib

Install the dependencies:

```bash
python -m pip install numpy matplotlib
Run
python main.py
The program prints the best route length from every run, the overall best route, and the number of runs that reach a route length of 480 or less.
It also displays:
A chart comparing route-length progress across all runs
A chart with the best route through the points
Project Structure
.
|-- main.py          # TSP problem definition and genetic algorithm experiment
`-- genetic_all.py   # Course-provided genetic algorithm helper functions
Toolbox Note
genetic_all.py provides helper functions used by this project, including population generation, elite selection, tournament selection, and swap mutation.
Notes
This project was created for coursework in intelligent systems and genetic algorithms.
