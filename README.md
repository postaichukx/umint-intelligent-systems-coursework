# Genetic Algorithm for TSP with Mutation Strategies

A university project that uses a genetic algorithm to solve a Traveling Salesperson Problem (TSP) variant.

The algorithm searches for the shortest route that starts at a fixed point, visits 23 intermediate points exactly once, and returns to the final fixed point.

## Features

- Route optimization for 25 two-dimensional points
- Fixed start and end point
- Permutation-based chromosome representation
- Precomputed Euclidean distance matrix
- Elite selection and tournament selection
- Multiple mutation strategies:
  - Swap mutation
  - Segment-swap mutation
  - Inversion mutation
  - Combined mutation strategy
- Candidate-pool selection for the next generation
- Random immigrant generation after 75 generations without improvement
- Ten independent runs
- Convergence chart and best-route visualization

## Genetic Algorithm Settings

The default experiment uses:

- Population size: `450`
- Generations: `600`
- Elite individuals: `15`
- Swap mutation rate: `0.03`
- Segment-swap mutation rate: `0.05`
- Inversion mutation rate: `0.24`
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
The program prints the best route length from each run and the overall best route.
It also displays:
A chart showing route-length progress across all runs
The average convergence curve
A visualization of the shortest route found
Project Structure
.
|-- main.py          # TSP problem, genetic algorithm, and visualizations
`-- genetic_all.py   # Course-provided genetic algorithm helper functions
Toolbox Note
genetic_all.py provides helper functions for population generation, sorting, tournament selection, swap mutation, and segment-swap mutation.
Notes
This project was created for coursework in intelligent systems and genetic algorithms.
