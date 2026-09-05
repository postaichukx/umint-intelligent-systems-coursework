### Contains a revised, fixed and extended Python implementation of Genetic-toolbox [1]


# Genetic Algorithm Operators Library

This project contains a collection of Python functions for implementing various components of genetic algorithms. It includes several selection, mutation, and crossover operators along with helper utilities for generating populations and testing fitness functions. The code is designed to be modular and flexible, allowing you to experiment with different evolutionary strategies for optimization problems.

## Installation
You can install this package using pip directly from this repository:
```bash
pip install git+https://github.com/swrko/Genetic-toolbox
```

## Features

### Selection Operators
- **Elitist Selection**
  - `selbest`: Creates a new population based on an ordered list of individuals using a custom count (`n_list`).
  - `selsort`: Returns the top *n* individuals sorted by fitness.
- **Diversity-based Selection**
  - `seldiv`: Selects individuals based on their diversity relative to a reference individual (either the best or the mean).
- **Random Selection**
  - `selrand`: Chooses individuals randomly from the population.
- **Roulette Wheel Selection**
  - `selsus`: Implements a weighted roulette wheel (sus) selection strategy.
- **Tournament Selection**
  - `seltourn`: Performs selection by pitting random pairs against each other and choosing the better individual.

### Mutation Operators
- **Simple Mutation**
  - `mutx`: Randomly alters genes to values within defined bounds.
- **Additive Mutation**
  - `muta`: Adds a random value (scaled by an amplitude parameter) to selected genes.
- **Multiplicative Mutation**
  - `mutm`: Multiplies selected genes by a random factor within a given range.
- **Permutation Mutations**
  - `swapgen`: Swaps two randomly chosen genes within an individual.
  - `swappart`: Splits an individual at a random index and swaps the two parts.

### Crossover Operators
- **Intermediate Crossover**
  - `intmedx`: Combines pairs of individuals by adjusting their genes based on an alpha scaling factor.
- **Group-based Crossover**
  - `crossgrp`: Creates new individuals by randomly selecting genes from across the entire population.
- **Multi-point Crossover**
  - `crossov`: Splits parent individuals at specified points and exchanges segments between them.

### Additional Utilities
- **Population Generation**
  - `genrpop`: Generates a new population with genes initialized within specified bounds.
  - `genrpop_perm`: Generates a population suited for permutation problems.
- **Fitness Functions for Testing**
  - `schwefel`, `eggholder`, and `rastrigin`: Common benchmark functions for optimization.
- **Other Helpers**
  - `invfit`: Inverts fitness values for maximization problems.
  - `migrate`, `warming`, `reset`: Additional functions to support migration between subpopulations, noise application, and population reinitialization.
- **Utility Functions**
  - `output`: Prints matrices in a formatted style.
  - `rate_check`, `pop_check`, `mut_check`, and `splitting`: Internal functions to validate parameters and assist with genetic operations.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib (examples)

---

[1] SEKAJ, Ivan a Martin FOLTIN. MATLAB Toolbox - genetické algoritmy. MATLAB 2003: Sborník příspěvků 11.ročníku konference. Díl I. Praha: ČVUT, 2003, , 514-519. dostupne online - http://dsp.vscht.cz/konference_matlab/matlab03/sekaj.pdf

[2] Sekaj, I.: The Use of Matlab Parallel Computing Toolbox for Genetic Algorithm-Based MIMO Controller Design. Editors: Fikar, M., Kvasnica, M., In Proceedings of the 17th International Conference on Process Control ’09, Štrbské Pleso, Slovakia, 277–280, 2009
