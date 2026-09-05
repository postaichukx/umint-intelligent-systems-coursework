import numpy as np
import matplotlib.pyplot as plt
import genetic_all as ga


def profit(population):
    pop = np.atleast_2d(population)
    return (
        0.04 * pop[:, 0] +
        0.07 * pop[:, 1] +
        0.11 * pop[:, 2] +
        0.06 * pop[:, 3] +
        0.05 * pop[:, 4]
    )


def constraint_violations(population):
    pop = np.atleast_2d(population)
    x1, x2, x3, x4, x5 = pop.T

    v1 = np.maximum(0, x1 + x2 + x3 + x4 + x5 - 10)
    v2 = np.maximum(0, x1 + x2 - 2.5)
    v3 = np.maximum(0, x5 - x4)
    v4 = np.maximum(0, x3 + x4 - 0.5 * (x1 + x2 + x3 + x4 + x5))

    return v1, v2, v3, v4


def fitness_dead(population):
    pop = np.atleast_2d(population)
    J = profit(pop)

    v1, v2, v3, v4 = constraint_violations(pop)
    violation = (v1 > 0) | (v2 > 0) | (v3 > 0) | (v4 > 0)

    return np.where(violation, 100.0, -J)


def fitness_step(population):
    pop = np.atleast_2d(population)
    J = profit(pop)

    v1, v2, v3, v4 = constraint_violations(pop)

    violations_count = (
        (v1 > 0).astype(int) +
        (v2 > 0).astype(int) +
        (v3 > 0).astype(int) +
        (v4 > 0).astype(int)
    )

    penalty = violations_count * 20.0
    return -J + penalty


def fitness_prop(population):
    pop = np.atleast_2d(population)
    J = profit(pop)

    v1, v2, v3, v4 = constraint_violations(pop)
    penalty = 10.0 * (v1 + v2 + v3 + v4)

    return -J + penalty


def print_constraints(x):
    x = np.atleast_2d(x)
    v1, v2, v3, v4 = constraint_violations(x)

    print("constraint check:")
    print("v1 =", round(float(v1[0]), 6))
    print("v2 =", round(float(v2[0]), 6))
    print("v3 =", round(float(v3[0]), 6))
    print("v4 =", round(float(v4[0]), 6))

    feasible = (v1[0] == 0 and v2[0] == 0 and v3[0] == 0 and v4[0] == 0)
    print("feasible =", feasible)


def format_vector(x):
    return np.round(np.asarray(x, dtype=float), 6)


def run(fitness_fn, space, population_size, generations, elite_size, crossover_points, mutation_rate):
    population = ga.genrpop(population_size, space)

    evolution = []
    best_y = float("inf")
    best_x = None

    for _ in range(generations):
        fitness = fitness_fn(population)

        best_index = np.argmin(fitness)
        current_best_y = fitness[best_index]

        if current_best_y < best_y:
            best_y = float(current_best_y)
            best_x = population[best_index].copy()

        evolution.append(best_y)

        elite_population, _ = ga.selsort(
            population,
            fitness,
            elite_size,
            reverse=False
        )

        parents, _ = ga.seltourn(
            population,
            fitness,
            population_size - elite_size,
            reverse=False
        )

        children = ga.crossov(parents.copy(), pts=crossover_points, mode=0)
        children = ga.mutx(children, rate=mutation_rate, space=space)

        population = np.vstack([elite_population, children])

    return best_x, best_y, evolution


def multi_run(title, fitness_fn, space, population_size, generations,
              elite_size, crossover_points, mutation_rate, runs):
    all_evolutions = []
    global_best_y = float("inf")
    global_best_x = None
    global_best_evo = None

    for i in range(runs):
        x, y, evo = run(
            fitness_fn,
            space,
            population_size,
            generations,
            elite_size,
            crossover_points,
            mutation_rate
        )

        all_evolutions.append(evo)
        print(f"[{title}] run {i + 1}: best_y = {round(y, 6)}")

        if y < global_best_y:
            global_best_y = y
            global_best_x = x
            global_best_evo = evo

    print(f"\n[{title}] GLOBAL BEST")
    print("best_y =", round(global_best_y, 6))
    print("best_x =", format_vector(global_best_x))

    best_profit_eur = int(round(profit(global_best_x)[0] * 1_000_000))
    print("profit =", best_profit_eur, "EUR")

    print_constraints(global_best_x)

    plt.figure(figsize=(10, 6))
    for i, evo in enumerate(all_evolutions):
        plt.plot(evo, label=f"run {i + 1} ({evo[-1]:.4f})")

    plt.xlabel("generation")
    plt.ylabel("best fitness so far")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return global_best_x, global_best_y, global_best_evo


def main():
    space = ga.uniform_space(5, 0, 10)

    population_size = 150
    generations = 400
    elite_size = 4
    crossover_points = 3
    mutation_rate = 0.07
    runs = 10

    dead_x, dead_y, dead_evo = multi_run(
        "Dead penalty",
        fitness_dead,
        space,
        population_size,
        generations,
        elite_size,
        crossover_points,
        mutation_rate,
        runs
    )

    step_x, step_y, step_evo = multi_run(
        "Step penalty",
        fitness_step,
        space,
        population_size,
        generations,
        elite_size,
        crossover_points,
        mutation_rate,
        runs
    )

    prop_x, prop_y, prop_evo = multi_run(
        "Proportional penalty",
        fitness_prop,
        space,
        population_size,
        generations,
        elite_size,
        crossover_points,
        mutation_rate,
        runs
    )

    plt.figure(figsize=(10, 6))
    plt.plot(dead_evo, label=f"Dead ({dead_y:.4f})")
    plt.plot(step_evo, label=f"Step ({step_y:.4f})")
    plt.plot(prop_evo, label=f"Proportional ({prop_y:.4f})")

    plt.xlabel("generation")
    plt.ylabel("best fitness so far")
    plt.title("Best runs comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()