import random
import numpy as np
import matplotlib.pyplot as plt

from genetic_all import genrpop, uniform_space, selsort, seltourn, intmedx, muta

RATES = np.array([0.04, 0.07, 0.11, 0.06, 0.05], dtype=float)
CAPITAL = 10_000_000.0
STOCK_LIMIT = 2_500_000.0

SPACE = uniform_space(5, 0.0, CAPITAL)

DEAD_PENALTY = 10_000_000.0
STEP_PENALTY = 1_000_000.0
PROP_C = 1.0


def profit(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.dot(RATES, x))


def violations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = float(np.sum(x))

    return np.array([
        max(0.0, s - CAPITAL),               # x1+x2+x3+x4+x5 <= 10 000 000
        max(0.0, x[0] + x[1] - STOCK_LIMIT), # x1+x2 <= 2 500 000
        max(0.0, x[4] - x[3]),               # x4 >= x5
        max(0.0, x[2] + x[3] - 0.5 * s),     # x3+x4 <= 0.5*sum(x)
        float(np.sum(np.maximum(0.0, -x)))   # xi >= 0
    ], dtype=float)


def fit_dead(pop: np.ndarray) -> np.ndarray:
    """Mrtva pokuta."""
    pop = np.asarray(pop, dtype=float)
    if pop.ndim == 1:
        pop = pop[None, :]

    fit = np.zeros(pop.shape[0], dtype=float)
    for i, x in enumerate(pop):
        fit[i] = -profit(x) if np.sum(violations(x)) <= 1e-9 else DEAD_PENALTY
    return fit


def fit_step(pop: np.ndarray) -> np.ndarray:
    pop = np.asarray(pop, dtype=float)
    if pop.ndim == 1:
        pop = pop[None, :]

    fit = np.zeros(pop.shape[0], dtype=float)
    for i, x in enumerate(pop):
        fit[i] = -profit(x) + STEP_PENALTY * np.sum(violations(x) > 1e-9)
    return fit


def fit_prop(pop: np.ndarray) -> np.ndarray:
    pop = np.asarray(pop, dtype=float)
    if pop.ndim == 1:
        pop = pop[None, :]

    fit = np.zeros(pop.shape[0], dtype=float)
    for i, x in enumerate(pop):
        fit[i] = -profit(x) + PROP_C * np.sum(violations(x))
    return fit


def init_population(pop_size: int) -> np.ndarray:
    pop = genrpop(pop_size, SPACE)

    for i in range(pop_size):
        x = pop[i].copy()
        x = np.maximum(x, 0.0)

        s = np.sum(x)
        if s > CAPITAL and s > 0:
            x *= CAPITAL / s

        stocks = x[0] + x[1]
        if stocks > STOCK_LIMIT and stocks > 0:
            k = STOCK_LIMIT / stocks
            x[0] *= k
            x[1] *= k

        if x[4] > x[3]:
            x[4] = x[3]

        s = np.sum(x)
        bonds = x[2] + x[3]
        limit = 0.5 * s
        if bonds > limit and bonds > 0:
            k = limit / bonds
            x[2] *= k
            x[3] *= k

        pop[i] = x

    return pop


def ga_investment(
    fitfun,
    pop_size: int,
    generations: int,
    elite_count: int,
    alpha: float,
    mut_rate: float,
    mut_amp: np.ndarray
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Jeden GA beh pre jeden typ pokuty."""
    pop = init_population(pop_size)

    best_hist = np.zeros(generations, dtype=float)
    best_global = pop[0].copy()
    best_global_fit = float("inf")
    stagnation = 0
    immigrant_count = max(40, pop_size // 8)

    for generation in range(generations):
        fit = fitfun(pop)

        best_idx = int(np.argmin(fit))
        best_fit = float(fit[best_idx])
        best_hist[generation] = best_fit

        if best_fit < best_global_fit:
            best_global_fit = best_fit
            best_global = pop[best_idx].copy()
            stagnation = 0
        else:
            stagnation += 1

        elite_pop, _ = selsort(pop, fit, elite_count, reverse=False)

        parents, _ = seltourn(pop, fit, pop_size - elite_count, reverse=False)
        children = parents.copy()

        intmedx(children, alpha=alpha, mode=0)
        children = muta(children, rate=mut_rate, amp=mut_amp, space=SPACE)

        candidate_pool = np.vstack((elite_pop, children))

        if stagnation >= 60:
            immigrants = init_population(immigrant_count)
            candidate_pool = np.vstack((candidate_pool, immigrants))
            stagnation = 0

        candidate_fit = fitfun(candidate_pool)
        pop, _ = selsort(candidate_pool, candidate_fit, pop_size, reverse=False)

    fit = fitfun(pop)
    best_idx = int(np.argmin(fit))
    best_fit = float(fit[best_idx])

    if best_fit < best_global_fit:
        best_global_fit = best_fit
        best_global = pop[best_idx].copy()

    best_profit = profit(best_global)
    best_viol = violations(best_global)

    return best_global, best_global_fit, best_profit, best_viol, best_hist


def print_solution(x: np.ndarray, fitness: float, prof: float, v: np.ndarray) -> None:
    s = float(np.sum(x))

    print("x =", np.round(x, 2))
    print(f"fitness = {fitness:.4f}")
    print(f"profit = {prof:.2f}")
    # print(f"sum(x) <= 10 000 000      : {s:.2f}")
    # print(f"x1 + x2 <= 2 500 000      : {x[0] + x[1]:.2f}")
    # print(f"x4 - x5 >= 0              : {x[3] - x[4]:.2f}")
    # print(f"0.5*sum(x) - (x3+x4) >= 0 : {0.5 * s - (x[2] + x[3]):.2f}")
    # print("violations =", np.round(v, 4))
    # print("feasible =", bool(np.sum(v) <= 1e-9))


def plot_method_histories(method_name: str, all_histories: list[np.ndarray], all_final_fitness: list[float]) -> None:
    plt.figure(figsize=(10, 6))

    for i, hist in enumerate(all_histories, start=1):
        plt.plot(hist, linewidth=1.3, alpha=0.7, label=f"run {i:02d} ({all_final_fitness[i - 1]:.2f})")

    plt.title(f"{method_name}: fitness vs. generation")
    plt.xlabel("Generacia")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_best_comparison(best_histories: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(10, 6))

    for name, hist in best_histories.items():
        plt.plot(hist, linewidth=2, label=name)

    plt.title("Porovnanie najlepsich")
    plt.xlabel("Generacie")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    random.seed()
    np.random.seed()

    pop_size = 300
    generations = 600
    elite_count = 6
    cross_alpha = 0.8
    mut_rate = 0.1
    mut_amp = np.array([200_000.0, 200_000.0, 200_000.0, 200_000.0, 200_000.0], dtype=float)
    runs = 5

    methods = {
        "DEAD": fit_dead,
        "STEP": fit_step,
        "PROPORTIONAL": fit_prop,
    }

    best_histories = {}

    for name, fitfun in methods.items():
        print(f"\n===== {name} =====")

        all_histories = []
        all_final_fitness = []
        all_results = []

        for run in range(runs):
            best_x, best_fitness, best_profit, best_viol, history = ga_investment(
                fitfun=fitfun,
                pop_size=pop_size,
                generations=generations,
                elite_count=elite_count,
                alpha=cross_alpha,
                mut_rate=mut_rate,
                mut_amp=mut_amp,
            )

            all_histories.append(history)
            all_final_fitness.append(best_fitness)
            all_results.append((best_x, best_fitness, best_profit, best_viol, history))

            print(
                f"run {run + 1:02d}: "
                f"final fitness = {best_fitness:.4f}, "
                f"profit = {best_profit:.2f}, "
                # f"feasible = {bool(np.sum(best_viol) <= 1e-9)}"
            )

        best_x, best_fitness, best_profit, best_viol, best_history = min(all_results, key=lambda r: r[1])
        best_histories[name] = best_history

        print(f"\n=== BEST {name} ===")
        print_solution(best_x, best_fitness, best_profit, best_viol)

        plot_method_histories(name, all_histories, all_final_fitness)

    plot_best_comparison(best_histories)


if __name__ == "__main__":
    main()