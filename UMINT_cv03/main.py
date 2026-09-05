import math
import random
import numpy as np
import matplotlib.pyplot as plt

from genetic_all import genrpop_perm, selsort, seltourn, swapgen, swappart

# Body zadania (start je prvy bod, ciel je posledny bod)
B = np.array([
    [0, 0], [17, 100], [51, 15], [70, 62], [42, 25], [32, 17],
    [51, 64], [39, 45], [68, 89], [20, 19], [12, 87], [80, 37],
    [35, 82], [2, 15], [38, 95], [33, 50], [85, 52], [97, 27],
    [99, 10], [37, 67], [20, 82], [49, 0], [62, 14], [7, 60], [0, 0]
], dtype=float)

# V Pythone pouzivame indexy 0..24.
# Body 0 a 24 su fixne [0, 0], preto sa permutuju len vnutorne indexy 1..23.
START_INDEX = 0
END_INDEX = len(B) - 1
INNER_MIN_INDEX = 1
INNER_MAX_INDEX = len(B) - 2

DIFF = B[:, None, :] - B[None, :, :]
DIST = np.sqrt(np.sum(DIFF * DIFF, axis=2))

def route_length(chromosome: np.ndarray) -> float:
    """Dlzka jednej trasy."""
    chrom = np.asarray(chromosome, dtype=int)
    route = np.concatenate(([START_INDEX], chrom, [END_INDEX]))
    return float(np.sum(DIST[route[:-1], route[1:]]))


def fitness_population(pop: np.ndarray) -> np.ndarray:
    """Fitness pre celu populaciu naraz."""
    pop = np.asarray(pop, dtype=int)
    starts = np.full((pop.shape[0], 1), START_INDEX, dtype=int)
    ends = np.full((pop.shape[0], 1), END_INDEX, dtype=int)
    routes = np.hstack((starts, pop, ends))
    return np.sum(DIST[routes[:, :-1], routes[:, 1:]], axis=1)


def invord(old_pop: np.ndarray, rate: float) -> np.ndarray:
    """Inversion mutation from UMINT-GA/Kod/invord.py."""
    old_pop = np.asarray(old_pop)
    if old_pop.ndim == 1:
        old_pop = old_pop[None, :]

    lpop, lstring = old_pop.shape
    rate = 1.0 if rate > 1 else (0.0 if rate < 0 else float(rate))
    n = int(np.ceil(lpop * rate * np.random.rand()))
    new_pop = old_pop.copy()

    for _ in range(n):
        r = int(np.ceil(np.random.rand() * lpop)) - 1
        p1 = int(np.ceil(0.001 + np.random.rand() * (lstring - 1)))
        p2 = int(np.ceil(0.001 + np.random.rand() * (lstring - p1))) + p1

        if p1 == lstring:
            p1 = lstring - 1
        if p2 > lstring:
            p2 = lstring

        new_pop[r, p1 - 1:p2] = old_pop[r, p1 - 1:p2][::-1]
        old_pop = new_pop.copy()

    return new_pop


def full_route_indices(best_inner: np.ndarray) -> list[int]:
    """Vrati plnu trasu v Python indexoch."""
    return [START_INDEX] + [int(x) for x in best_inner] + [END_INDEX]

#
# def full_route_indices_matlab_style(best_inner: np.ndarray) -> list[int]:
#     """Vrati trasu v style zadania/prezentacie (1-based indexy)."""
#     return [1] + [int(x) + 1 for x in best_inner] + [25]


def ga_tsp(pop_size: int, generations: int, elite_count: int, mut_swap: float, mut_part: float, mut_inv: float) -> tuple[np.ndarray, float, np.ndarray]:
    """GA pre TSP s permutacnymi operatormi."""
    pop = genrpop_perm(pop_size, INNER_MIN_INDEX, INNER_MAX_INDEX).astype(int)

    best_hist = np.zeros(generations, dtype=float)
    best_global = pop[0].copy()
    best_global_fit = math.inf
    stagnation = 0

    for generation in range(generations):
        fit = fitness_population(pop)

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

        base1, _ = seltourn(pop, fit, pop_size - elite_count, reverse=False)
        base2, _ = seltourn(pop, fit, pop_size - elite_count, reverse=False)
        base3, _ = seltourn(pop, fit, pop_size - elite_count, reverse=False)
        base4, _ = seltourn(pop, fit, pop_size - elite_count, reverse=False)

        cand1 = base1.copy()
        cand2 = base2.copy()
        cand3 = base3.copy()
        cand4 = base4.copy()

        swapgen(cand1, rate=mut_swap)
        swappart(cand2, rate=mut_part)
        cand3 = invord(cand3, rate=mut_inv)
        swapgen(cand4, rate=mut_swap)
        cand4 = invord(cand4, rate=mut_inv)

        swappart(cand1, mut_part)
        swapgen(cand1, mut_swap)
        cand1 = invord(cand1, mut_inv)


        candidate_pool = np.vstack((elite_pop, cand1, cand2, cand3, cand4))
        #candidate_pool = np.vstack((elite_pop, cand1))


        if stagnation >= 75:
            immigrant_count = max(60, pop_size // 6)
            immigrants = genrpop_perm(immigrant_count, INNER_MIN_INDEX, INNER_MAX_INDEX).astype(int)
            candidate_pool = np.vstack((candidate_pool, immigrants))
            stagnation = 0

        candidate_fit = fitness_population(candidate_pool)
        pop, _ = selsort(candidate_pool, candidate_fit, pop_size, reverse=False)

    fit = fitness_population(pop)
    best_idx = int(np.argmin(fit))
    best_fit = float(fit[best_idx])

    if best_fit < best_global_fit:
        best_global_fit = best_fit
        best_global = pop[best_idx].copy()

    return best_global, best_global_fit, best_hist


def plot_histories(all_histories: list[np.ndarray]) -> None:
    """Spolocny graf fitness vs. generacia pre vsetky behy + priemer."""
    plt.figure(figsize=(11, 6))

    for i, hist in enumerate(all_histories, start=1):
        plt.plot(hist, alpha=0.45, linewidth=1.2, label=f"run {i}")

    mean_hist = np.mean(np.vstack(all_histories), axis=0)
    plt.plot(mean_hist, linewidth=3, label="average")

    plt.title("GA pre TSP: fitness vs. generacia")
    plt.xlabel("Generacia")
    plt.ylabel("Dlzka trasy")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_best_route(best_inner: np.ndarray, best_length: float) -> None:
    """Vykresli body v rovine a najlepsiu najdenu trasu."""
    route = full_route_indices(best_inner)
    pts = B[route]

    plt.figure(figsize=(8, 8))
    plt.plot(pts[:, 0], pts[:, 1], marker="o")

    for idx in route:
        plt.text(B[idx, 0] + 1.5, B[idx, 1] + 1.5, str(idx + 1), fontsize=8)

    plt.scatter(B[START_INDEX, 0], B[START_INDEX, 1], s=120, marker="s", label="start/ciel")
    plt.title(f"Najlepsia trasa robota, dlzka = {best_length:.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main() -> None:
    random.seed()
    np.random.seed()

    pop_size = 450
    generations = 600
    elite_count = 15
    mut_swap = 0.03
    mut_part = 0.05
    mut_inv = 0.24
    runs = 10

    all_histories: list[np.ndarray] = []
    all_final_lengths: list[float] = []

    best_overall = None
    best_overall_length = math.inf

    for run in range(runs):
        best_inner, best_length, history = ga_tsp(
            pop_size=pop_size,
            generations=generations,
            elite_count=elite_count,
            mut_swap=mut_swap,
            mut_part=mut_part,
            mut_inv=mut_inv,
        )

        all_histories.append(history)
        all_final_lengths.append(best_length)

        if best_length < best_overall_length:
            best_overall = best_inner.copy()
            best_overall_length = best_length

        print(f"run {run + 1:02d}: best_length = {best_length:.4f}")

    print(f"najlepsia dlzka: {best_overall_length:.4f}")
    print("najlepsi gen (Python indexy):")
    print(best_overall)
    # print("Najlepsi genom v style zadania (1-based indexy):")
    # print(full_route_indices_matlab_style(best_overall))

    plot_histories(all_histories)
    plot_best_route(best_overall, best_overall_length)


if __name__ == "__main__":
    main()
