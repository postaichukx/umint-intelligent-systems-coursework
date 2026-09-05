import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Pouzite casti z dodaneho genetic toolboxu
from genetic_all import genrpop_perm, selbest, seltourn, swapgen

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
INNER_COUNT = INNER_MAX_INDEX - INNER_MIN_INDEX + 1


def build_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Predvypocet euklidovskych vzdialenosti medzi vsetkymi bodmi."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


DIST = build_distance_matrix(B)


def route_length(chromosome: np.ndarray) -> float:
    """Fitness = celkova dlzka trasy pre permutacny chromozom.

    Chromozom obsahuje iba vnutorne body 1..23.
    Realna trasa je: [0] + chromosome + [24].
    """
    chromosome = np.asarray(chromosome, dtype=int)

    total = DIST[START_INDEX, chromosome[0]]
    total += DIST[chromosome[-1], END_INDEX]
    total += np.sum(DIST[chromosome[:-1], chromosome[1:]])
    return float(total)


def fitness_population(pop: np.ndarray) -> np.ndarray:
    """Vrati fitness pre celu populaciu."""
    pop = np.asarray(pop, dtype=int)
    fit = np.zeros(pop.shape[0], dtype=float)
    for i in range(pop.shape[0]):
        fit[i] = route_length(pop[i])
    return fit


def order_crossover_pair(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Order Crossover (OX) vhodny pre permutacne ulohy.

    Zachova platnu permutaciu bez duplikatov a bez straty prvkov.
    """
    n = len(parent1)
    left, right = sorted(rng.choice(n, size=2, replace=False))

    child1 = np.full(n, -1, dtype=int)
    child2 = np.full(n, -1, dtype=int)

    child1[left:right + 1] = parent1[left:right + 1]
    child2[left:right + 1] = parent2[left:right + 1]

    used1 = set(child1[left:right + 1].tolist())
    used2 = set(child2[left:right + 1].tolist())

    write_positions = list(range(right + 1, n)) + list(range(0, left))

    fill1 = [gene for gene in list(parent2[right + 1:]) + list(parent2[:right + 1]) if gene not in used1]
    fill2 = [gene for gene in list(parent1[right + 1:]) + list(parent1[:right + 1]) if gene not in used2]

    for pos, gene in zip(write_positions, fill1):
        child1[pos] = gene
    for pos, gene in zip(write_positions, fill2):
        child2[pos] = gene

    return child1, child2


def order_crossover(pop: np.ndarray, pcross: float, rng: np.random.Generator) -> np.ndarray:
    """Krizenie populacie po dvojiciach pomocou OX."""
    children = pop.copy()
    pair_order = np.arange(children.shape[0])
    rng.shuffle(pair_order)

    for i in range(0, len(pair_order) - 1, 2):
        if rng.random() < pcross:
            p1 = pair_order[i]
            p2 = pair_order[i + 1]
            children[p1], children[p2] = order_crossover_pair(children[p1], children[p2], rng)

    return children


def inversion_mutation(pop: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Inverzna mutacia: vyberie usek a otoci jeho poradie.

    Tento operator je vhodny pre TSP/permutacie, lebo stale zachova rovnake prvky.
    """
    mutated = pop.copy()

    for row in range(mutated.shape[0]):
        if rng.random() < rate:
            i, j = sorted(rng.choice(mutated.shape[1], size=2, replace=False))
            mutated[row, i:j + 1] = mutated[row, i:j + 1][::-1]

    return mutated


def full_route_indices(best_inner: np.ndarray) -> list[int]:
    """Vrati plnu trasu v Python indexoch."""
    return [START_INDEX] + [int(x) for x in best_inner] + [END_INDEX]


def full_route_indices_matlab_style(best_inner: np.ndarray) -> list[int]:
    """Vrati trasu v style zadania/prezentacie (1-based indexy)."""
    return [1] + [int(x) + 1 for x in best_inner] + [25]


def ga_tsp(
    rng: np.random.Generator,
    pop_size: int,
    generations: int,
    elite_count: int,
    pcross: float,
    pmut_inv: float,
    pmut_swap: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Jedno spustenie GA pre TSP ulohu."""

    # Inicializacia populacie: kazdy jedinec je permutacia 1..23
    pop = genrpop_perm(pop_size, INNER_MIN_INDEX, INNER_MAX_INDEX).astype(int)

    best_hist = np.zeros(generations, dtype=float)
    best_global = None
    best_global_fit = math.inf

    for generation in range(generations):
        fit = fitness_population(pop)

        current_best_idx = int(np.argmin(fit))
        current_best_fit = float(fit[current_best_idx])
        best_hist[generation] = current_best_fit

        if current_best_fit < best_global_fit:
            best_global_fit = current_best_fit
            best_global = pop[current_best_idx].copy()

        # Elitizmus
        elite_pop, _ = selbest(pop, fit, [elite_count], reverse=False)

        # Rodicia - turnajovy vyber z toolboxu
        parents, _ = seltourn(pop, fit, pop_size - elite_count, reverse=False)

        # Permutacne krizene OX
        children = order_crossover(parents, pcross=pcross, rng=rng)

        # Permutacna mutacia 1: inverzia useku
        children = inversion_mutation(children, rate=pmut_inv, rng=rng)

        # Permutacna mutacia 2: swap dvoch genov z toolboxu
        children = children.copy()
        swapgen(children, rate=pmut_swap)

        # Nova generacia
        pop = np.vstack([elite_pop, children])

    final_fit = fitness_population(pop)
    final_best_idx = int(np.argmin(final_fit))
    final_best = pop[final_best_idx].copy()
    final_best_fit = float(final_fit[final_best_idx])

    if final_best_fit < best_global_fit:
        best_global = final_best
        best_global_fit = final_best_fit

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
        plt.text(B[idx, 0] + 1.5, B[idx, 1] + 1.5, str(idx), fontsize=8)

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
    rng = np.random.default_rng()
    random.seed()
    np.random.seed()

    # Parametre - kludne mozes dalej ladit
    pop_size = 120
    generations = 220
    elite_count = 8
    pcross = 0.95
    pmut_inv = 0.30
    pmut_swap = 0.12
    runs = 10

    all_histories: list[np.ndarray] = []
    all_final_lengths: list[float] = []

    best_overall = None
    best_overall_length = math.inf

    for run in range(runs):
        best_inner, best_length, history = ga_tsp(
            rng=rng,
            pop_size=pop_size,
            generations=generations,
            elite_count=elite_count,
            pcross=pcross,
            pmut_inv=pmut_inv,
            pmut_swap=pmut_swap,
        )

        all_histories.append(history)
        all_final_lengths.append(best_length)

        if best_length < best_overall_length:
            best_overall = best_inner.copy()
            best_overall_length = best_length

        print(f"run {run + 1:02d}: best_length = {best_length:.4f}")

    success_count = sum(length <= 480.0 for length in all_final_lengths)

    print("\nFinalne dlzky tras pre vsetky behy:")
    for i, length in enumerate(all_final_lengths, start=1):
        print(f"  run {i:02d}: {length:.4f}")

    print(f"\nPocet behov s vysledkom <= 480: {success_count}/{runs}")
    print(f"Najlepsia najdena dlzka: {best_overall_length:.4f}")
    print("Najlepsi genom (Python indexy):")
    print(best_overall)
    print("Najlepsi genom v style zadania (1-based indexy):")
    print(full_route_indices_matlab_style(best_overall))

    plot_histories(all_histories)
    plot_best_route(best_overall, best_overall_length)


if __name__ == "__main__":
    main()
