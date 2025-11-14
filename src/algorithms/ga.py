# ga_solver_improved.py
import random
import copy
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.schedule_generator import (
    extract_assignments_from_XB,
    build_XB_from_assignments,
    make_neighbor_from_XB
)
from src.eval import objective_fn, compute_total_tardiness, compute_peak_power
from src.algorithms.greedy import greedy_schedule


def initialize_population(params: Dict[str, Any],
                         pop_size: int,
                         rng: random.Random) -> List[Dict[str, Any]]:
    """
    Initialize population with diverse solutions using multiple strategies.
    """
    print(f"Initializing population of size {pop_size}...")
    population = []

    # Start with greedy solution
    X_greedy, B_greedy = greedy_schedule(params)
    fitness_greedy = objective_fn(X_greedy, B_greedy, params)
    assignments_greedy = extract_assignments_from_XB(X_greedy, B_greedy, params)

    population.append({
        'X': X_greedy,
        'B': B_greedy,
        'fitness': fitness_greedy,
        'assignments': assignments_greedy
    })

    # Generate diverse variations with different mutation intensities
    attempts = 0
    max_total_attempts = pop_size * 150

    while len(population) < pop_size and attempts < max_total_attempts:
        attempts += 1

        # Choose base solution
        base_idx = rng.randint(0, len(population) - 1)
        X_base = population[base_idx]['X']
        B_base = population[base_idx]['B']

        # Apply multiple mutations with varying intensity
        if len(population) < pop_size // 3:
            num_mutations = rng.randint(1, 3)  # Light mutations
        elif len(population) < 2 * pop_size // 3:
            num_mutations = rng.randint(2, 5)  # Medium mutations
        else:
            num_mutations = rng.randint(3, 7)  # Heavy mutations for diversity

        X_new, B_new = X_base.copy(), B_base.copy()
        valid = False

        for _ in range(num_mutations):
            X_new, B_new, _, valid, _ = make_neighbor_from_XB(
                X_new, B_new, params, max_attempts=5, neighbor_tries=50
            )
            if not valid:
                break

        if valid:
            fitness_new = objective_fn(X_new, B_new, params)
            assignments_new = extract_assignments_from_XB(X_new, B_new, params)

            # Add if sufficiently different from existing solutions
            is_diverse = True
            if len(population) > 5:
                # Check diversity: fitness should differ by at least 1%
                for existing in population[-5:]:
                    if abs(fitness_new - existing['fitness']) / (existing['fitness'] + 1e-6) < 0.01:
                        is_diverse = False
                        break

            if is_diverse or len(population) < pop_size * 0.7:
                population.append({
                    'X': X_new,
                    'B': B_new,
                    'fitness': fitness_new,
                    'assignments': assignments_new
                })

    # Fill remaining with copies if needed
    while len(population) < pop_size:
        idx = rng.randint(0, len(population) - 1)
        population.append(copy.deepcopy(population[idx]))

    print(f"Population initialized with {len(population)} individuals")
    return population


def crossover_time_block(parent1_assign: List[Dict[str, Any]],
                         parent2_assign: List[Dict[str, Any]],
                         params: Dict[str, Any],
                         rng: random.Random,
                         max_tries: int = 50) -> Tuple[List[Dict[str, Any]],
                                                       List[Dict[str, Any]], bool]:
    """
    Time-based crossover: Split schedule at a time point and swap segments.
    This preserves temporal structure better than random EV swapping.
    """
    J = len(parent1_assign)
    T = params['time_slots']

    for attempt in range(max_tries):
        # Choose a time split point (avoid extremes)
        split_time = rng.randint(T // 4, 3 * T // 4)

        child1_assign = copy.deepcopy(parent1_assign)
        child2_assign = copy.deepcopy(parent2_assign)

        # For each EV, decide which parent to take from based on their schedule timing
        for j in range(J):
            p1_start = parent1_assign[j].get('start', T)
            p2_start = parent2_assign[j].get('start', T)

            # Child1: Take from parent1 if starts before split, else parent2
            # Child2: opposite
            if p1_start < split_time:
                child1_assign[j] = copy.deepcopy(parent1_assign[j])
                child2_assign[j] = copy.deepcopy(parent2_assign[j])
            else:
                child1_assign[j] = copy.deepcopy(parent2_assign[j])
                child2_assign[j] = copy.deepcopy(parent1_assign[j])

        # Validate children
        X1, B1, valid1, _ = build_XB_from_assignments(child1_assign, params)
        X2, B2, valid2, _ = build_XB_from_assignments(child2_assign, params)

        if valid1 and valid2:
            return child1_assign, child2_assign, True

    return copy.deepcopy(parent1_assign), copy.deepcopy(parent2_assign), False


def crossover_line_based(parent1_assign: List[Dict[str, Any]],
                         parent2_assign: List[Dict[str, Any]],
                         params: Dict[str, Any],
                         rng: random.Random,
                         max_tries: int = 50) -> Tuple[List[Dict[str, Any]],
                                                       List[Dict[str, Any]], bool]:
    """
    Line-based crossover: Swap EVs assigned to specific charging lines.
    This preserves spatial structure.
    """
    J = len(parent1_assign)
    L = len(params['spots_per_line'])

    if L <= 1:
        return copy.deepcopy(parent1_assign), copy.deepcopy(parent2_assign), False

    for attempt in range(max_tries):
        # Choose random subset of lines to swap
        num_lines_to_swap = rng.randint(1, max(1, L // 2))
        lines_to_swap = set(rng.sample(range(L), num_lines_to_swap))

        child1_assign = copy.deepcopy(parent1_assign)
        child2_assign = copy.deepcopy(parent2_assign)

        # Swap assignments for EVs on selected lines
        for j in range(J):
            p1_line = parent1_assign[j].get('line', -1)
            p2_line = parent2_assign[j].get('line', -1)

            if p1_line in lines_to_swap:
                child1_assign[j] = copy.deepcopy(parent2_assign[j])
            if p2_line in lines_to_swap:
                child2_assign[j] = copy.deepcopy(parent1_assign[j])

        # Validate
        X1, B1, valid1, _ = build_XB_from_assignments(child1_assign, params)
        X2, B2, valid2, _ = build_XB_from_assignments(child2_assign, params)

        if valid1 and valid2:
            return child1_assign, child2_assign, True

    return copy.deepcopy(parent1_assign), copy.deepcopy(parent2_assign), False


def crossover_power_level_focused(parent1_assign: List[Dict[str, Any]],
                                   parent2_assign: List[Dict[str, Any]],
                                   params: Dict[str, Any],
                                   rng: random.Random,
                                   max_tries: int = 50) -> Tuple[List[Dict[str, Any]],
                                                                 List[Dict[str, Any]], bool]:
    """
    Power-level crossover: Swap timing/location but mix power level strategies.
    Child inherits timing from one parent and power levels from another.
    """
    J = len(parent1_assign)

    for attempt in range(max_tries):
        # Choose random split
        num_evs = rng.randint(1, max(1, J // 2))
        evs_to_mix = set(rng.sample(range(J), num_evs))

        child1_assign = copy.deepcopy(parent1_assign)
        child2_assign = copy.deepcopy(parent2_assign)

        # For selected EVs: keep timing/location from one parent, levels from other
        for j in evs_to_mix:
            # Child1: parent1 timing, parent2 levels
            if parent1_assign[j].get('assigned') and parent2_assign[j].get('assigned'):
                child1_assign[j]['levels'] = copy.deepcopy(parent2_assign[j]['levels'])
                # Adjust if duration differs
                dur1 = child1_assign[j]['end'] - child1_assign[j]['start']
                if len(child1_assign[j]['levels']) != dur1:
                    # Repeat or truncate
                    if len(child1_assign[j]['levels']) > dur1:
                        child1_assign[j]['levels'] = child1_assign[j]['levels'][:dur1]
                    else:
                        child1_assign[j]['levels'] = child1_assign[j]['levels'] * (dur1 // len(child1_assign[j]['levels']) + 1)
                        child1_assign[j]['levels'] = child1_assign[j]['levels'][:dur1]

                # Child2: parent2 timing, parent1 levels
                child2_assign[j]['levels'] = copy.deepcopy(parent1_assign[j]['levels'])
                dur2 = child2_assign[j]['end'] - child2_assign[j]['start']
                if len(child2_assign[j]['levels']) != dur2:
                    if len(child2_assign[j]['levels']) > dur2:
                        child2_assign[j]['levels'] = child2_assign[j]['levels'][:dur2]
                    else:
                        child2_assign[j]['levels'] = child2_assign[j]['levels'] * (dur2 // len(child2_assign[j]['levels']) + 1)
                        child2_assign[j]['levels'] = child2_assign[j]['levels'][:dur2]

        # Validate
        X1, B1, valid1, _ = build_XB_from_assignments(child1_assign, params)
        X2, B2, valid2, _ = build_XB_from_assignments(child2_assign, params)

        if valid1 and valid2:
            return child1_assign, child2_assign, True

    return copy.deepcopy(parent1_assign), copy.deepcopy(parent2_assign), False


def crossover_random_subset(parent1_assign: List[Dict[str, Any]],
                            parent2_assign: List[Dict[str, Any]],
                            params: Dict[str, Any],
                            rng: random.Random,
                            max_tries: int = 50) -> Tuple[List[Dict[str, Any]],
                                                          List[Dict[str, Any]], bool]:
    """
    Original random subset crossover (kept for diversity).
    """
    J = len(parent1_assign)

    if J <= 1:
        return copy.deepcopy(parent1_assign), copy.deepcopy(parent2_assign), True

    for attempt in range(max_tries):
        max_r = max(1, J // 2)
        r = rng.randint(1, max_r)
        subset_indices = rng.sample(range(J), r)

        child1_assign = copy.deepcopy(parent1_assign)
        child2_assign = copy.deepcopy(parent2_assign)

        for idx in subset_indices:
            child1_assign[idx] = copy.deepcopy(parent2_assign[idx])
            child2_assign[idx] = copy.deepcopy(parent1_assign[idx])

        X1, B1, valid1, _ = build_XB_from_assignments(child1_assign, params)
        X2, B2, valid2, _ = build_XB_from_assignments(child2_assign, params)

        if valid1 and valid2:
            return child1_assign, child2_assign, True

    return copy.deepcopy(parent1_assign), copy.deepcopy(parent2_assign), False


def adaptive_crossover(parent1_assign: List[Dict[str, Any]],
                      parent2_assign: List[Dict[str, Any]],
                      params: Dict[str, Any],
                      rng: random.Random,
                      generation: int) -> Tuple[List[Dict[str, Any]],
                                               List[Dict[str, Any]], bool]:
    """
    Adaptive crossover that chooses strategy based on generation.
    Early generations: more exploration (random, time-based)
    Later generations: more exploitation (line-based, power-focused)
    """
    crossover_strategies = [
        crossover_time_block,
        crossover_line_based,
        crossover_power_level_focused,
        crossover_random_subset
    ]

    # Weight strategies differently based on generation progress
    # Early: [0.3, 0.2, 0.2, 0.3]
    # Late: [0.2, 0.3, 0.4, 0.1]
    if generation < 100:
        weights = [0.3, 0.2, 0.2, 0.3]
    elif generation < 500:
        weights = [0.25, 0.25, 0.3, 0.2]
    else:
        weights = [0.2, 0.3, 0.4, 0.1]

    # Choose strategy
    strategy = rng.choices(crossover_strategies, weights=weights)[0]

    return strategy(parent1_assign, parent2_assign, params, rng)


def genetic_algorithm(params: Dict[str, Any],
                     imax: int = 150,
                     population_size: int = 100,
                     survivor_rate: float = 0.1,
                     crossover_rate: float = 0.4,
                     mutation_rate: float = 0.5,
                     rng_seed: Optional[int] = 42,
                     adaptive: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Improved Genetic Algorithm for EV charging scheduling.

    Args:
        params: Problem parameters
        imax: Maximum number of iterations (generations)
        population_size: Number of solutions per generation
        survivor_rate: Fraction of best individuals kept as elites
        crossover_rate: Fraction of population from crossover
        mutation_rate: Fraction of population from mutation
        rng_seed: Random seed
        adaptive: Use adaptive crossover strategies

    Returns:
        Best X, Best B, states history
    """
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    start_time = time.time()
    states = []

    # Calculate counts
    survivor_count = max(1, int(population_size * survivor_rate))
    crossover_count = max(0, int(population_size * crossover_rate))
    mutation_count = max(0, population_size - survivor_count - crossover_count)

    print(f"=== Improved Genetic Algorithm ===")
    print(f"Population size: {population_size}")
    print(f"Max iterations: {imax}")
    print(f"Survivors: {survivor_rate*100:.1f}% ({survivor_count})")
    print(f"Crossover: {crossover_rate*100:.1f}% ({crossover_count})")
    print(f"Mutation: {mutation_rate*100:.1f}% ({mutation_count})")
    print(f"Adaptive crossover: {adaptive}")
    print("=" * 35 + "\n")

    # Initialize population
    population = initialize_population(params, population_size, rng)

    # Track best solution
    best_individual = min(population, key=lambda x: x['fitness'])
    X_best = best_individual['X'].copy()
    B_best = best_individual['B'].copy()
    f_best = best_individual['fitness']

    # Record initial state
    states.append({
        'generation': 0,
        'X': X_best.tolist(),
        'B': B_best.tolist(),
        'objective': f_best,
        'best_objective': f_best,
        'avg_fitness': np.mean([ind['fitness'] for ind in population]),
        'tardiness': compute_total_tardiness(X_best, params),
        'peak_power': compute_peak_power(B_best, params["level_powers"])
    })

    print(f"Generation 0: Best = {f_best:.3f}")

    no_improvement_count = 0
    best_fitness_last = f_best

    # Evolution loop
    for gen in range(1, imax + 1):
        new_population = []

        # Sort by fitness
        population_sorted = sorted(population, key=lambda x: x['fitness'])

        # Elitism: Keep best survivors
        for i in range(survivor_count):
            new_population.append(copy.deepcopy(population_sorted[i]))

        # Crossover
        crossover_generated = 0
        best_parent = population_sorted[0]

        # Try different pairing strategies
        parent_pool_size = min(len(population_sorted), max(10, population_size // 5))
        other_parent_idx = 1

        while crossover_generated < crossover_count:
            if other_parent_idx >= parent_pool_size:
                other_parent_idx = 1

            other_parent = population_sorted[other_parent_idx]
            other_parent_idx += 1

            # Perform adaptive or random crossover
            if adaptive:
                child1_assign, child2_assign, success = adaptive_crossover(
                    best_parent['assignments'],
                    other_parent['assignments'],
                    params,
                    rng,
                    gen
                )
            else:
                child1_assign, child2_assign, success = crossover_random_subset(
                    best_parent['assignments'],
                    other_parent['assignments'],
                    params,
                    rng
                )

            if success:
                for child_assign in [child1_assign, child2_assign]:
                    if crossover_generated >= crossover_count:
                        break

                    X_child, B_child, valid, _ = build_XB_from_assignments(child_assign, params)

                    if valid:
                        fitness_child = objective_fn(X_child, B_child, params)
                        new_population.append({
                            'X': X_child,
                            'B': B_child,
                            'fitness': fitness_child,
                            'assignments': child_assign
                        })
                        crossover_generated += 1

        # Mutation: Apply to worst individuals
        mutation_generated = 0
        # Mutation intensity increases if stuck
        mutation_intensity = 1 if no_improvement_count < 50 else 2

        worst_indices = list(range(len(population_sorted)))

        for idx in worst_indices:
            if mutation_generated >= mutation_count:
                break

            individual = population_sorted[idx]

            # Apply mutations with varying intensity
            X_mutated, B_mutated = individual['X'], individual['B']
            valid = False

            for _ in range(mutation_intensity):
                X_mutated, B_mutated, assign_mutated, valid, _ = make_neighbor_from_XB(
                    X_mutated, B_mutated,
                    params,
                    max_attempts=10,
                    neighbor_tries=100
                )
                if not valid:
                    break

            if valid:
                fitness_mutated = objective_fn(X_mutated, B_mutated, params)
                new_population.append({
                    'X': X_mutated,
                    'B': B_mutated,
                    'fitness': fitness_mutated,
                    'assignments': assign_mutated
                })
                mutation_generated += 1

        # Fill remaining slots if needed
        while len(new_population) < population_size:
            idx = rng.randint(0, min(len(population_sorted) - 1, survivor_count))
            new_population.append(copy.deepcopy(population_sorted[idx]))

        new_population = new_population[:population_size]
        population = new_population

        # Update best solution
        current_best = min(population, key=lambda x: x['fitness'])
        if current_best['fitness'] < f_best:
            f_best = current_best['fitness']
            X_best = current_best['X'].copy()
            B_best = current_best['B'].copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Record state
        avg_fitness = np.mean([ind['fitness'] for ind in population])
        states.append({
            'generation': gen,
            'X': X_best.tolist(),
            'B': B_best.tolist(),
            'objective': current_best['fitness'],
            'best_objective': f_best,
            'avg_fitness': avg_fitness,
            'tardiness': compute_total_tardiness(X_best, params),
            'peak_power': compute_peak_power(B_best, params["level_powers"])
        })

        # Progress reporting
        if gen % 10 == 0 or gen == 1:
            elapsed = time.time() - start_time
            print(f"Gen {gen:4d}/{imax}: Best={f_best:.3f}, Current={current_best['fitness']:.3f}, "
                  f"Avg={avg_fitness:.3f}, NoImpr={no_improvement_count}, Time={elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\n=== GA Completed ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best objective: {f_best:.6f}")
    print(f"Tardiness: {compute_total_tardiness(X_best, params):.3f} slots")
    print(f"Peak power: {compute_peak_power(B_best, params['level_powers']):.3f} kW")
    print("=" * 20 + "\n")

    return X_best, B_best, states