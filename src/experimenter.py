#!/usr/bin/env python3
"""
Comprehensive experiment runner for comparing SA and GA algorithms
Generates all metrics needed for the milestone report
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utilities import load_params
from src.algorithms.greedy import greedy_schedule
from src.algorithms.sa import simulated_annealing
from src.algorithms.ga import genetic_algorithm
from src.eval import compute_total_tardiness, compute_peak_power, objective_fn


def run_multiple_trials(algorithm_name: str, params: Dict[str, Any],
                        n_trials: int = 10, **kwargs) -> Dict[str, Any]:
    """Run algorithm multiple times to collect statistics"""
    results = {
        'objectives': [],
        'tardiness': [],
        'peak_power': [],
        'runtimes': [],
        'convergence_data': []
    }

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}...")
        start_time = time.time()

        if algorithm_name == "SA":
            X, B, states = simulated_annealing(
                params,
                T0=kwargs.get('T0', 1.0),
                Tf=kwargs.get('Tf', 1e-3),
                imax=kwargs.get('imax', 200),
                nT=kwargs.get('nT', 50),
                rng_seed=42 + trial
            )
        elif algorithm_name == "GA":
            X, B, states = genetic_algorithm(
                params,
                imax=kwargs.get('imax', 150),
                population_size=kwargs.get('population_size', 100),
                survivor_rate=kwargs.get('survivor_rate', 0.1),
                crossover_rate=kwargs.get('crossover_rate', 0.4),
                mutation_rate=kwargs.get('mutation_rate', 0.5),
                rng_seed=42 + trial,
                adaptive=kwargs.get('adaptive', True)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        runtime = time.time() - start_time

        obj = objective_fn(X, B, params)
        tard = compute_total_tardiness(X, params)
        peak = compute_peak_power(B, params['level_powers'])

        results['objectives'].append(obj)
        results['tardiness'].append(tard)
        results['peak_power'].append(peak)
        results['runtimes'].append(runtime)
        results['convergence_data'].append(states)

    # Compute statistics
    stats = {
        'mean_objective': np.mean(results['objectives']),
        'std_objective': np.std(results['objectives']),
        'best_objective': np.min(results['objectives']),
        'worst_objective': np.max(results['objectives']),
        'mean_tardiness': np.mean(results['tardiness']),
        'std_tardiness': np.std(results['tardiness']),
        'mean_peak_power': np.mean(results['peak_power']),
        'std_peak_power': np.std(results['peak_power']),
        'mean_runtime': np.mean(results['runtimes']),
        'std_runtime': np.std(results['runtimes']),
        'convergence': results['convergence_data'][0]  # Use first trial for convergence plot
    }

    return stats


def run_case_study(case_name: str, input_file: str, n_trials: int = 10):
    """Run complete comparison for one case study"""
    print(f"\n{'=' * 80}")
    print(f"CASE STUDY: {case_name}")
    print(f"{'=' * 80}\n")

    params = load_params(input_file)

    # Run Greedy (deterministic, single run)
    print("Running Greedy algorithm...")
    start = time.time()
    X_greedy, B_greedy = greedy_schedule(params)
    greedy_time = time.time() - start

    greedy_obj = objective_fn(X_greedy, B_greedy, params)
    greedy_tard = compute_total_tardiness(X_greedy, params)
    greedy_peak = compute_peak_power(B_greedy, params['level_powers'])

    print(f"  Objective: {greedy_obj:.3f}")
    print(f"  Tardiness: {greedy_tard:.3f}")
    print(f"  Peak Power: {greedy_peak:.3f} kW")
    print(f"  Runtime: {greedy_time:.3f}s\n")

    # Run SA multiple trials
    print(f"Running Simulated Annealing ({n_trials} trials)...")
    sa_stats = run_multiple_trials("SA", params, n_trials,
                                   T0=1.0, Tf=1e-3, imax=200, nT=50)

    print(f"  Best Objective: {sa_stats['best_objective']:.3f}")
    print(f"  Mean Objective: {sa_stats['mean_objective']:.3f} ± {sa_stats['std_objective']:.3f}")
    print(f"  Mean Runtime: {sa_stats['mean_runtime']:.3f}s ± {sa_stats['std_runtime']:.3f}s\n")

    # Run GA multiple trials
    print(f"Running Genetic Algorithm ({n_trials} trials)...")
    ga_stats = run_multiple_trials("GA", params, n_trials,
                                   imax=150, population_size=100,
                                   survivor_rate=0.1, crossover_rate=0.4,
                                   mutation_rate=0.5, adaptive=True)

    print(f"  Best Objective: {ga_stats['best_objective']:.3f}")
    print(f"  Mean Objective: {ga_stats['mean_objective']:.3f} ± {ga_stats['std_objective']:.3f}")
    print(f"  Mean Runtime: {ga_stats['mean_runtime']:.3f}s ± {ga_stats['std_runtime']:.3f}s\n")

    return {
        'case_name': case_name,
        'greedy': {
            'objective': greedy_obj,
            'tardiness': greedy_tard,
            'peak_power': greedy_peak,
            'runtime': greedy_time
        },
        'sa': sa_stats,
        'ga': ga_stats
    }


def print_latex_tables(all_results: List[Dict[str, Any]]):
    """Generate LaTeX tables for the report"""

    print("\n" + "=" * 80)
    print("LATEX TABLE 1: ALGORITHM COMPARISON - BEST SOLUTIONS")
    print("=" * 80 + "\n")

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(
        r"\textbf{Case} & \multicolumn{2}{c}{\textbf{Greedy}} & \multicolumn{2}{c}{\textbf{SA}} & \multicolumn{2}{c}{\textbf{GA}} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    print(r" & Tard. & Peak & Tard. & Peak & Tard. & Peak \\")
    print(r"\midrule")

    for result in all_results:
        case = result['case_name']
        g_tard = result['greedy']['tardiness']
        g_peak = result['greedy']['peak_power']

        # Use best from trials
        sa_best_idx = np.argmin([result['sa']['best_objective']])
        ga_best_idx = np.argmin([result['ga']['best_objective']])

        sa_tard = result['sa']['mean_tardiness']
        sa_peak = result['sa']['mean_peak_power']
        ga_tard = result['ga']['mean_tardiness']
        ga_peak = result['ga']['mean_peak_power']

        print(
            f"{case} & {g_tard:.1f} & {g_peak:.1f} & {sa_tard:.1f} & {sa_peak:.1f} & {ga_tard:.1f} & {ga_peak:.1f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Comparison of best solutions found by each algorithm}")
    print(r"\label{tab:algo_comparison_best}")
    print(r"\end{table}")

    print("\n" + "=" * 80)
    print("LATEX TABLE 2: PERFORMANCE METRICS")
    print("=" * 80 + "\n")

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r"\textbf{Case} & \textbf{Algorithm} & \textbf{Best} & \textbf{Mean} & \textbf{Std} & \textbf{Time (s)} \\")
    print(r"\midrule")

    for result in all_results:
        case = result['case_name']

        # Greedy
        print(
            f"{case} & Greedy & {result['greedy']['objective']:.2f} & {result['greedy']['objective']:.2f} & 0.00 & {result['greedy']['runtime']:.2f} \\\\")

        # SA
        print(
            f" & SA & {result['sa']['best_objective']:.2f} & {result['sa']['mean_objective']:.2f} & {result['sa']['std_objective']:.2f} & {result['sa']['mean_runtime']:.2f} \\\\")

        # GA
        print(
            f" & GA & {result['ga']['best_objective']:.2f} & {result['ga']['mean_objective']:.2f} & {result['ga']['std_objective']:.2f} & {result['ga']['mean_runtime']:.2f} \\\\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Performance metrics over 10 independent runs}")
    print(r"\label{tab:performance_metrics}")
    print(r"\end{table}")

    print("\n" + "=" * 80)
    print("LATEX TABLE 3: IMPROVEMENT OVER GREEDY")
    print("=" * 80 + "\n")

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Case} & \textbf{Algorithm} & \textbf{Obj. Impr.} & \textbf{Peak Impr.} & \textbf{Peak \%} \\")
    print(r"\midrule")

    for result in all_results:
        case = result['case_name']
        g_obj = result['greedy']['objective']
        g_peak = result['greedy']['peak_power']

        # SA improvement
        sa_obj_impr = g_obj - result['sa']['best_objective']
        sa_peak_impr = g_peak - result['sa']['mean_peak_power']
        sa_peak_pct = (sa_peak_impr / g_peak * 100) if g_peak > 0 else 0

        print(f"{case} & SA & {sa_obj_impr:.2f} & {sa_peak_impr:.2f} & {sa_peak_pct:.1f}\\% \\\\")

        # GA improvement
        ga_obj_impr = g_obj - result['ga']['best_objective']
        ga_peak_impr = g_peak - result['ga']['mean_peak_power']
        ga_peak_pct = (ga_peak_impr / g_peak * 100) if g_peak > 0 else 0

        print(f" & GA & {ga_obj_impr:.2f} & {ga_peak_impr:.2f} & {ga_peak_pct:.1f}\\% \\\\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Improvement over greedy baseline}")
    print(r"\label{tab:improvements}")
    print(r"\end{table}")


def main():
    # Define all case studies
    case_studies = [
        ("Sample", "inputs/t1.json"),
        ("Medium", "inputs/t2.json"),
        ("Big", "inputs/t3.json"),
        ("Large", "inputs/t4.json"),
        ("Peak Hour", "inputs/t5.json")
    ]

    all_results = []

    # Run all experiments
    for case_name, input_file in case_studies:
        result = run_case_study(case_name, input_file, n_trials=10)
        all_results.append(result)

    # Save results
    output_file = "experiment_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = []
        for r in all_results:
            json_r = {
                'case_name': r['case_name'],
                'greedy': {k: float(v) for k, v in r['greedy'].items()},
                'sa': {k: float(v) if not isinstance(v, list) else v
                       for k, v in r['sa'].items()},
                'ga': {k: float(v) if not isinstance(v, list) else v
                       for k, v in r['ga'].items()}
            }
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    print(f"\n\nResults saved to {output_file}")

    # Generate LaTeX tables
    print_latex_tables(all_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()