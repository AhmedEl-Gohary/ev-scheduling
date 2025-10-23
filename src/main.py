#!/usr/bin/env python3

"""
EV Charging: objective evaluators and simple demo scheduler.

Usage:
  - Provide parameters in 'sample_input.json'.
  - To run a small demo (generate a trivial feasible schedule and evaluate it):
      python main.py --demo
  - To only load params and exit (useful as library):
      python main.py
"""

import argparse
from utilities import *
from algorithms.greedy import greedy_schedule
from algorithms.sa import simulated_annealing


def compare_algorithms(params):
    """Compare greedy and simulated annealing solutions."""

    print("\n" + "="*80)
    print("RUNNING GREEDY ALGORITHM")
    print("="*80)

    # Run greedy algorithm
    X_greedy, B_greedy = greedy_schedule(params)
    f1_greedy = compute_total_tardiness(X_greedy, params)
    f2_greedy = compute_peak_power(B_greedy, params["level_powers"])
    valid_spot_g, v_spot_g = check_spot_capacity(X_greedy, params)
    valid_power_g, v_power_g = check_station_power(B_greedy, params["level_powers"], params.get("P_max"))
    violations_greedy = {"spot_capacity": v_spot_g, "station_power": v_power_g}

    print("\nGreedy Solution Evaluation:")
    pretty_print_evaluation(f1_greedy, f2_greedy, violations_greedy)

    print("\n" + "="*80)
    print("RUNNING SIMULATED ANNEALING ALGORITHM")
    print("="*80)

    # Run simulated annealing starting from greedy solution
    X_sa, B_sa, sa_states = simulated_annealing(
        X_greedy, B_greedy, params,
        T0=10.0,
        Tf=1e-3,
        imax=100,  # number of temperature reductions
        nT=50,     # iterations per temperature
        rng_seed=42
    )

    f1_sa = compute_total_tardiness(X_sa, params)
    f2_sa = compute_peak_power(B_sa, params["level_powers"])
    valid_spot_sa, v_spot_sa = check_spot_capacity(X_sa, params)
    valid_power_sa, v_power_sa = check_station_power(B_sa, params["level_powers"], params.get("P_max"))
    violations_sa = {"spot_capacity": v_spot_sa, "station_power": v_power_sa}

    print("\nSimulated Annealing Solution Evaluation:")
    pretty_print_evaluation(f1_sa, f2_sa, violations_sa)

    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<30} {'Greedy':<20} {'SA':<20} {'Improvement':<20}")
    print("-"*80)
    print(f"{'Total Tardiness (slots)':<30} {f1_greedy:<20.3f} {f1_sa:<20.3f} {f1_greedy - f1_sa:<20.3f}")
    print(f"{'Peak Power (kW)':<30} {f2_greedy:<20.3f} {f2_sa:<20.3f} {f2_greedy - f2_sa:<20.3f}")

    # Calculate percentage improvements
    if f1_greedy > 0:
        tardiness_improvement = ((f1_greedy - f1_sa) / f1_greedy) * 100
        print(f"{'Tardiness Improvement (%)':<30} {'-':<20} {'-':<20} {tardiness_improvement:<20.2f}%")

    if f2_greedy > 0:
        peak_improvement = ((f2_greedy - f2_sa) / f2_greedy) * 100
        print(f"{'Peak Power Reduction (%)':<30} {'-':<20} {'-':<20} {peak_improvement:<20.2f}%")

    print("\n" + "="*80)
    print(f"Number of violations (Greedy): Spot={len(v_spot_g)}, Power={len(v_power_g)}")
    print(f"Number of violations (SA): Spot={len(v_spot_sa)}, Power={len(v_power_sa)}")
    print("="*80)

    # Show SA convergence info
    if len(sa_states) > 0:
        initial_obj = sa_states[0]['objective']
        final_obj = sa_states[-1]['objective']
        best_obj = sa_states[-1]['best_objective']
        print(f"\nSA Convergence:")
        print(f"  Initial objective: {initial_obj:.3f}")
        print(f"  Final objective: {final_obj:.3f}")
        print(f"  Best objective: {best_obj:.3f}")
        print(f"  Total iterations: {len(sa_states)}")

    return X_greedy, B_greedy, X_sa, B_sa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="inputs/t1.json", help="Path to JSON input parameters.")
    parser.add_argument("--demo", action="store_true", help="Run demo greedy scheduler and evaluate it.")
    parser.add_argument("--compare", action="store_true", help="Compare greedy and SA algorithms.")
    args = parser.parse_args()
    args.compare = True
    params = load_params(args.input)
    print("Loaded parameters from", args.input)

    if args.compare:
        X_greedy, B_greedy, X_sa, B_sa = compare_algorithms(params)

        # Export SA solution
        print("\nExporting SA solution to files...")
        export_output_to_txt(X_sa, B_sa, params)
        export_schedule_to_csv(X_sa, B_sa, params)
        print("SA solution exported to schedule_output.txt and schedule.csv")

    elif args.demo:
        print("Running demo greedy scheduler (very simple heuristic)...")
        X, B = greedy_schedule(params)
        f1 = compute_total_tardiness(X, params)
        f2 = compute_peak_power(B, params["level_powers"])
        valid_spot, v_spot = check_spot_capacity(X, params)
        valid_power, v_power = check_station_power(B, params["level_powers"], params.get("P_max"))
        violations = {"spot_capacity": v_spot, "station_power": v_power}
        pretty_print_evaluation(f1, f2, violations)
        pretty_print_schedule(X, B, params)
        export_output_to_txt(X, B, params)
        export_schedule_to_csv(X, B, params)
    else:
        print("Demo not requested. This script exposes evaluation functions for schedules.")
        print("To test evaluation with a generated schedule, run with --demo")
        print("To compare greedy and SA algorithms, run with --compare")
        print("Later milestones: provide schedules (X and B) and call compute_total_tardiness / compute_peak_power.")


if __name__ == "__main__":
    main()