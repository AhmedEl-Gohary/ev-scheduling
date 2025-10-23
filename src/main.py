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
from src.algorithms.greedy import greedy_schedule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="sample_input.json", help="Path to JSON input parameters.")
    parser.add_argument("--demo", action="store_true", help="Run demo greedy scheduler and evaluate it.")
    args = parser.parse_args()

    params = load_params(args.input)
    print("Loaded parameters from", args.input)
    args.demo = True
    if args.demo:
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

        # X0, B0 = X, B  # your starting schedule
        # X_best, B_best, sa_info = simulated_annealing(
        #     X0, B0, params,
        #     T0=10.0, Tf=1e-3, imax=120, nT=60,
        #     w_tardiness=1.0, w_peak=0.5,
        #     rng_seed=42,
        #     verbose=True,  # prints top-level progress & improvements
        #     per_temp_verbose=False  # set True if you want every inner iteration printed (very chatty)
        # )
        #
        # # summary & final schedule
        # print_sa_summary(sa_info)
        # print("Final evaluation of X_best:")
        # f1_best = compute_total_tardiness(X_best, params)
        # f2_best = compute_peak_power(B_best, params["level_powers"])
        # pretty_print_evaluation(f1_best, f2_best, {"spot_capacity": [], "station_power": []})
        # pretty_print_schedule(X_best, B_best, params)
    else:
        print("Demo not requested. This script exposes evaluation functions for schedules.")
        print("To test evaluation with a generated schedule, run with --demo")
        print("Later milestones: provide schedules (X and B) and call compute_total_tardiness / compute_peak_power.")

if __name__ == "__main__":
    main()
