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
import math
from utilities import *
from sa import simulated_annealing
# ---------------------------
# Simple greedy demo scheduler (optional)
# ---------------------------
def greedy_schedule(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    A simple earliest-fit greedy scheduler to produce a feasible schedule for demonstration.

    Strategy (very simple):
      - Iterate EVs in order of arrival (earliest first).
      - For each EV, try levels from lowest power to highest (to avoid peaks).
      - Try to place contiguous block at earliest start >= arrival on any available spot such that:
            * spot is free for the contiguous block
            * station power constraint not violated
      - If found, assign; else skip (leave unassigned).

    Returns:
      X: occupancy tensor shape (J, L, Smax, T)
      B: level tensor shape (J, K, T)
    """
    T = params["time_slots"]
    L = len(params["spots_per_line"])
    S_i = params["spots_per_line"]
    Smax = max(S_i)
    K = len(params["level_powers"])
    J = len(params["evs"])

    # initialize tensors
    X = np.zeros((J, L, Smax, T), dtype=int)
    B = np.zeros((J, K, T), dtype=int)
    station_power = np.zeros(T, dtype=float)

    # EV order by arrival
    ev_order = sorted(range(J), key=lambda j: params["evs"][j]["arrival_slot"])

    for j in ev_order:
        ev = params["evs"][j]
        arrival = ev["arrival_slot"]
        Ereq = ev["energy_required"]
        assigned = False

        # try levels from lowest to highest power (index order of level_powers)
        for ell in range(K):
            r_l = params["level_powers"][ell]
            # compute required slots (ceiling)
            p_jl = math.ceil(Ereq / (r_l * params["delta_t"]))
            if p_jl <= 0:
                p_jl = 1
            # try start times
            for start in range(arrival, T - p_jl + 1):
                end = start + p_jl  # exclusive
                # check station power feasibility if we add this EV at level ell in slots [start,end)
                can_place_power = True
                for t in range(start, end):
                    if station_power[t] + r_l > (params.get("P_max") or 1e12):
                        can_place_power = False
                        break
                if not can_place_power:
                    continue
                # find a spot in any line with contiguous free slots
                spot_found = False
                for i in range(L):
                    for s in range(S_i[i]):  # only up to actual spots
                        # check spot free in all slots
                        if X[j, :, :, :].shape:  # just dummy to avoid lint error
                            pass
                        occ = X[:, i, s, start:end].sum(axis=(0,1))  # sum over EVs, slots -> array
                        # actually above returns shape (end-start,), but we want any occupancy
                        occ_any = X[:, i, s, start:end].sum()
                        if occ_any == 0:
                            # place EV j there
                            X[j, i, s, start:end] = 1
                            B[j, ell, start:end] = 1
                            station_power[start:end] += r_l
                            spot_found = True
                            break
                    if spot_found:
                        break
                if spot_found:
                    assigned = True
                    break
            if assigned:
                break
        # if not assigned, EV stays unassigned (CT = None)
    return X, B

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
