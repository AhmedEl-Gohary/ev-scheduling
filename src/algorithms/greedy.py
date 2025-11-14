from src.utilities import *
from math import ceil

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
            p_jl = ceil(Ereq / (r_l * params["delta_t"]))
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