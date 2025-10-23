import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def compute_completion_times_from_X(X: np.ndarray) -> List[Optional[int]]:
    """
    Given X[j,i,s,t] occupancy tensor, return completion slot CT_j (0-based index) for each EV j,
    or None if EV never charged.
    """
    J = X.shape[0]
    T = X.shape[-1]
    CT = [None] * J
    for j in range(J):
        # sum over i,s -> occupancy per time
        occ_t = X[j].sum(axis=(0,1))
        charged_slots = np.where(occ_t > 0)[0]
        if charged_slots.size > 0:
            CT[j] = int(charged_slots.max())  # 0-based slot index of last charging slot
        else:
            CT[j] = None
    return CT

def compute_total_tardiness(X: np.ndarray, params: Dict[str, Any]) -> float:
    """
    f1 = sum_j max(0, CT_j - d_j)
    CT_j and d_j are in discrete slot indices; here d_j is expected 0-based.
    Return float (sum of tardiness in slots). If you'd prefer hours multiply by delta_t.
    """
    CT = compute_completion_times_from_X(X)
    tardiness_sum = 0.0
    evs = params["evs"]
    for j, ev in enumerate(evs):
        d_j = ev["departure_slot"]  # expected 0-based in params
        if CT[j] is None:
            # treat never charged as finishing at +inf -> very large tardiness
            # For evaluation we can count as (T - d_j) or raise — here we count remaining horizon as tardiness
            # but better to warn
            # We'll count as (T) - d_j
            T = params["time_slots"]
            tard = max(0, T - d_j)
            tardiness_sum += tard
        else:
            tard = max(0, CT[j] - d_j)
            tardiness_sum += float(tard)
    # return tardiness in slots; multiply by delta_t to convert to hours if desired
    return tardiness_sum


def compute_total_power_profile(B: np.ndarray, level_powers: List[float]) -> np.ndarray:
    """
    B shape: (J, K, T)
    level_powers: list of length K containing power (kW) for each level index 0..K-1
    returns: array length T with total station power per slot (kW)
    """
    J, K, T = B.shape
    power_per_level = np.array(level_powers).reshape((K, 1))  # (K,1)
    # compute per EV per slot power: (J,K,T) * (K,1) sum over K -> (J,T)
    ev_power = np.tensordot(B, level_powers, axes=([1],[0]))  # (J,T)
    total_power = ev_power.sum(axis=0)  # (T,)
    return total_power


def compute_peak_power(B: np.ndarray, level_powers: List[float]) -> float:
    """
    f2 = max_t total_power(t)
    """
    total_power = compute_total_power_profile(B, level_powers)
    return float(total_power.max()) if total_power.size > 0 else 0.0

def objective_fn(X: np.ndarray, B: np.ndarray, params: Dict[str, Any],
                      w_tardiness: float = 1.0, w_peak: float = 1.0) -> float:
    """
    Combined objective: weighted sum of total tardiness (f1, in slots)
    and peak power (f2, in kW). Lower is better.
    """
    f1 = compute_total_tardiness(X, params)
    f2 = compute_peak_power(B, params["level_powers"])
    return w_tardiness * f1 + w_peak * f2

def check_spot_capacity(X: np.ndarray, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verify spot occupancy constraints: at most one EV per spot per slot,
    and at most S_i occupancies per line per slot (this latter condition is redundant
    given per-spot limit but we check both).
    Returns (valid, list_of_violations)
    """
    violations = []
    J, L, Smax, T = X.shape
    S_i = params["spots_per_line"]  # list length L
    # per-spot check
    for i in range(L):
        for s in range(Smax):
            for t in range(T):
                occ = int(X[:, i, s, t].sum())
                if occ > 1:
                    violations.append(f"Spot overbooked: line {i}, spot {s}, slot {t}, occ={occ}")
    # per-line capacity
    for i in range(L):
        for t in range(T):
            occ_line = int(X[:, i, :S_i[i], t].sum())
            if occ_line > S_i[i]:
                violations.append(f"Line capacity exceeded: line {i}, slot {t}, occ={occ_line}, S_i={S_i[i]}")
    valid = len(violations) == 0
    return valid, violations

def check_station_power(B: np.ndarray, level_powers: List[float], P_max: Optional[float]) -> Tuple[bool, List[str]]:
    """
    Checks if at any slot total power exceeds P_max (if P_max provided).
    """
    violations = []
    if P_max is None:
        return True, violations
    total_power = compute_total_power_profile(B, level_powers)
    for t, p in enumerate(total_power.tolist()):
        if p > P_max + 1e-9:
            violations.append(f"Station power limit exceeded at slot {t}: {p:.3f} > {P_max:.3f}")
    return (len(violations) == 0), violations