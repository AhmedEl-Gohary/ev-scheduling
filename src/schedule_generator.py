import math
import random
import copy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# ---------------------------
# Helpers (unchanged)
# ---------------------------
def compute_required_slots(Ereq: float, r_l: float, delta_t: float) -> int:
    if r_l <= 0 or delta_t <= 0:
        return 10**9
    return max(1, math.ceil(Ereq / (r_l * delta_t)))

def energy_delivered_by_pattern(level_pattern: List[int], params: Dict[str,Any]) -> float:
    delta_t = params['delta_t']
    level_powers = params['level_powers']
    energy = 0.0
    for ell in level_pattern:
        energy += level_powers[int(ell)] * delta_t
    return energy

def clamp_start_for_duration(start: int, duration: int, arrival: int, T: int) -> int:
    if duration <= 0:
        return -1
    earliest = max(0, arrival)
    latest = T - duration
    if latest < earliest:
        return -1
    return max(earliest, min(start, latest))

# ---------------------------
# Extract / build functions (unchanged)
# ---------------------------
def extract_assignments_from_XB(X: np.ndarray, B: np.ndarray, params: Dict[str,Any]) -> List[Dict[str,Any]]:
    J, L, Smax, T = X.shape
    assignments = []
    for j in range(J):
        occ = np.where(X[j] == 1)
        if occ[0].size == 0:
            assignments.append({'assigned': False})
            continue
        t_idxs = np.unique(occ[2])
        start = int(t_idxs.min())
        end = int(t_idxs.max()) + 1
        counts = {}
        for i,s,t in zip(*occ):
            counts[(int(i),int(s))] = counts.get((int(i),int(s)), 0) + 1
        (line, spot) = max(counts.items(), key=lambda kv: kv[1])[0]
        levels = []
        for t in range(start, end):
            lvl = np.where(B[j, :, t] == 1)[0]
            if lvl.size > 0:
                levels.append(int(lvl[0]))
            else:
                levels.append(0)
        assignments.append({
            'assigned': True,
            'start': start,
            'end': end,
            'line': int(line),
            'spot': int(spot),
            'levels': levels
        })
    return assignments

def build_XB_from_assignments(assignments: List[Dict[str,Any]], params: Dict[str,Any]) -> Tuple[np.ndarray, np.ndarray, bool, List[str]]:
    evs = params['evs']
    J = len(evs)
    L = len(params['spots_per_line'])
    Smax = max(params['spots_per_line'])
    K = len(params['level_powers'])
    T = params['time_slots']
    X = np.zeros((J, L, Smax, T), dtype=int)
    B = np.zeros((J, K, T), dtype=int)
    violations: List[str] = []
    station_power = np.zeros(T, dtype=float)
    delta_t = params['delta_t']
    level_powers = params['level_powers']

    for j, a in enumerate(assignments):
        if not a.get('assigned', False):
            violations.append(f"EV {j}: unassigned in candidate (all EVs must be assigned)")
            return X, B, False, violations
        start = int(a['start'])
        end = int(a['end'])
        if not (0 <= start < end <= T):
            violations.append(f"EV {j}: invalid window [{start},{end}) over horizon [0,{T})")
            return X, B, False, violations
        line = int(a['line'])
        spot = int(a['spot'])
        if not (0 <= line < L):
            violations.append(f"EV {j}: invalid line {line}")
            return X, B, False, violations
        if not (0 <= spot < params['spots_per_line'][line]):
            violations.append(f"EV {j}: invalid spot {spot} on line {line}")
            return X, B, False, violations
        dur = end - start
        levels = a.get('levels', None)
        if levels is None:
            violations.append(f"EV {j}: missing levels pattern")
            return X, B, False, violations
        if isinstance(levels, int):
            levels = [levels] * dur
        if len(levels) != dur:
            violations.append(f"EV {j}: levels length {len(levels)} != duration {dur}")
            return X, B, False, violations
        for ell in levels:
            if not (0 <= int(ell) < K):
                violations.append(f"EV {j}: invalid level {ell}")
                return X, B, False, violations
        arrival = evs[j].get('arrival_slot', 0)
        if start < arrival:
            violations.append(f"EV {j}: start {start} before arrival {arrival}")
            return X, B, False, violations
        X[j, line, spot, start:end] = 1
        for offset, ell in enumerate(levels):
            t = start + offset
            B[j, int(ell), t] = 1
            station_power[t] += level_powers[int(ell)]

    # per-spot exclusivity & line capacity
    for i in range(L):
        for s in range(params['spots_per_line'][i]):
            for t in range(T):
                occ = int(X[:, i, s, t].sum())
                if occ > 1:
                    violations.append(f"Spot overbooked: line {i}, spot {s}, slot {t}, occ={occ}")
    for i in range(L):
        for t in range(T):
            occ_line = int(X[:, i, :params['spots_per_line'][i], t].sum())
            if occ_line > params['spots_per_line'][i]:
                violations.append(f"Line capacity exceeded: line {i}, t={t}, occ={occ_line}, S_i={params['spots_per_line'][i]}")
    P_max = params.get('P_max', None)
    if P_max is not None:
        for t, p in enumerate(station_power):
            if p > P_max + 1e-9:
                violations.append(f"Station power exceeded at slot {t}: {p:.3f} > {P_max:.3f}")
    # energy requirement
    for j, a in enumerate(assignments):
        levels = a['levels']
        energy = energy_delivered_by_pattern(levels, params)
        if energy + 1e-9 < evs[j]['energy_required']:
            violations.append(f"EV {j}: energy delivered {energy:.3f} < required {evs[j]['energy_required']:.3f}")
    valid = len(violations) == 0
    return X, B, valid, violations

# ---------------------------
# Neighbor generator (updated: no unassign allowed)
# ---------------------------
def random_neighbor_assignments(assignments: List[Dict[str,Any]], params: Dict[str,Any],
                                max_tries: int = 400) -> Tuple[List[Dict[str,Any]], bool]:
    """
    Generate a feasible neighbor while ensuring all EVs remain assigned.
    Moves:
      - shift (move entire block)
      - change_level_slot
      - change_level_range
      - level_uniform
      - move_spot
      - swap_spots
    The function never sets an EV to unassigned; candidates containing any unassigned EV are rejected.
    """
    rng = random.Random()
    J = len(assignments)
    K = len(params['level_powers'])
    T = params['time_slots']
    spots_per_line = params['spots_per_line']
    evs = params['evs']

    for attempt in range(max_tries):
        new_a = copy.deepcopy(assignments)
        move_type = rng.choice(['shift', 'change_level_slot', 'change_level_range', 'level_uniform',
                                'move_spot', 'swap_spots'])
        j = rng.randrange(J)
        assigned_j = new_a[j].get('assigned', False)
        # print("Move Type:", move_type)
        # SHIFT
        if move_type == 'shift':
            if not assigned_j:
                continue
            dur = new_a[j]['end'] - new_a[j]['start']
            max_shift = max(1, T // 12)
            delta = rng.randint(-max_shift, max_shift)
            arrival = evs[j].get('arrival_slot', 0)
            proposed_start = new_a[j]['start'] + delta
            clamped = clamp_start_for_duration(proposed_start, dur, arrival, T)
            if clamped == -1:
                continue
            new_a[j]['start'] = clamped
            new_a[j]['end'] = clamped + dur

        # CHANGE ONE SLOT LEVEL
        elif move_type == 'change_level_slot':
            if not assigned_j:
                continue
            dur = new_a[j]['end'] - new_a[j]['start']
            if dur <= 0:
                continue
            idx = rng.randrange(dur)
            old = new_a[j]['levels'][idx]
            choices = [ell for ell in range(K) if ell != old]
            if not choices:
                continue
            new_a[j]['levels'][idx] = rng.choice(choices)

        # CHANGE RANGE LEVELS
        elif move_type == 'change_level_range':
            if not assigned_j:
                continue
            dur = new_a[j]['end'] - new_a[j]['start']
            if dur <= 1:
                continue
            a_idx = rng.randrange(dur)
            b_idx = rng.randrange(dur)
            lo = min(a_idx, b_idx)
            hi = max(a_idx, b_idx) + 1
            for pos in range(lo, hi):
                new_a[j]['levels'][pos] = rng.randrange(K)

        # SET WHOLE BLOCK TO ONE LEVEL
        elif move_type == 'level_uniform':
            if not assigned_j:
                continue
            new_level = rng.randrange(K)
            dur = new_a[j]['end'] - new_a[j]['start']
            new_a[j]['levels'] = [new_level] * dur

        # MOVE SPOT (line/spot)
        elif move_type == 'move_spot':
            if not assigned_j:
                continue
            i_candidates = list(range(len(spots_per_line)))
            rng.shuffle(i_candidates)
            chosen = None
            for i in i_candidates:
                if spots_per_line[i] <= 0:
                    continue
                s = rng.randrange(spots_per_line[i])
                if i == new_a[j]['line'] and s == new_a[j]['spot']:
                    continue
                chosen = (i, s)
                break
            if chosen is None:
                continue
            new_a[j]['line'], new_a[j]['spot'] = chosen

        # SWAP SPOTS BETWEEN TWO ASSIGNED EVS
        elif move_type == 'swap_spots':
            k = rng.randrange(J)
            if k == j:
                continue
            if not new_a[j].get('assigned', False) or not new_a[k].get('assigned', False):
                continue
            new_a[j]['line'], new_a[k]['line'] = new_a[k]['line'], new_a[j]['line']
            new_a[j]['spot'], new_a[k]['spot'] = new_a[k]['spot'], new_a[j]['spot']

        # Validate candidate globally and ensure no EV is unassigned
        # If any EV unassigned or validation fails, reject candidate and continue
        any_unassigned = any(not a.get('assigned', False) for a in new_a)
        if any_unassigned:
            continue
        X_new, B_new, valid, viol = build_XB_from_assignments(new_a, params)
        if valid:
            return new_a, True
        # otherwise continue attempts
    return assignments, False

# ---------------------------
# Wrapper: produce neighbor Xn,Bn (guaranteed all EVs assigned if returned valid)
# ---------------------------
def make_neighbor_from_XB(X: np.ndarray, B: np.ndarray, params: Dict[str,Any],
                          max_attempts: int = 10,
                          neighbor_tries: int = 400) -> Tuple[np.ndarray, np.ndarray, List[Dict[str,Any]], bool, List[str]]:
    orig_assign = extract_assignments_from_XB(X, B, params)
    # verify original all assigned & valid
    X0, B0, valid0, viol0 = build_XB_from_assignments(orig_assign, params)
    if not valid0:
        # original schedule invalid -> cannot safely generate neighbors
        return X0, B0, orig_assign, False, viol0
    for attempt in range(max_attempts):
        new_assign, ok = random_neighbor_assignments(orig_assign, params, max_tries=neighbor_tries)
        if not ok:
            continue
        Xn, Bn, valid, viol = build_XB_from_assignments(new_assign, params)
        # ensure all EVs assigned
        if not valid:
            continue
        # success
        return Xn, Bn, new_assign, True, []
    # fallback: return original if no neighbor found
    return X0, B0, orig_assign, True, []
