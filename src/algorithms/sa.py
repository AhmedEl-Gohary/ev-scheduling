# simulated_annealing.py
import math
import random
import time
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

from schedule_generator import make_neighbor_from_XB
from eval import compute_total_tardiness, compute_peak_power, check_spot_capacity, check_station_power, objective_fn




def simulated_annealing(X0: np.ndarray, B0: np.ndarray, params: Dict[str, Any],
                        T0: float = 1.0, Tf: float = 1e-3,
                        imax: int = 200, nT: int = 50,
                        rng_seed: Optional[int] = 42,
                        ) -> Tuple[np.ndarray, np.ndarray, list[Any]]:
    rng = random.Random(rng_seed)
    np_rng = np.random.RandomState(rng_seed)
    states = []

    # initial
    X_cur, B_cur = X0.copy(), B0.copy()
    f_cur = objective_fn(X_cur, B_cur, params)
    X_best, B_best = X_cur.copy(), B_cur.copy()
    f_best = f_cur

    Tcur = T0

    states.append({
        'iteration': 0,
        'temperature': Tcur,
        'X': X_cur.tolist(),
        'B': B_cur.tolist(),
        'objective': f_cur,
        'best_objective': f_best,
        'tardiness': compute_total_tardiness(X_cur, params),
        'peak_power': compute_peak_power(B_cur, params["level_powers"])
    })

    # cooling schedule factor (geometric): T_{k+1} = alpha * T_k
    if imax <= 1:
        alpha = 0.9
    else:
        alpha = (Tf / T0) ** (1.0 / max(1, imax - 1))

    history = []
    Tcur = T0
    start_time = time.time()

    # Counters for global stats
    total_attempts = 0
    total_accepts = 0
    total_improvements = 0

    for i in range(imax):
        if Tcur <= Tf:
            break

        temp_attempts = 0
        temp_accepts = 0
        temp_improvements = 0

        for k in range(nT):
            total_attempts += 1
            temp_attempts += 1

            # generate neighbor
            Xn, Bn, assign_n, valid_flag, violations = make_neighbor_from_XB(
                X_cur, B_cur, params
            )
            if not valid_flag:
                # skip invalid neighbor (counts as an attempt but not an accept)
                continue

            f_new = objective_fn(Xn, Bn, params)
            delta = f_new - f_cur

            accept = False
            if delta < 0:
                accept = True
            else:
                try:
                    p = math.exp(-delta / (Tcur if Tcur > 1e-300 else 1e-300))
                except OverflowError:
                    p = 0.0
                r = rng.random()
                if r < p:
                    accept = True

            if accept:
                total_accepts += 1
                temp_accepts += 1
                X_cur, B_cur, f_cur = Xn.copy(), Bn.copy(), f_new
                # improvement?
                if f_cur < f_best:
                    f_best = f_cur
                    X_best, B_best = X_cur.copy(), B_cur.copy()
                    total_improvements += 1
                    temp_improvements += 1

        states.append({
            'iteration': i + 1,
            'temperature': Tcur,
            'X': X_cur.tolist(),
            'B': B_cur.tolist(),
            'objective': f_cur,
            'best_objective': f_best,
            'tardiness': compute_total_tardiness(X_cur, params),
            'peak_power': compute_peak_power(B_cur, params["level_powers"])
        })
        # cool
        Tcur *= alpha

    return X_best, B_best, states

