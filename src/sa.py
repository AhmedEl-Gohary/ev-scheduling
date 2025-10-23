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
                        w_tardiness: float = 1.0, w_peak: float = 1.0,
                        neighbor_max_attempts: int = 10,
                        rng_seed: Optional[int] = 42,
                        verbose: bool = False,
                        per_temp_verbose: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulated Annealing with verbose logging.

    New args:
      - verbose: prints top-level progress (default False)
      - per_temp_verbose: print per-temperature detailed stats (accepts, attempts, best-improvement)
    """
    rng = random.Random(rng_seed)
    np_rng = np.random.RandomState(rng_seed)

    # initial
    X_cur, B_cur = X0.copy(), B0.copy()
    f_cur = objective_fn(X_cur, B_cur, params)
    X_best, B_best = X_cur.copy(), B_cur.copy()
    f_best = f_cur

    # cooling schedule factor (geometric): T_{k+1} = alpha * T_k
    if imax <= 1:
        alpha = 0.9
    else:
        alpha = (Tf / T0) ** (1.0 / max(1, imax - 1))

    history = []
    Tcur = T0
    start_time = time.time()

    if verbose:
        print(f"[SA] start f = {f_cur:.6f}, T0={T0}, Tf={Tf}, alpha={alpha:.6g}, imax={imax}, nT={nT}")

    # Counters for global stats
    total_attempts = 0
    total_accepts = 0
    total_improvements = 0

    for i in range(imax):
        if Tcur <= Tf:
            if verbose:
                print(f"[SA] stopping because Tcur <= Tf (Tcur={Tcur:.6g} <= Tf={Tf})")
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
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"[SA] improvement: new best f={f_best:.6f} @ iter (temp={i},inner={k}), elapsed={elapsed:.1f}s")

            # optional inline debug (very verbose)
            if per_temp_verbose:
                print(f"  temp {i:03d} inner {k:04d} T={Tcur:.6g} f_cur={f_cur:.6f} f_new={f_new:.6f} delta={delta:.6f} accept={accept}")

        # log per temperature
        history.append({"iter": i, "T": Tcur, "f_cur": f_cur, "f_best": f_best,
                        "attempts": temp_attempts, "accepts": temp_accepts, "improvements": temp_improvements})

        # print per-temperature summary (if requested)
        if per_temp_verbose or verbose:
            elapsed = time.time() - start_time
            acc_rate = (temp_accepts / temp_attempts) if temp_attempts > 0 else 0.0
            print(f"[SA] temp {i:03d} T={Tcur:.6g} attempts={temp_attempts} accepts={temp_accepts} acc_rate={acc_rate:.3f} "
                  f"impr={temp_improvements} f_cur={f_cur:.6f} f_best={f_best:.6f} elapsed={elapsed:.1f}s")

        # cool
        Tcur *= alpha

    total_time = time.time() - start_time
    info = {
        "f_best": f_best,
        "f_final": f_cur,
        "history": history,
        "time_s": total_time,
        "T0": T0,
        "Tf": Tf,
        "alpha": alpha,
        "imax": imax,
        "nT": nT,
        "w_tardiness": w_tardiness,
        "w_peak": w_peak,
        "total_attempts": total_attempts,
        "total_accepts": total_accepts,
        "total_improvements": total_improvements
    }

    if verbose:
        print(f"[SA] finished best f={f_best:.6f}, final f={f_cur:.6f}, total_attempts={total_attempts}, "
              f"total_accepts={total_accepts}, total_improvements={total_improvements}, time={total_time:.2f}s")

    return X_best, B_best, info

