import json
import csv
import sys
from eval import *

# ---------------------------
# IO / runner
# ---------------------------
def load_params(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        data = json.load(f)
    # validation & normalization
    # canonicalize ev fields to include arrival_slot and departure_slot 0-based
    for ev in data["evs"]:
        # accept either 'arrival' or 'arrival_slot'
        if "arrival_slot" not in ev:
            ev["arrival_slot"] = ev.get("arrival", 0)
        if "departure_slot" not in ev:
            ev["departure_slot"] = ev.get("departure", data["time_slots"] - 1)
    # spots per line list
    if "spots_per_line" not in data:
        # if user supplied "lines" as dict mapping name->count, convert
        if "lines" in data:
            if isinstance(data["lines"], dict):
                data["spots_per_line"] = [int(v) for v in data["lines"].values()]
            elif isinstance(data["lines"], list):
                data["spots_per_line"] = [int(v) for v in data["lines"]]
        else:
            raise ValueError("Input JSON must include 'spots_per_line' or 'lines'")
    # level powers: list
    if "level_powers" not in data:
        if "charging_levels" in data:
            # expect mapping of str->power
            data["level_powers"] = [float(v) for k,v in sorted(data["charging_levels"].items(), key=lambda x:int(x[0]))]
        else:
            raise ValueError("Input JSON must include 'level_powers' or 'charging_levels'")
    # delta_t
    if "delta_t" not in data:
        data["delta_t"] = float(1.0)
    return data


def pretty_print_evaluation(f1: float, f2: float, constraint_violations: Dict[str, List[str]]):
    print("===== Evaluation Results =====")
    print(f"Total tardiness (f1) [in slots]: {f1:.3f}")
    print(f"Peak power (f2) [kW]: {f2:.3f}")
    print("")
    print("Constraint checks:")
    for k, v in constraint_violations.items():
        if not v:
            print(f" - {k}: OK")
        else:
            print(f" - {k}: VIOLATIONS ({len(v)})")
            for msg in v[:5]:
                print("    *", msg)
            if len(v) > 5:
                print("    ... (more)")

def pretty_print_schedule(X: np.ndarray, B: np.ndarray, params: Dict[str, Any],
                          max_cols: int = 200):
    """
    Pretty-print the schedule grid.
    - X shape: (J, L, Smax, T)
    - B shape: (J, K, T)
    Prints for each line i:
      Time header: slot indices 0..T-1
      One row per spot s (only up to spots_per_line[i])
      Each cell: '.' if free, otherwise 'E{j}:L{ell}'
    If T is huge, you may want to print chunks or use pretty_print_schedule_compact().
    """
    J, L, Smax, T = X.shape
    S_i = params["spots_per_line"]
    K = len(params["level_powers"])

    # compute width for each cell
    # cell examples: '.' or 'E12:L2' -> compute pad
    max_j_digits = max(2, len(str(J-1)))
    max_k_digits = max(1, len(str(K-1)))
    cell_example_len = 2 + max_j_digits + 3 + max_k_digits  # E + j + :L + k
    cell_width = max(3, cell_example_len)  # at least 3 for '.'
    slot_fmt = f"{{:^{cell_width}}}"

    # header
    print("\n=== Schedule per line / spot ===")
    for i in range(L):
        print(f"\n-- Line {(i+1)} (spots = {S_i[i]}) --")
        # print time header in one line (chunk if too many slots)
        if T <= max_cols:
            header = "Time     ".ljust(6) + " ".join(slot_fmt.format(t+1) for t in range(T))
            print(header)
        else:
            # show a note; user can use compact or chunking
            print(f"Time slots: 0 .. {T-1}  (total {T} slots) -- use compact view if this is too wide)")

        # per spot rows
        for s in range(S_i[i]):
            row_cells = []
            for t in range(T):
                occ = np.where(X[:, i, s, t] == 1)[0]
                if occ.size == 0:
                    cell = "."
                else:
                    j = int(occ[0])  # assume at most one as validity check
                    # find level active for this EV at this time (B[j,:,t])
                    level_idx = np.where(B[j, :, t] == 1)[0]
                    if level_idx.size > 0:
                        ell = int(level_idx[0]) + 1
                        cell = f"E{(j+1)}:L{ell}"
                    else:
                        cell = f"E{(j+1)}:L?"
                row_cells.append(slot_fmt.format(cell))
            # print row (or chunk if wide)
            if T <= max_cols:
                print(f"Spot {(s+1):<3} " + " ".join(row_cells))
            else:
                # print compact intervals if row very wide
                print(f"Spot {(s+1):<3} " + pretty_print_schedule_compact_row(row_cells, cell_width))
    print("=== End schedule ===\n")


def pretty_print_schedule_compact_row(row_cells: List[str], cell_width: int) -> str:
    """
    Helper to collapse consecutive identical cells into ranges.
    row_cells: list of already formatted cell strings (for visual alignment)
    Returns a single-line compact representation like:
      [0-2]:.  [3-6]:E1:L0  [7]:.  [8-11]:E4:L2
    """
    # strip padding and get raw labels
    raw = [c.strip() for c in row_cells]
    parts = []
    start = 0
    cur = raw[0] if raw else "."
    for idx in range(1, len(raw)):
        if raw[idx] != cur:
            # close current run
            if start == idx - 1:
                rng = f"[{start}]"
            else:
                rng = f"[{start}-{idx-1}]"
            parts.append(f"{rng}:{cur}")
            start = idx
            cur = raw[idx]
    # close last
    if raw:
        if start == len(raw) - 1:
            rng = f"[{start}]"
        else:
            rng = f"[{start}-{len(raw)-1}]"
        parts.append(f"{rng}:{cur}")
    return "  ".join(parts)

def print_sa_summary(info: Dict[str, Any]):
    print("\n=== SA Summary ===")
    print(f"Best objective (f_best): {info['f_best']:.6f}")
    print(f"Final objective (f_final): {info['f_final']:.6f}")
    print(f"Time (s): {info['time_s']:.2f}")
    print(f"T0={info['T0']}, Tf={info['Tf']}, imax={info['imax']}, nT={info['nT']}")
    print(f"Total attempts: {info.get('total_attempts',0)}, accepts: {info.get('total_accepts',0)}, improvements: {info.get('total_improvements',0)}")
    # print last few temperature rows
    hist = info.get("history", [])
    print("\nLast temperatures:")
    for row in hist[-6:]:
        print(f"  iter={row['iter']:3d} T={row['T']:.6g} attempts={row.get('attempts',0):3d} accepts={row.get('accepts',0):3d} "
              f"impr={row.get('improvements',0):2d} f_cur={row['f_cur']:.6f} f_best={row['f_best']:.6f}")
    print("===================\n")


def export_output_to_txt(X: np.ndarray, B: np.ndarray, params: Dict[str, Any]):
    with open("schedule_output.txt", "w") as f:
        # send all print() calls to this file
        sys.stdout = f
        pretty_print_schedule(X, B, params)
        f1 = compute_total_tardiness(X, params)
        f2 = compute_peak_power(B, params["level_powers"])
        valid_spot, v_spot = check_spot_capacity(X, params)
        valid_power, v_power = check_station_power(B, params["level_powers"], params.get("P_max"))
        violations = {"spot_capacity": v_spot, "station_power": v_power}
        pretty_print_evaluation(f1, f2, violations)
        sys.stdout = sys.__stdout__  # restore normal printing

def export_schedule_to_csv(X: np.ndarray, B: np.ndarray, params: Dict[str, Any],
                           path: str = "schedule.csv"):
    """
    Write a CSV with columns:
      EV,line,spot,slot,level,power(kW)
    One row per occupied time slot.
    """
    J, L, Smax, T = X.shape
    S_i = params["spots_per_line"]
    level_powers = params["level_powers"]

    rows = []
    for j in range(J):
        for i in range(L):
            for s in range(S_i[i]):
                for t in range(T):
                    if X[j, i, s, t] == 1:
                        level_idx = np.where(B[j, :, t] == 1)[0]
                        level = int(level_idx[0]) if level_idx.size > 0 else None
                        power = level_powers[level] if level is not None else 0
                        rows.append({
                            "EV": j + 1,
                            "line": i + 1,
                            "spot": s + 1,
                            "slot": t + 1,
                            "level": level + 1,
                            "power(kW)": power
                        })
    # write CSV
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)