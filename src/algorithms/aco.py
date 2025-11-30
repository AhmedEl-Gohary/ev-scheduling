import numpy as np
import random
import time
import math
from typing import Dict, Any, Tuple, List, Optional, Callable

from src.eval import objective_fn, compute_total_tardiness, compute_peak_power
from src.utilities import load_params

class AntColonyOptimizer:
    def __init__(self, params: Dict[str, Any],
                 n_ants: int = 20,
                 n_iterations: int = 50,
                 alpha: float = 1.0,  # Pheromone importance
                 beta: float = 2.0,  # Heuristic importance
                 rho: float = 0.1,  # Evaporation rate
                 Q: float = 100.0,  # Pheromone deposit factor
                 rng_seed: int = 42):

        self.params = params
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # Problem Dimensions
        self.J = len(params['evs'])  # Number of EVs
        self.T = params['time_slots']  # Time horizon
        self.K = len(params['level_powers'])  # Charging levels
        self.levels = params['level_powers']
        self.delta_t = params['delta_t']
        self.P_max = params.get('P_max', float('inf'))
        self.spots_per_line = params['spots_per_line']
        self.L = len(self.spots_per_line)
        self.S_max = max(self.spots_per_line)

        # Precompute duration (slots) required for each EV at each Power Level
        self.durations = np.zeros((self.J, self.K), dtype=int)
        for j, ev in enumerate(params['evs']):
            req_energy = ev['energy_required']
            for k, power in enumerate(self.levels):
                if power <= 1e-6:
                    self.durations[j, k] = 999999  # Invalid
                else:
                    slots = math.ceil(req_energy / (power * self.delta_t))
                    self.durations[j, k] = slots

        # Pheromone Matrix: [EV_index][Start_Time][Power_Level_Index]
        self.pheromones = np.ones((self.J, self.T, self.K)) * 0.1

        # Heuristic Matrix (Static): [EV_index][Start_Time][Power_Level_Index]
        self.heuristic_info = self._calculate_heuristics()

    def _calculate_heuristics(self) -> np.ndarray:
        eta = np.zeros((self.J, self.T, self.K))
        for j, ev in enumerate(self.params['evs']):
            d_j = ev['departure_slot']
            arr_j = ev['arrival_slot']
            for t in range(self.T):
                if t < arr_j: continue
                for k in range(self.K):
                    dur = self.durations[j, k]
                    if dur > self.T: continue
                    completion_time = t + dur
                    tardiness = max(0, completion_time - d_j)
                    power_cost = self.levels[k] / (max(self.levels) if max(self.levels) > 0 else 1)
                    horizon_penalty = 1000 if completion_time > self.T else 0

                    # Heuristic: Prefer On-Time > Low Power
                    cost_val = (10.0 * tardiness) + (1.0 * power_cost) + horizon_penalty
                    eta[j, t, k] = 1.0 / (cost_val + 1.0)
        return eta

    def _construct_solution(self) -> Tuple[np.ndarray, np.ndarray, float]:
        X = np.zeros((self.J, self.L, self.S_max, self.T), dtype=int)
        B = np.zeros((self.J, self.K, self.T), dtype=int)
        usage_lines = np.zeros((self.L, self.T), dtype=int)
        usage_power = np.zeros(self.T, dtype=float)

        # Iterate through EVs (Layers)
        for j in range(self.J):
            ev = self.params['evs'][j]
            arr_j = ev['arrival_slot']

            candidates = []
            probs = []

            for t_start in range(arr_j, self.T):
                for k in range(self.K):
                    dur = self.durations[j, k]
                    t_end = t_start + dur

                    if t_end > self.T: continue

                    # A. Station Power Check
                    power_req = self.levels[k]
                    if self.P_max is not None:
                        projected_load = usage_power[t_start:t_end] + power_req
                        if np.any(projected_load > self.P_max):
                            continue

                            # B. Line/Spot Availability Check
                    valid_line_idx = -1
                    for line_idx in range(self.L):
                        cap = self.spots_per_line[line_idx]
                        if np.all(usage_lines[line_idx, t_start:t_end] < cap):
                            valid_line_idx = line_idx
                            break

                    if valid_line_idx == -1: continue

                    tau = self.pheromones[j, t_start, k]
                    eta = self.heuristic_info[j, t_start, k]
                    p = (tau ** self.alpha) * (eta ** self.beta)

                    candidates.append((t_start, k, valid_line_idx))
                    probs.append(p)

            if not candidates: continue

            probs = np.array(probs)
            if probs.sum() == 0: probs = np.ones(len(probs))
            probs = probs / probs.sum()

            choice_idx = np.random.choice(len(candidates), p=probs)
            chosen_t, chosen_k, chosen_line = candidates[choice_idx]
            chosen_end = chosen_t + self.durations[j, chosen_k]

            # Assign to specific spot
            assigned_spot = -1
            for s in range(self.spots_per_line[chosen_line]):
                if np.all(X[:, chosen_line, s, chosen_t:chosen_end].sum(axis=0) == 0):
                    assigned_spot = s
                    break

            if assigned_spot != -1:
                X[j, chosen_line, assigned_spot, chosen_t:chosen_end] = 1
                B[j, chosen_k, chosen_t:chosen_end] = 1
                usage_lines[chosen_line, chosen_t:chosen_end] += 1
                usage_power[chosen_t:chosen_end] += self.levels[chosen_k]

        fit = objective_fn(X, B, self.params)
        return X, B, fit

    def solve(self, progress_callback: Optional[Callable] = None):
        best_X = None
        best_B = None
        best_fitness = float('inf')
        history = []

        for it in range(self.n_iterations):
            iteration_solutions = []

            # 1. Construction
            for _ in range(self.n_ants):
                X_ant, B_ant, fit_ant = self._construct_solution()
                iteration_solutions.append((X_ant, B_ant, fit_ant))

                if fit_ant < best_fitness:
                    best_fitness = fit_ant
                    best_X = np.copy(X_ant)
                    best_B = np.copy(B_ant)

            # 2. Pheromone Update
            self.pheromones *= (1.0 - self.rho)  # Evaporation

            # Deposit on Iteration Best (or Global Best - using Global Best here for stability)
            # Find iteration best first
            iter_best = min(iteration_solutions, key=lambda x: x[2])
            iter_best_fitness = iter_best[2]

            # Hybrid strategy: Deposit on Global Best strongly
            if best_X is not None:
                deposit_amt = self.Q / (best_fitness + 1e-6)
                for j in range(self.J):
                    # Decode X to find start time
                    occ_t = best_X[j].sum(axis=(0, 1))
                    starts = np.where((occ_t[:-1] == 0) & (occ_t[1:] == 1))[0]
                    start_t = -1
                    if occ_t[0] == 1:
                        start_t = 0
                    elif starts.size > 0:
                        start_t = starts[0] + 1

                    if start_t != -1:
                        # Decode B to find level
                        k_indices = np.where(best_B[j, :, start_t] == 1)[0]
                        if k_indices.size > 0:
                            self.pheromones[j, start_t, k_indices[0]] += deposit_amt

            # Stats
            avg_fit = sum(sol[2] for sol in iteration_solutions) / self.n_ants

            # Structure state for history/callback
            current_state = {
                'iteration': it + 1,
                'best_fitness': float(best_fitness),
                'avg_fitness': float(avg_fit),
                'objective': float(iter_best_fitness),  # Current iteration best
                'X': iter_best[0].tolist(),  # Send iteration best for visualization
                'B': iter_best[1].tolist(),
                'tardiness': float(compute_total_tardiness(iter_best[0], self.params)),
                'peak_power': float(compute_peak_power(iter_best[1], self.params['level_powers']))
            }
            history.append(current_state)

            if progress_callback:
                progress_callback(it + 1, iter_best_fitness, best_fitness, avg_fit, current_state)

        # Final return needs to be compatible with other algos
        # Return Global Best
        final_state = history[-1]
        # Overwrite final state X/B with GLOBAL best for final result
        final_state['X'] = best_X.tolist() if best_X is not None else []
        final_state['B'] = best_B.tolist() if best_B is not None else []

        return best_X, best_B, history


def ant_colony_optimization(params: Dict[str, Any],
                            n_ants: int = 20,
                            n_iterations: int = 50,
                            alpha: float = 1.0,
                            beta: float = 2.0,
                            rho: float = 0.1,
                            Q: float = 100.0,
                            rng_seed: Optional[int] = 42,
                            progress_callback: Optional[Callable] = None
                            ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Wrapper function matching the signature expected by the App runner.
    """
    optimizer = AntColonyOptimizer(params, n_ants, n_iterations, alpha, beta, rho, Q, rng_seed)
    return optimizer.solve(progress_callback)