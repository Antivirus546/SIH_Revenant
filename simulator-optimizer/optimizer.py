"""
optimizer.py — NSGA-II plan optimizer + greedy baseline for UrbanHeat AI.

Team Revenant | SIH 2026 | IHSIH031

Implemented WITHOUT pymoo or any external optimization library — only
numpy + the simulator's public evaluation interface. Standard NSGA-II
components:

    - population initialization (uniform random in [0, 1]^4)
    - objective evaluation via InterventionSimulator.evaluate_plan_metrics
    - fast non-dominated sorting (with constraint domination for budgets)
    - crowding distance
    - binary tournament selection
    - simulated binary crossover (SBX)
    - polynomial mutation
    - generational loop with environmental selection
    - Pareto front extraction

Decision variables (4 intensities, each clamped to [0, 1]):
    [cool_roofs, green_cover, albedo_boost, water_bodies]

Objectives (all minimised internally):
    1. −population-weighted cooling (°C)      [maximise cooling]
    2. total cost (INR)                        [minimise cost]
    3. OPTIONAL: std of per-cell cooling (°C)  [minimise inequality]
       enable with objectives=("cooling", "cost", "inequality")

Optional budget constraint:
    A candidate whose cost exceeds budget_inr is treated as INFEASIBLE
    using Deb's constraint-domination rule (feasible beats infeasible;
    among infeasible, lower violation wins).

Cell targeting:
    Pass target_cells=None (all cells), an explicit list of cell ids,
    or use InterventionSimulator.select_hottest_cells(fraction) to focus
    on the hottest top-k% of cells.

HONESTY NOTE
------------
The optimizer explores SCENARIO space under ASSUMED coefficients (see
intervention_simulater.py). It does not validate physical outcomes.
The greedy baseline is included purely as a comparison reference — no
assumption is made about which method performs better; results decide.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from intervention_simulater import InterventionSimulator

INTERVENTION_NAMES = ["cool_roofs", "green_cover", "albedo_boost", "water_bodies"]
N_VARS = len(INTERVENTION_NAMES)
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0


# ----------------------------------------------------------------------
# Core NSGA-II primitives
# ----------------------------------------------------------------------

def _constraint_dominates(f1: np.ndarray, cv1: float,
                          f2: np.ndarray, cv2: float) -> bool:
    """Deb's constraint-domination rule.

    Both feasible  -> normal Pareto dominance.
    One feasible   -> the feasible one dominates.
    Both infeasible-> lower constraint violation dominates.
    """
    if cv1 <= 0 and cv2 <= 0:
        return bool(np.all(f1 <= f2) and np.any(f1 < f2))
    if cv1 <= 0:
        return True
    if cv2 <= 0:
        return False
    return cv1 < cv2


def fast_non_dominated_sort(F: np.ndarray,
                            CV: np.ndarray) -> Tuple[List[List[int]], np.ndarray]:
    """Return (fronts, ranks). fronts is a list of index lists."""
    n = F.shape[0]
    dominated_sets: List[List[int]] = [[] for _ in range(n)]
    domination_counts = np.zeros(n, dtype=int)

    for p in range(n):
        for q in range(p + 1, n):
            if _constraint_dominates(F[p], CV[p], F[q], CV[q]):
                dominated_sets[p].append(q)
                domination_counts[q] += 1
            elif _constraint_dominates(F[q], CV[q], F[p], CV[p]):
                dominated_sets[q].append(p)
                domination_counts[p] += 1

    current = [i for i in range(n) if domination_counts[i] == 0]
    fronts: List[List[int]] = []
    ranks = np.empty(n, dtype=int)
    rank = 0
    while current:
        fronts.append(current)
        nxt: List[int] = []
        for p in current:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    nxt.append(q)
        for i in current:
            ranks[i] = rank
        rank += 1
        current = nxt
    return fronts, ranks


def crowding_distance(F: np.ndarray, front: Sequence[int]) -> Dict[int, float]:
    """Crowding distance for one front (index -> distance)."""
    l = len(front)
    dist: Dict[int, float] = {i: 0.0 for i in front}
    if l <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist
    Ff = F[list(front)]
    for m in range(F.shape[1]):
        order = np.argsort(Ff[:, m])
        dist[front[order[0]]] = float("inf")
        dist[front[order[-1]]] = float("inf")
        rng = Ff[order[-1], m] - Ff[order[0], m]
        if rng <= 0:
            continue
        for k in range(1, l - 1):
            i = front[order[k]]
            if dist[i] != float("inf"):
                dist[i] += (Ff[order[k + 1], m] - Ff[order[k - 1], m]) / rng
    return dist


def tournament_select(ranks: np.ndarray,
                      crowd: Dict[int, float],
                      rng: np.random.Generator) -> int:
    """Binary tournament: lower rank wins; tie -> larger crowding."""
    a, b = rng.integers(0, len(ranks)), rng.integers(0, len(ranks))
    if ranks[a] < ranks[b]:
        return int(a)
    if ranks[b] < ranks[a]:
        return int(b)
    ca, cb = crowd.get(a, 0.0), crowd.get(b, 0.0)
    if ca > cb:
        return int(a)
    if cb > ca:
        return int(b)
    return int(a if rng.random() < 0.5 else b)


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float,
                  prob: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover (per-variable probability)."""
    c1, c2 = p1.copy(), p2.copy()
    for j in range(len(p1)):
        if rng.random() > prob:
            continue
        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
        c2[j] = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
    return np.clip(c1, LOWER_BOUND, UPPER_BOUND), np.clip(c2, LOWER_BOUND, UPPER_BOUND)


def polynomial_mutation(x: np.ndarray, eta: float, prob: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Polynomial mutation (per-variable probability)."""
    y = x.copy()
    for j in range(len(y)):
        if rng.random() > prob:
            continue
        u = rng.random()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
        y[j] = min(UPPER_BOUND, max(LOWER_BOUND, y[j] + delta))
    return y


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Container for NSGA-II output."""

    pareto_solutions: List[Dict[str, Any]] = field(default_factory=list)
    n_evaluations: int = 0
    objectives: Tuple[str, ...] = ("cooling", "cost")
    budget_inr: Optional[float] = None
    seed: int = 42


class NSGA2Optimizer:
    """NSGA-II over 4 intervention intensities using the simulator."""

    def __init__(
        self,
        simulator: InterventionSimulator,
        target_cells: Optional[List[str]] = None,
        objectives: Sequence[str] = ("cooling", "cost"),
        budget_inr: Optional[float] = None,
        pop_size: int = 60,
        n_generations: int = 120,
        crossover_prob: float = 0.9,
        crossover_eta: float = 15.0,
        mutation_eta: float = 20.0,
        seed: int = 42,
        verbose: bool = False,
    ):
        if simulator.mode not in ("heuristic", "ml"):
            raise ValueError("simulator mode must be 'heuristic' or 'ml'")
        unknown = [o for o in objectives if o not in ("cooling", "cost", "inequality")]
        if unknown:
            raise ValueError(f"Unknown objectives: {unknown}")
        self.simulator = simulator
        self.target_cells = list(target_cells) if target_cells is not None else None
        self.objectives = tuple(objectives)
        self.budget_inr = budget_inr
        self.pop_size = int(pop_size)
        self.n_generations = int(n_generations)
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.seed = seed
        self.verbose = verbose

        self._eval_cache: Dict[tuple, Tuple[np.ndarray, float, Dict[str, Any]]] = {}
        self.n_evaluations = 0

    # ------------------------------------------------------------------

    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Repair an infeasible candidate under the budget constraint.

        Total cost is EXACTLY LINEAR in the intensity vector
        (cost = Σ_k x_k · C_k), so a single proportional rescale makes
        any candidate feasible whenever the zero plan is feasible
        (which it always is, cost = 0). Without a budget this is a no-op.
        """
        if self.budget_inr is None or self.budget_inr <= 0:
            return x
        _, cv, metrics = self._evaluate_raw(x)
        cost = metrics["total_cost_inr"]
        if cost > self.budget_inr and cost > 0:
            x = np.clip(x * (self.budget_inr / cost), LOWER_BOUND, UPPER_BOUND)
        return x

    def _evaluate_raw(self, x: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Evaluate WITHOUT caching or repair (used by _repair itself)."""
        plan = {name: float(v) for name, v in zip(INTERVENTION_NAMES, x)}
        m = self.simulator.evaluate_plan_metrics(plan, target_cells=self.target_cells)

        f = [-m["pop_weighted_cooling"], m["total_cost_inr"]]
        if "inequality" in self.objectives:
            f.append(float(np.std(m["delta_per_cell"])))
        F = np.asarray(f, dtype=float)

        cv = 0.0
        if self.budget_inr is not None and self.budget_inr > 0:
            cv = max(0.0, m["total_cost_inr"] - self.budget_inr) / self.budget_inr

        metrics = {
            "plan": dict(plan),
            "pop_weighted_cooling": m["pop_weighted_cooling"],
            "mean_cooling": m["mean_cooling"],
            "total_cooling": m["total_cooling"],
            "max_cooling": m["max_cooling"],
            "total_cost_inr": m["total_cost_inr"],
            "cooling_per_lakh_inr": m["cooling_per_lakh_inr"],
            "inequality_std_c": float(np.std(m["delta_per_cell"])),
        }
        return F, cv, metrics

    def _evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        key = tuple(np.round(x, 6))
        if key in self._eval_cache:
            return self._eval_cache[key]

        F, cv, metrics = self._evaluate_raw(x)
        self.n_evaluations += 1
        out = (F, cv, metrics)
        self._eval_cache[key] = out
        return out

    def _solution_dict(self, x: np.ndarray, F: np.ndarray, cv: float,
                       metrics: Dict[str, Any]) -> Dict[str, Any]:
        sol = {
            "plan": dict(metrics["plan"]),
            "intensities": [float(v) for v in x],
            "cooling": metrics["pop_weighted_cooling"],
            "cost": metrics["total_cost_inr"],
            "mean_cooling": metrics["mean_cooling"],
            "total_cooling": metrics["total_cooling"],
            "max_cooling": metrics["max_cooling"],
            "cooling_per_lakh_inr": metrics["cooling_per_lakh_inr"],
            "inequality_std_c": metrics["inequality_std_c"],
            "constraint_violation": float(cv),
        }
        return sol

    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        rng = np.random.default_rng(self.seed)

        # ---- initial population --------------------------------------
        # Seed with known-feasible anchors (zero plan and a small plan) so
        # constraint-domination always has traction under tight budgets;
        # the rest of the population is uniform random. All individuals
        # are repaired to respect the budget when one is set.
        X = rng.uniform(LOWER_BOUND, UPPER_BOUND, size=(self.pop_size, N_VARS))
        X[0] = 0.0                                   # zero plan (always feasible)
        if self.pop_size > 1:
            X[1] = 0.05                              # small low-cost plan
        if self.budget_inr is not None:
            X = np.vstack([self._repair(row) for row in X])
        evals = [self._evaluate(x) for x in X]
        F = np.vstack([e[0] for e in evals])
        CV = np.array([e[1] for e in evals])

        # ---- generation loop ------------------------------------------
        for gen in range(self.n_generations):
            fronts, ranks = fast_non_dominated_sort(F, CV)
            crowd: Dict[int, float] = {}
            for front in fronts:
                crowd.update(crowding_distance(F, front))

            children: List[np.ndarray] = []
            while len(children) < self.pop_size:
                p1 = tournament_select(ranks, crowd, rng)
                p2 = tournament_select(ranks, crowd, rng)
                c1, c2 = sbx_crossover(X[p1], X[p2], self.crossover_eta,
                                       self.crossover_prob, rng)
                c1 = polynomial_mutation(c1, self.mutation_eta, 1.0 / N_VARS, rng)
                c2 = polynomial_mutation(c2, self.mutation_eta, 1.0 / N_VARS, rng)
                if self.budget_inr is not None:
                    c1 = self._repair(c1)
                    c2 = self._repair(c2)
                children.extend([c1, c2])
            children = children[: self.pop_size]

            child_evals = [self._evaluate(x) for x in children]
            Xc = np.vstack([X, np.vstack(children)])
            Fc = np.vstack([F, np.vstack([e[0] for e in child_evals])])
            CVc = np.concatenate([CV, np.array([e[1] for e in child_evals])])

            # ---- environmental selection ------------------------------
            fronts_c, ranks_c = fast_non_dominated_sort(Fc, CVc)
            selected: List[int] = []
            for front in fronts_c:
                if len(selected) + len(front) <= self.pop_size:
                    selected.extend(front)
                else:
                    cd = crowding_distance(Fc, front)
                    remaining = self.pop_size - len(selected)
                    sorted_front = sorted(front, key=lambda i: -cd[i])
                    selected.extend(sorted_front[:remaining])
                    break

            X = Xc[selected]
            F = Fc[selected]
            CV = CVc[selected]

            if self.verbose and (gen % 10 == 0 or gen == self.n_generations - 1):
                feas = CV <= 0
                best_cool = (-F[feas, 0].min()) if feas.any() else float("nan")
                print(f"[gen {gen:>3}] best feasible cooling so far: "
                      f"{best_cool:.3f} °C")

        # ---- extract Pareto set from ALL evaluated points --------------
        keys = list(self._eval_cache.keys())
        A = np.vstack([np.asarray(k, dtype=float) for k in keys])
        Fa = np.vstack([self._eval_cache[k][0] for k in keys])
        CVa = np.array([self._eval_cache[k][1] for k in keys])

        fronts_a, _ = fast_non_dominated_sort(Fa, CVa)
        first_front = fronts_a[0]

        solutions: List[Dict[str, Any]] = []
        seen_plans = set()
        for idx in first_front:
            x = A[idx]
            Fv = Fa[idx]
            cv = CVa[idx]
            metrics = self._eval_cache[keys[idx]][2]
            plan_key = tuple(round(v, 3) for v in x)
            if plan_key in seen_plans:
                continue
            seen_plans.add(plan_key)
            solutions.append(self._solution_dict(x, Fv, cv, metrics))

        solutions.sort(key=lambda s: s["cost"])
        return OptimizationResult(
            pareto_solutions=solutions,
            n_evaluations=self.n_evaluations,
            objectives=self.objectives,
            budget_inr=self.budget_inr,
            seed=self.seed,
        )


# ----------------------------------------------------------------------
# Greedy baseline (comparison reference ONLY — no assumed superiority)
# ----------------------------------------------------------------------

def greedy_baseline(
    simulator: InterventionSimulator,
    target_cells: Optional[List[str]] = None,
    budget_inr: Optional[float] = None,
    step: float = 0.05,
    max_iterations: int = 400,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Greedy budget-allocation baseline.

    Repeatedly adds `step` of intensity to whichever intervention gives
    the best marginal cooling-per-rupee, until no improvement remains
    (or the optional budget would be exceeded). This is a simple
    hill-climb on the same scenario model — NOT a claim of optimality.
    """
    plan: Dict[str, float] = {name: 0.0 for name in INTERVENTION_NAMES}
    current = simulator.evaluate_plan_metrics(plan, target_cells=target_cells)

    for _ in range(max_iterations):
        best: Optional[Tuple[float, str, Dict[str, float], Dict[str, Any]]] = None
        for name in INTERVENTION_NAMES:
            if plan[name] >= UPPER_BOUND - 1e-12:
                continue
            cand = dict(plan)
            cand[name] = min(UPPER_BOUND, plan[name] + step)
            m = simulator.evaluate_plan_metrics(cand, target_cells=target_cells)
            if budget_inr is not None and m["total_cost_inr"] > budget_inr:
                continue
            d_cool = m["pop_weighted_cooling"] - current["pop_weighted_cooling"]
            d_cost = m["total_cost_inr"] - current["total_cost_inr"]
            if d_cool <= 0:
                continue
            eff = d_cool / (d_cost if d_cost > 0 else 1e-9)
            if best is None or eff > best[0]:
                best = (eff, name, cand, m)
        if best is None:
            break
        plan = best[2]
        current = best[3]

    summary = {
        "plan": dict(plan),
        "pop_weighted_cooling": current["pop_weighted_cooling"],
        "mean_cooling": current["mean_cooling"],
        "total_cooling": current["total_cooling"],
        "max_cooling": current["max_cooling"],
        "total_cost_inr": current["total_cost_inr"],
        "cooling_per_lakh_inr": current["cooling_per_lakh_inr"],
        "inequality_std_c": float(np.std(current["delta_per_cell"])),
    }
    return plan, summary


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _run_self_tests() -> bool:
    print("=== optimizer.py self-tests ===\n")

    from intervention_simulater import load_ward1_grid


    # Load real data from the Data folder
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent
    csv_path = root_dir / "Data" / "ward1_processed.csv"
    
    grid = load_ward1_grid(csv_path)

    
    sim = InterventionSimulator(grid)
    hot = sim.select_hottest_cells(0.3)
    print(f"Target cells: hottest 30% → {len(hot)} cells\n")

    checks: List[tuple] = []

    # --- NSGA-II run ---------------------------------------------------
    opt = NSGA2Optimizer(
        sim, target_cells=hot, pop_size=24, n_generations=25, seed=42
    )
    result = opt.optimize()
    sols = result.pareto_solutions

    checks.append(("Pareto set is non-empty", len(sols) > 0))
    checks.append((
        "all intensities within [0, 1]",
        all(all(0.0 <= v <= 1.0 for v in s["intensities"]) for s in sols),
    ))
    checks.append((
        "returned solutions are mutually non-dominated",
        _verify_mutually_nondominated(sols),
    ))

    print(f"NSGA-II: {len(sols)} Pareto solutions after "
          f"{result.n_evaluations} evaluations\n")
    _print_solution_table(sols[:8], title="Cheapest 8 Pareto solutions")

    # --- Budget-constrained run ----------------------------------------
    budget = 5.0e7  # ₹5 crore
    opt_b = NSGA2Optimizer(
        sim, target_cells=hot, pop_size=20, n_generations=15,
        budget_inr=budget, seed=7,
    )
    res_b = opt_b.optimize()
    checks.append((
        f"budget constraint respected (≤ ₹{budget/1e7:.0f} Cr)",
        all(s["cost"] <= budget * (1 + 1e-9) for s in res_b.pareto_solutions),
    ))

    # --- Greedy baseline -------------------------------------------------
    g_plan, g_sum = greedy_baseline(sim, target_cells=hot)
    print("\nGreedy baseline:")
    print(f"  plan                 : {{{', '.join(f'{k}: {v:.2f}' for k, v in g_plan.items())}}}")
    print(f"  pop-weighted cooling : {g_sum['pop_weighted_cooling']:.3f} °C")
    print(f"  total cost           : ₹{g_sum['total_cost_inr']/1e7:.2f} Cr")

    # Neutral comparison: nearest-cost Pareto neighbour
    if sols:
        nearest = min(sols, key=lambda s: abs(s["cost"] - g_sum["total_cost_inr"]))
        diff = g_sum["pop_weighted_cooling"] - nearest["cooling"]
        better = "greedy" if diff > 0 else "NSGA-II"
        print(f"\nNeutral comparison at ≈₹{nearest['cost']/1e7:.2f} Cr: "
              f"{better} achieves more pop-weighted cooling "
              f"(Δ = {abs(diff):.3f} °C)")
        print("(No assumption is made about which method should win.)")

    print()
    all_ok = True
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return all_ok


def _verify_mutually_nondominated(sols: List[Dict[str, Any]]) -> bool:
    """Check no returned solution dominates another (feasible ones only)."""
    feas = [(np.array([-s["cooling"], s["cost"]] +
                      ([s["inequality_std_c"]] if "inequality_std_c" in s else [])),
             s["constraint_violation"])
            for s in sols]
    for i in range(len(feas)):
        for j in range(len(feas)):
            if i == j:
                continue
            fi, ci = feas[i]
            fj, cj = feas[j]
            if _constraint_dominates(fi, ci, fj, cj):
                return False
    return True


def _print_solution_table(sols: List[Dict[str, Any]], title: str) -> None:
    print(title)
    header = (f"{'cost ₹Cr':>9} | {'cool °C':>7} | {'mean °C':>7} | {'cool/lakh':>9} | "
              f"{'std °C':>6} | plan (cr/gc/ab/wb)")
    print("-" * len(header))
    print(header)
    for s in sols:
        p = s["plan"]
        plan_str = "/".join(f"{p[n]:.2f}" for n in INTERVENTION_NAMES)
        print(f"{s['cost']/1e7:>9.2f} | {s['cooling']:>7.3f} | "
              f"{s['mean_cooling']:>7.3f} | {s['cooling_per_lakh_inr']:>9.4f} | "
              f"{s['inequality_std_c']:>6.3f} | {plan_str}")


if __name__ == "__main__":
    ok = _run_self_tests()
    raise SystemExit(0 if ok else 1)