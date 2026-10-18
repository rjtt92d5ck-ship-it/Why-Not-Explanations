from __future__ import annotations

import json
import itertools
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


Family = tuple[frozenset[int], ...]


@dataclass(frozen=True)
class ExplicitInstance:
    n: int
    conflicts: Family
    provenance: Family
    target: tuple

    @classmethod
    def from_artifact(cls, path: Path, target: tuple) -> "ExplicitInstance":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_data(data, target)

    @classmethod
    def from_data(cls, data: dict, target: tuple) -> "ExplicitInstance":
        if data.get("status") != "complete" or data.get("format_version") != 2:
            raise ValueError("expected a complete version-2 materialization")
        entries = [entry for entry in data["answers"] if tuple(entry["answer"]) == target]
        if len(entries) != 1:
            raise ValueError(f"target {target!r} is not exactly one ordinary answer")
        n = len(data["facts"])
        if [fact["id"] for fact in data["facts"]] != list(range(n)):
            raise ValueError("fact IDs must be consecutive from zero")
        conflicts = tuple(frozenset(edge) for edge in data["mis"])
        provenance = tuple(
            frozenset(support) for support in entries[0]["provenance_sets"]
        )
        for family in (conflicts, provenance):
            if any(not support or min(support) < 0 or max(support) >= n for support in family):
                raise ValueError("invalid fact ID in materialized family")
            if len(set(family)) != len(family):
                raise ValueError("duplicate materialized sets")
        return cls(n, conflicts, provenance, target)

    def consistent_provenance(self, indexed: bool = False) -> Family:
        if indexed:
            # For a fixed small CQ support, hash lookup of possible contained
            # MISs avoids scanning the entire conflict family for every support.
            # Fall back to the old scan when subset enumeration would cost more.
            by_rank: dict[int, set[frozenset[int]]] = defaultdict(set)
            for conflict in self.conflicts:
                by_rank[len(conflict)].add(conflict)
            kept = []
            for support in self.provenance:
                ranks = [r for r in by_rank if r <= len(support)]
                if not ranks:
                    kept.append(support)
                    continue
                enumerate_subsets = (len(support) <= 16 and
                    sum(math.comb(len(support), r) for r in ranks) <= len(self.conflicts))
                if enumerate_subsets:
                    inconsistent = any(frozenset(s) in by_rank[r]
                        for r in ranks for s in itertools.combinations(sorted(support), r))
                else:
                    inconsistent = any(c <= support for r in ranks for c in by_rank[r])
                if not inconsistent:
                    kept.append(support)
            return tuple(kept)
        return tuple(
            support for support in self.provenance
            if not any(conflict <= support for conflict in self.conflicts)
        )


class PreparedInstances:
    """Validate shared materialization once for a batch of answer targets."""

    def __init__(self, data: dict):
        if data.get("status") != "complete" or data.get("format_version") != 2:
            raise ValueError("expected a complete version-2 materialization")
        self.n = len(data["facts"])
        if any(fact["id"] != i for i, fact in enumerate(data["facts"])):
            raise ValueError("fact IDs must be consecutive from zero")
        self.conflicts = tuple(frozenset(edge) for edge in data["mis"])
        if any(not edge or min(edge) < 0 or max(edge) >= self.n
               for edge in self.conflicts):
            raise ValueError("invalid conflict fact ID")
        if len(set(self.conflicts)) != len(self.conflicts):
            raise ValueError("duplicate materialized conflicts")
        self.answers = {tuple(entry["answer"]): entry for entry in data["answers"]}
        if len(self.answers) != len(data["answers"]):
            raise ValueError("duplicate ordinary answers")

    def instance(self, target: tuple) -> ExplicitInstance:
        try:
            entry = self.answers[target]
        except KeyError as exc:
            raise ValueError(f"target {target!r} is not an ordinary answer") from exc
        provenance = tuple(frozenset(support) for support in entry["provenance_sets"])
        if any(not support or min(support) < 0 or max(support) >= self.n
               for support in provenance):
            raise ValueError("invalid provenance fact ID")
        if len(set(provenance)) != len(provenance):
            raise ValueError("duplicate materialized provenance sets")
        return ExplicitInstance(self.n, self.conflicts, provenance, target)


@dataclass
class SolveResult:
    semantics: str
    target: tuple
    status: str
    deletions: list[int] = field(default_factory=list)
    objective_upper_bound: int | None = None
    objective_lower_bound: int | None = None
    elapsed_seconds: float = 0.0
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "semantics": self.semantics,
            "target": list(self.target),
            "status": self.status,
            "deletions": self.deletions,
            "objective_upper_bound": self.objective_upper_bound,
            "objective_lower_bound": self.objective_lower_bound,
            "elapsed_seconds": self.elapsed_seconds,
            "details": self.details,
        }


def iar_accepts(instance: ExplicitInstance, deletions: set[int], indexed: bool = False) -> bool:
    if indexed:
        # IAR excludes exactly the deleted tuples and vertices of surviving
        # MISs. This builds no set proportional to the irrelevant universe.
        unavailable = set(deletions)
        for conflict in instance.conflicts:
            if not conflict & deletions:
                unavailable.update(conflict)
        return any(not support & unavailable for support in instance.provenance)
    # A support belongs to the repair intersection exactly when it is kept
    # and no still-active conflict touches any of its members. This avoids
    # building an O(|D|) set for each target in large sparse workloads.
    return any(
        not support & deletions
        and all(not conflict & support or bool(conflict & deletions)
                for conflict in instance.conflicts)
        for support in instance.provenance
    )


def _minimal_edges(edges: set[frozenset[int]]) -> tuple[frozenset[int], ...]:
    minimal: list[frozenset[int]] = []
    for edge in sorted(edges, key=lambda s: (len(s), tuple(sorted(s)))):
        if not any(previous <= edge for previous in minimal):
            minimal.append(edge)
    return tuple(minimal)


class _TransversalTimeout(Exception):
    def __init__(self, incumbent: set[int], nodes: int):
        self.incumbent = incumbent.copy()
        self.nodes = nodes
        super().__init__("transversal search timed out with a feasible incumbent")


def _minimum_transversal(
    edges: tuple[frozenset[int], ...], deadline: float
) -> tuple[set[int], int]:
    if not edges:
        return set(), 0
    if any(not edge for edge in edges):
        raise ValueError("inconsistent provenance has an empty residual edge")
    edges = _minimal_edges(set(edges))
    greedy: set[int] = set()
    uncovered = list(edges)
    while uncovered:
        frequency = Counter(vertex for edge in uncovered for vertex in edge)
        vertex = max(frequency, key=lambda v: (frequency[v], -v))
        greedy.add(vertex)
        uncovered = [edge for edge in uncovered if vertex not in edge]
    best = greedy
    nodes = 0
    seen_cost: dict[tuple[frozenset[int], ...], int] = {}

    def search(remaining: tuple[frozenset[int], ...], chosen: frozenset[int]) -> None:
        nonlocal best, nodes
        nodes += 1
        if nodes % 128 == 0 and time.monotonic() >= deadline:
            raise _TransversalTimeout(best, nodes)
        forced = {next(iter(edge)) for edge in remaining if len(edge) == 1}
        if forced:
            chosen = chosen | forced
            remaining = tuple(edge for edge in remaining if not edge & forced)
        if len(chosen) >= len(best):
            return
        if not remaining:
            best = set(chosen)
            return
        previous = seen_cost.get(remaining)
        if previous is not None and previous <= len(chosen):
            return
        seen_cost[remaining] = len(chosen)
        used: set[int] = set()
        packing = 0
        for edge in sorted(remaining, key=len):
            if not edge & used:
                used.update(edge)
                packing += 1
        if len(chosen) + packing >= len(best):
            return
        selected = min(remaining, key=lambda edge: (len(edge), tuple(sorted(edge))))
        frequency = Counter(vertex for edge in remaining for vertex in edge)
        for vertex in sorted(selected, key=lambda v: (-frequency[v], v)):
            search(
                tuple(edge for edge in remaining if vertex not in edge),
                chosen | {vertex},
            )

    search(edges, frozenset())
    if time.monotonic() >= deadline:
        raise _TransversalTimeout(best, nodes)
    return best, nodes


def solve_iar(instance: ExplicitInstance, time_limit: float = 60.0) -> SolveResult:
    started = time.monotonic()
    deadline = started + time_limit

    def result(status: str, best: set[int] | None, details: dict) -> SolveResult:
        optimal = status == "OPTIMAL"
        return SolveResult(
            "IAR", instance.target, status,
            sorted(best) if best is not None else [],
            len(best) if best is not None else None,
            len(best) if optimal and best is not None else None,
            time.monotonic() - started, details,
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", None, {})
    if iar_accepts(instance, set()):
        return result("INVALID_ALREADY_ACCEPTED", set(), {})
    candidates = instance.consistent_provenance()
    if not candidates:
        return result("INFEASIBLE", None, {"consistent_provenance": 0})
    best: set[int] | None = None
    nodes = 0
    processed = 0
    for support in candidates:
        if time.monotonic() >= deadline:
            return result("FEASIBLE_NOT_PROVEN" if best is not None else "TIMEOUT", best,
                          {"processed_provenance": processed, "search_nodes": nodes})
        residual = tuple(
            conflict - support
            for conflict in instance.conflicts
            if conflict & support
        )
        if not residual:
            raise AssertionError("IAR precondition/core and residual disagreement")
        try:
            candidate, count = _minimum_transversal(residual, deadline)
        except _TransversalTimeout as exc:
            nodes += exc.nodes
            if best is None or len(exc.incumbent) < len(best):
                best = exc.incumbent
            if not iar_accepts(instance, best):
                raise AssertionError("timeout incumbent is not a valid IAR explanation")
            return result("FEASIBLE_NOT_PROVEN", best,
                          {"processed_provenance": processed,
                           "search_nodes": nodes, "timeout_during_provenance": True})
        processed += 1
        nodes += count
        if best is None or (len(candidate), sorted(candidate)) < (len(best), sorted(best)):
            best = candidate
        if len(best) == 1:
            # The valid Why-Not precondition excludes a zero-deletion answer.
            break
    if best is None:
        return result("INFEASIBLE", None, {"processed_provenance": processed})
    if not iar_accepts(instance, best):
        raise AssertionError("returned IAR deletion does not restore the answer")
    return result("OPTIMAL", best,
                  {"processed_provenance": processed, "search_nodes": nodes,
                   "consistent_provenance": len(candidates)})


def solve_iar_approx(instance: ExplicitInstance, time_limit: float = 60.0) -> SolveResult:
    """Paper's maximal-disjoint-residual-edge union approximation."""
    started = time.monotonic()
    deadline = started + time_limit
    rank = max((len(edge) for edge in instance.conflicts), default=0)

    def result(status: str, best: set[int] | None, details: dict) -> SolveResult:
        return SolveResult(
            "IAR", instance.target, status,
            sorted(best) if best is not None else [],
            len(best) if best is not None else None,
            len(best) if status == "OPTIMAL" and best is not None else None,
            time.monotonic() - started,
            {"method": "maximal_disjoint_residual_union",
             "conflict_rank": rank, **details},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", None, {})
    if iar_accepts(instance, set()):
        return result("INVALID_ALREADY_ACCEPTED", set(), {})
    candidates = instance.consistent_provenance()
    if not candidates:
        return result("INFEASIBLE", None, {"consistent_provenance": 0})
    best: set[int] | None = None
    processed = 0
    packing_lower_bounds: list[int] = []
    for support in candidates:
        if time.monotonic() >= deadline:
            return result("FEASIBLE_NOT_PROVEN" if best is not None else "TIMEOUT",
                          best, {"processed_provenance": processed})
        residual = sorted(
            {conflict - support for conflict in instance.conflicts if conflict & support},
            key=lambda edge: (len(edge), tuple(sorted(edge))),
        )
        if any(not edge for edge in residual):
            raise AssertionError("consistent provenance has empty residual edge")
        used: set[int] = set()
        packed = 0
        for edge in residual:
            if not edge & used:
                used.update(edge)
                packed += 1
        packing_lower_bounds.append(packed)
        processed += 1
        if best is None or (len(used), sorted(used)) < (len(best), sorted(best)):
            best = used
        if len(best) == 1:
            break
    if best is None or not iar_accepts(instance, best):
        raise AssertionError("approximation returned an invalid IAR explanation")
    exact = len(best) == 1 or rank <= 2
    return result("OPTIMAL" if exact else "APPROXIMATE", best,
                  {"processed_provenance": processed,
                   "consistent_provenance": len(candidates),
                   "worst_case_ratio": max(1, rank - 1),
                   "packing_lower_bound": (min(packing_lower_bounds)
                                           if processed == len(candidates) else 1)})


class _BudgetSearchTimeout(Exception):
    pass


def solve_iar_fpt_decision(instance: ExplicitInstance, budget: int,
                           time_limit: float = 60.0) -> SolveResult:
    """Decide whether an IAR explanation of cardinality at most budget exists."""
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    started = time.monotonic()
    deadline = started + time_limit
    rank = max((len(edge) for edge in instance.conflicts), default=0)
    processed = 0
    nodes = 0

    def result(status: str, candidate: set[int] | None = None) -> SolveResult:
        return SolveResult(
            "IAR", instance.target, status,
            sorted(candidate) if candidate is not None else [],
            len(candidate) if candidate is not None else None,
            None,
            time.monotonic() - started,
            {"method": "bounded_residual_search", "budget": budget,
             "conflict_rank": rank, "processed_provenance": processed,
             "search_nodes": nodes},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY")
    if iar_accepts(instance, set()):
        return result("INVALID_ALREADY_ACCEPTED", set())
    candidates = instance.consistent_provenance()
    if not candidates:
        return result("BUDGET_INFEASIBLE")
    seen_budget: dict[tuple[frozenset[int], ...], int] = {}

    def branch(edges: tuple[frozenset[int], ...], chosen: frozenset[int],
               remaining_budget: int) -> set[int] | None:
        nonlocal nodes
        nodes += 1
        if time.monotonic() >= deadline:
            raise _BudgetSearchTimeout
        while True:
            forced = {next(iter(edge)) for edge in edges if len(edge) == 1}
            if not forced:
                break
            if len(forced) > remaining_budget:
                return None
            chosen |= forced
            remaining_budget -= len(forced)
            edges = tuple(edge for edge in edges if not edge & forced)
        if not edges:
            return set(chosen)
        if remaining_budget == 0:
            return None
        previous_budget = seen_budget.get(edges)
        if previous_budget is not None and previous_budget >= remaining_budget:
            return None
        seen_budget[edges] = remaining_budget
        used: set[int] = set()
        packing = 0
        for edge in sorted(edges, key=len):
            if not edge & used:
                used.update(edge)
                packing += 1
        if packing > remaining_budget:
            return None
        frequency = Counter(vertex for edge in edges for vertex in edge)
        selected = min(edges, key=lambda edge: (
            len(edge), -sum(frequency[v] for v in edge), tuple(sorted(edge))))
        for vertex in sorted(selected, key=lambda v: (-frequency[v], v)):
            found = branch(tuple(edge for edge in edges if vertex not in edge),
                           chosen | {vertex}, remaining_budget - 1)
            if found is not None:
                return found
        return None

    try:
        for support in candidates:
            if time.monotonic() >= deadline:
                raise _BudgetSearchTimeout
            residual = _minimal_edges({
                conflict - support for conflict in instance.conflicts
                if conflict & support
            })
            if any(not edge for edge in residual):
                raise AssertionError("consistent provenance has empty residual edge")
            found = branch(residual, frozenset(), budget)
            processed += 1
            if found is not None:
                if len(found) > budget or not iar_accepts(instance, found):
                    raise AssertionError("budget search returned an invalid explanation")
                return result("BUDGET_FEASIBLE", found)
    except _BudgetSearchTimeout:
        return result("TIMEOUT")
    return result("BUDGET_INFEASIBLE")


def solve_iar_boolean_opt(instance: ExplicitInstance, time_limit: float = 60.0) -> SolveResult:
    """Direct Boolean optimization of the complete IAR explanation formula."""
    started = time.monotonic()

    def result(status: str, best: set[int] | None, details: dict) -> SolveResult:
        optimum = status == "OPTIMAL"
        return SolveResult(
            "IAR", instance.target, status,
            sorted(best) if best is not None else [],
            len(best) if best is not None else None,
            len(best) if optimum and best is not None else None,
            time.monotonic() - started,
            {"method": "boolean_opt", **details},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", None, {})
    if iar_accepts(instance, set()):
        return result("INVALID_ALREADY_ACCEPTED", set(), {})
    candidates = instance.consistent_provenance()
    if not candidates:
        return result("INFEASIBLE", None, {"consistent_provenance": 0})

    z3 = _z3()
    x = [z3.Bool(f"iar_keep_{i}") for i in range(instance.n)]
    disjuncts = []
    residual_clauses = 0
    for support in candidates:
        clauses = [x[i] for i in sorted(support)]
        for conflict in instance.conflicts:
            if conflict & support:
                residual = conflict - support
                if not residual:
                    raise AssertionError("consistent provenance has empty residual")
                clauses.append(z3.Or(*[z3.Not(x[i]) for i in sorted(residual)]))
                residual_clauses += 1
        disjuncts.append(z3.And(*clauses))

    optimize = z3.Optimize()
    optimize.add(z3.Or(*disjuncts))
    optimize.minimize(z3.Sum(*[z3.If(z3.Not(variable), 1, 0) for variable in x]))
    remaining_ms = int((started + time_limit - time.monotonic()) * 1000)
    if remaining_ms <= 0:
        return result("TIMEOUT", None, {"consistent_provenance": len(candidates),
                                         "residual_clauses": residual_clauses})
    optimize.set(timeout=max(1, remaining_ms))
    status = optimize.check()
    details = {"consistent_provenance": len(candidates),
               "residual_clauses": residual_clauses}
    if status == z3.unknown:
        details["unknown_reason"] = optimize.reason_unknown()
        try:
            model = optimize.model()
            incumbent = {i for i, variable in enumerate(x)
                         if not z3.is_true(model.eval(variable, model_completion=True))}
            if iar_accepts(instance, incumbent):
                return result("FEASIBLE_NOT_PROVEN", incumbent, details)
        except z3.Z3Exception:
            pass
        return result("TIMEOUT", None, details)
    if status == z3.unsat:
        raise AssertionError("consistent provenance yielded unsatisfiable IAR encoding")
    model = optimize.model()
    deletions = {i for i, variable in enumerate(x)
                 if not z3.is_true(model.eval(variable, model_completion=True))}
    if not iar_accepts(instance, deletions):
        raise AssertionError("Boolean IAR encoding returned an invalid explanation")
    return result("OPTIMAL", deletions, details)


def solve_iar_maxsat(instance: ExplicitInstance, time_limit: float = 60.0) -> SolveResult:
    """Weighted partial MaxSAT encoding of the exact IAR deletion objective.

    Each consistent provenance set has a selector. Its hard implications keep
    that set and hit every conflict touching it. Unit soft clauses reward
    retaining facts, so their optimum is the minimum deletion cardinality.
    Z3 Optimize implements the MaxSMT backend for these Boolean clauses.
    """
    started = time.monotonic()
    deadline = started + time_limit

    def result(status: str, best: set[int] | None, details: dict) -> SolveResult:
        optimum = status == "OPTIMAL"
        return SolveResult(
            "IAR", instance.target, status,
            sorted(best) if best is not None else [],
            len(best) if best is not None else None,
            len(best) if optimum and best is not None else None,
            time.monotonic() - started,
            {"method": "maxsat_z3_soft", **details},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", None, {})
    if iar_accepts(instance, set()):
        return result("INVALID_ALREADY_ACCEPTED", set(), {})
    candidates = instance.consistent_provenance()
    if not candidates:
        return result("INFEASIBLE", None, {"consistent_provenance": 0})

    residual_families = []
    relevant = set().union(*candidates)
    for support in candidates:
        residual = _minimal_edges({conflict - support
                                   for conflict in instance.conflicts
                                   if conflict & support})
        if not residual or any(not edge for edge in residual):
            raise AssertionError("IAR premise and residual instance disagree")
        residual_families.append(residual)
        relevant.update(set().union(*residual))
    relevant_ids = sorted(relevant)
    details = {"consistent_provenance": len(candidates),
               "relevant_variables": len(relevant_ids),
               "hard_residual_clauses": sum(map(len, residual_families)),
               "soft_unit_clauses": len(relevant_ids)}
    if time.monotonic() >= deadline:
        return result("TIMEOUT", None, details)

    z3 = _z3()
    deletion = {v: z3.Bool(f"iar_maxsat_delete_{v}") for v in relevant_ids}
    selectors = [z3.Bool(f"iar_maxsat_support_{i}")
                 for i in range(len(candidates))]
    optimize = z3.Optimize()
    optimize.add(z3.Or(*selectors))
    for selector, support, residual in zip(selectors, candidates, residual_families):
        for vertex in support:
            optimize.add(z3.Or(z3.Not(selector), z3.Not(deletion[vertex])))
        for edge in residual:
            optimize.add(z3.Or(z3.Not(selector),
                               *(deletion[vertex] for vertex in sorted(edge))))
    for vertex in relevant_ids:
        optimize.add_soft(z3.Not(deletion[vertex]), weight="1")
    remaining_ms = int((deadline - time.monotonic()) * 1000)
    if remaining_ms <= 0:
        return result("TIMEOUT", None, details)
    optimize.set(timeout=max(1, remaining_ms))
    status = optimize.check()
    if status == z3.unknown:
        details["unknown_reason"] = optimize.reason_unknown()
        try:
            model = optimize.model()
            incumbent = {vertex for vertex in relevant_ids
                         if z3.is_true(model.eval(deletion[vertex],
                                                  model_completion=True))}
            if iar_accepts(instance, incumbent):
                return result("FEASIBLE_NOT_PROVEN", incumbent, details)
        except z3.Z3Exception:
            pass
        return result("TIMEOUT", None, details)
    if status == z3.unsat:
        raise AssertionError("consistent provenance yielded unsatisfiable MaxSAT encoding")
    model = optimize.model()
    answer = {vertex for vertex in relevant_ids
              if z3.is_true(model.eval(deletion[vertex], model_completion=True))}
    if not iar_accepts(instance, answer):
        raise AssertionError("MaxSAT IAR encoding returned an invalid explanation")
    return result("OPTIMAL", answer, details)


def solve_iar_rc2(instance: ExplicitInstance, time_limit: float = 60.0) -> SolveResult:
    """Exact IAR baseline using dedicated PySAT RC2 weighted partial MaxSAT."""
    started = time.monotonic()
    deadline = started + time_limit

    def result(status: str, best: set[int] | None, details: dict) -> SolveResult:
        optimum = status == "OPTIMAL"
        return SolveResult(
            "IAR", instance.target, status,
            sorted(best) if best is not None else [],
            len(best) if best is not None else None,
            len(best) if optimum and best is not None else None,
            time.monotonic() - started,
            {"method": "pysat_rc2_maxsat", **details},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", None, {})
    if iar_accepts(instance, set()):
        return result("INVALID_ALREADY_ACCEPTED", set(), {})
    candidates = instance.consistent_provenance()
    if not candidates:
        return result("INFEASIBLE", None, {"consistent_provenance": 0})

    from pysat.formula import WCNF

    residual_families = []
    relevant = set().union(*candidates)
    for support in candidates:
        residual = _minimal_edges({conflict - support
                                   for conflict in instance.conflicts
                                   if conflict & support})
        if not residual or any(not edge for edge in residual):
            raise AssertionError("IAR premise and residual instance disagree")
        residual_families.append(residual)
        relevant.update(set().union(*residual))
    relevant_ids = sorted(relevant)
    variables = {vertex: i + 1 for i, vertex in enumerate(relevant_ids)}
    selectors = [len(relevant_ids) + i + 1 for i in range(len(candidates))]
    details = {"consistent_provenance": len(candidates),
               "relevant_variables": len(relevant_ids),
               "hard_residual_clauses": sum(map(len, residual_families)),
               "soft_unit_clauses": len(relevant_ids)}
    if time.monotonic() >= deadline:
        return result("TIMEOUT", None, details)

    formula = WCNF()
    formula.append(selectors)
    for selector, support, residual in zip(selectors, candidates, residual_families):
        for vertex in support:
            formula.append([-selector, -variables[vertex]])
        for edge in residual:
            formula.append([-selector, *(variables[vertex]
                                         for vertex in sorted(edge))])
    for vertex in relevant_ids:
        formula.append([-variables[vertex]], weight=1)
    if time.monotonic() >= deadline:
        return result("TIMEOUT", None, details)

    with tempfile.TemporaryDirectory(prefix="sigmod-rc2-") as temporary:
        formula_path = Path(temporary) / "iar.wcnf"
        formula.to_file(str(formula_path))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return result("TIMEOUT", None, details)
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("rc2_worker.py")),
                 str(formula_path), "--deletion-variables", str(len(relevant_ids))],
                capture_output=True, text=True, timeout=remaining, check=False,
            )
        except subprocess.TimeoutExpired:
            return result("TIMEOUT", None, details)
    if completed.returncode:
        return result("SOLVER_ERROR", None,
                      {**details, "stderr_tail": completed.stderr[-500:]})
    output = json.loads(completed.stdout)
    if output["status"] != "optimal":
        raise AssertionError("consistent provenance yielded UNSAT RC2 encoding")
    answer = {relevant_ids[variable - 1]
              for variable in output["deletion_variables"]}
    if output["cost"] != len(answer):
        raise AssertionError("RC2 soft cost disagrees with deletion cardinality")
    if not iar_accepts(instance, answer):
        raise AssertionError("RC2 returned an invalid IAR explanation")
    return result("OPTIMAL", answer, details)


def solve_iar_residual_boolean_opt(
    instance: ExplicitInstance, time_limit: float = 60.0
) -> SolveResult:
    """Solve each residual hitting-set instance with generic Z3 optimization."""
    started = time.monotonic()
    deadline = started + time_limit

    def result(status: str, best: set[int] | None, details: dict) -> SolveResult:
        optimum = status == "OPTIMAL"
        return SolveResult(
            "IAR", instance.target, status,
            sorted(best) if best is not None else [],
            len(best) if best is not None else None,
            len(best) if optimum and best is not None else None,
            time.monotonic() - started,
            {"method": "residual_boolean_opt", **details},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", None, {})
    if iar_accepts(instance, set()):
        return result("INVALID_ALREADY_ACCEPTED", set(), {})
    candidates = instance.consistent_provenance()
    if not candidates:
        return result("INFEASIBLE", None, {"consistent_provenance": 0})
    z3 = _z3()
    best: set[int] | None = None
    processed = 0
    total_variables = total_edges = 0

    def details() -> dict:
        return {"consistent_provenance": len(candidates),
                "processed_provenance": processed,
                "residual_variables_total": total_variables,
                "residual_edges_total": total_edges}

    for support in candidates:
        if time.monotonic() >= deadline:
            return result("FEASIBLE_NOT_PROVEN" if best is not None else "TIMEOUT",
                          best, details())
        residual = _minimal_edges({conflict - support for conflict in instance.conflicts
                                   if conflict & support})
        if not residual or any(not edge for edge in residual):
            raise AssertionError("IAR premise and residual instance disagree")
        vertices = sorted(set().union(*residual))
        variables = {v: z3.Bool(f"delete_{processed}_{v}") for v in vertices}
        optimize = z3.Optimize()
        for edge in residual:
            optimize.add(z3.Or(*[variables[v] for v in sorted(edge)]))
        optimize.minimize(z3.Sum(*[z3.If(variables[v], 1, 0) for v in vertices]))
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            return result("FEASIBLE_NOT_PROVEN" if best is not None else "TIMEOUT",
                          best, details())
        optimize.set(timeout=max(1, remaining_ms))
        status = optimize.check()
        if status == z3.unknown:
            try:
                model = optimize.model()
                incumbent = {v for v in vertices
                             if z3.is_true(model.eval(variables[v], model_completion=True))}
                if iar_accepts(instance, incumbent) and (
                    best is None or len(incumbent) < len(best)
                ):
                    best = incumbent
            except z3.Z3Exception:
                pass
            return result("FEASIBLE_NOT_PROVEN" if best is not None else "TIMEOUT",
                          best, {**details(), "unknown_reason": optimize.reason_unknown()})
        if status == z3.unsat:
            raise AssertionError("nonempty residual edges yielded UNSAT hitting set")
        model = optimize.model()
        candidate = {v for v in vertices
                     if z3.is_true(model.eval(variables[v], model_completion=True))}
        if not iar_accepts(instance, candidate):
            raise AssertionError("residual Boolean solver returned invalid explanation")
        processed += 1
        total_variables += len(vertices)
        total_edges += len(residual)
        if best is None or (len(candidate), sorted(candidate)) < (len(best), sorted(best)):
            best = candidate
        if len(best) == 1:
            break
    assert best is not None
    return result("OPTIMAL", best, details())


def _relevant_subinstance(instance: ExplicitInstance, indexed_provenance: bool = False) -> tuple[ExplicitInstance, list[int]]:
    """Keep conflict/provenance components that can affect the target."""
    active_provenance = instance.consistent_provenance(indexed=indexed_provenance)
    incident: dict[int, list[int]] = defaultdict(list)
    for edge_id, conflict in enumerate(instance.conflicts):
        for vertex in conflict:
            incident[vertex].append(edge_id)
    selected = set().union(*active_provenance)
    frontier = list(selected)
    expanded_edges: set[int] = set()
    while frontier:
        vertex = frontier.pop()
        for edge_id in incident[vertex]:
            if edge_id in expanded_edges:
                continue
            expanded_edges.add(edge_id)
            for neighbor in instance.conflicts[edge_id]:
                if neighbor not in selected:
                    selected.add(neighbor)
                    frontier.append(neighbor)
    original_ids = sorted(selected)
    local_id = {original: local for local, original in enumerate(original_ids)}
    conflicts = tuple(
        frozenset(local_id[i] for i in conflict)
        for edge_id, conflict in enumerate(instance.conflicts)
        if edge_id in expanded_edges
    )
    provenance = tuple(
        frozenset(local_id[i] for i in support) for support in active_provenance
    )
    return ExplicitInstance(len(original_ids), conflicts, provenance, instance.target), original_ids


def _z3():
    try:
        import z3
    except ImportError as exc:
        raise RuntimeError(
            "AR solver needs z3-solver; see experiments/requirements-solvers.txt"
        ) from exc
    return z3


class _ARBackend:
    def __init__(self, instance: ExplicitInstance):
        z3 = _z3()
        self.z3 = z3
        self.instance = instance
        self.x = [z3.Bool(f"keep_{i}") for i in range(instance.n)]
        self.y = [z3.Bool(f"repair_{i}") for i in range(instance.n)]
        self.incident: dict[int, list[frozenset[int]]] = defaultdict(list)
        for conflict in instance.conflicts:
            for vertex in conflict:
                self.incident[vertex].append(conflict)

        self.separator = z3.Solver()
        for i in range(instance.n):
            self.separator.add(z3.Implies(self.y[i], self.x[i]))
        for conflict in instance.conflicts:
            self.separator.add(z3.Or(*[z3.Not(self.y[i]) for i in conflict]))
        for i in range(instance.n):
            blockers = [
                z3.And(*[self.y[j] for j in conflict if j != i])
                for conflict in self.incident[i]
            ]
            self.separator.add(
                z3.Implies(
                    z3.And(self.x[i], z3.Not(self.y[i])), z3.Or(*blockers)
                )
            )
        for support in instance.provenance:
            self.separator.add(z3.Or(*[z3.Not(self.y[i]) for i in support]))

        self.master = z3.Optimize()
        self.master.add(
            z3.Or(*[
                z3.And(*[self.x[i] for i in support])
                for support in instance.provenance
            ])
        )
        self.deletion_cost = z3.Sum(*[z3.If(z3.Not(v), 1, 0) for v in self.x])
        self.master.minimize(self.deletion_cost)
        self.cut_keys: set[tuple[frozenset[int], frozenset[int]]] = set()
        self.cut_literals = 0

    def bad_repair(self, kept: set[int], deadline: float) -> tuple[str, set[int] | None]:
        z3 = self.z3
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        if time.monotonic() >= deadline:
            return "unknown", None
        self.separator.set(timeout=remaining_ms)
        assumptions = [self.x[i] if i in kept else z3.Not(self.x[i])
                       for i in range(self.instance.n)]
        status = self.separator.check(*assumptions)
        if status == z3.sat:
            model = self.separator.model()
            repair = {i for i in kept if z3.is_true(model.eval(self.y[i], model_completion=True))}
            self._validate_bad_repair(repair, kept)
            return "sat", repair
        if status == z3.unsat:
            return "unsat", None
        return "unknown", None

    def _validate_bad_repair(self, repair: set[int], kept: set[int]) -> None:
        instance = self.instance
        if not repair <= kept:
            raise AssertionError("separator returned a repair outside the candidate")
        if any(conflict <= repair for conflict in instance.conflicts):
            raise AssertionError("separator returned an inconsistent repair")
        if any(support <= repair for support in instance.provenance):
            raise AssertionError("separator returned a supporting repair")
        for vertex in kept - repair:
            if not any(conflict - {vertex} <= repair for conflict in self.incident[vertex]):
                raise AssertionError("separator repair is not maximal")

    def optimum_candidate(self, deadline: float) -> tuple[str, set[int] | None]:
        z3 = self.z3
        if time.monotonic() >= deadline:
            return "unknown", None
        self.master.set(timeout=max(1, int((deadline - time.monotonic()) * 1000)))
        status = self.master.check()
        if status == z3.sat:
            model = self.master.model()
            return "sat", {
                i for i, variable in enumerate(self.x)
                if z3.is_true(model.eval(variable, model_completion=True))
            }
        if status == z3.unsat:
            return "unsat", None
        return "unknown", None

    def add_interval_cut(self, repair: set[int]) -> None:
        z3 = self.z3
        blocked = {
            vertex for vertex in range(self.instance.n) if vertex not in repair
            and any(conflict - {vertex} <= repair for conflict in self.incident[vertex])
        }
        outside = set(range(self.instance.n)) - repair - blocked
        key = (frozenset(repair), frozenset(outside))
        if key in self.cut_keys:
            raise AssertionError("duplicate repair cut does not exclude current model")
        self.cut_keys.add(key)
        self.cut_literals += len(repair) + len(outside)
        self.master.add(
            z3.Or(*([z3.Not(self.x[i]) for i in sorted(repair)]
                     + [self.x[i] for i in sorted(outside)]))
        )

    def add_point_cut(self, kept: set[int]) -> None:
        z3 = self.z3
        outside = set(range(self.instance.n)) - kept
        key = (frozenset(kept), frozenset(outside))
        if key in self.cut_keys:
            raise AssertionError("duplicate point cut does not exclude current model")
        self.cut_keys.add(key)
        self.cut_literals += self.instance.n
        self.master.add(z3.Or(*([z3.Not(self.x[i]) for i in sorted(kept)]
                                 + [self.x[i] for i in sorted(outside)])))


def solve_ar(instance: ExplicitInstance, time_limit: float = 60.0,
             cut_mode: str = "interval", warm_iar_method: str = "residual",
             bound_from_warm: bool = False, warm_iar_seconds: float | None = None) -> SolveResult:
    if cut_mode not in ("interval", "point"):
        raise ValueError("cut_mode must be 'interval' or 'point'")
    if warm_iar_method not in ("residual", "rc2"):
        raise ValueError("warm_iar_method must be 'residual' or 'rc2'")
    if warm_iar_seconds is not None and warm_iar_seconds<=0:
        raise ValueError("warm IAR limit must be positive")
    warm_solver = solve_iar if warm_iar_method == "residual" else solve_iar_rc2
    started = time.monotonic()
    deadline = started + time_limit
    warm_best: set[int] | None = None
    warm_seconds = 0.0

    def result(status: str, deletions: set[int] | None, details: dict) -> SolveResult:
        optimal = status == "OPTIMAL"
        details = {"cut_mode": cut_mode, "bound_from_warm": bound_from_warm, **details,
                   "iar_feasible_fallback_method": warm_iar_method,
                   "iar_feasible_fallback_limit_seconds": warm_iar_seconds,
                   "iar_feasible_fallback_seconds": warm_seconds,
                   "iar_feasible_upper_bound": len(warm_best) if warm_best is not None else None}
        return SolveResult(
            "AR", instance.target, status,
            sorted(deletions) if deletions is not None else [],
            len(deletions) if deletions is not None else None,
            len(deletions) if optimal and deletions is not None else None,
            time.monotonic() - started, details,
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", None, {})
    consistent = instance.consistent_provenance()
    if not consistent:
        return result("INFEASIBLE", None, {"consistent_provenance": 0})
    if len(consistent) == 1:
        # With only one repair-usable provenance set, AR and IAR coincide:
        # every repair contains that same set iff their intersection does.
        iar = warm_solver(instance, max(0.0, deadline - time.monotonic()))
        return result(
            iar.status, set(iar.deletions) if iar.objective_upper_bound is not None else None,
            {"method": "single-provenance AR=IAR", "iar_details": iar.details},
        )
    local, original_ids = _relevant_subinstance(instance)
    backend = _ARBackend(local)
    full = set(range(local.n))
    separator_calls = 1
    initial_status, _ = backend.bad_repair(full, deadline)
    if initial_status == "unsat":
        return result("INVALID_ALREADY_ACCEPTED", set(),
                      {"relevant_facts": local.n, "separator_calls": separator_calls})
    if initial_status == "unknown":
        return result("TIMEOUT", None,
                      {"relevant_facts": local.n, "separator_calls": separator_calls})

    warm_started = time.monotonic()
    if warm_iar_seconds is not None:
        support=max(local.provenance,key=lambda s:(len(s),tuple(sorted(s))))
        warm_best={original_ids[i] for i in full-support}
        if not iar_accepts(instance,warm_best):
            raise AssertionError("trivial consistent-support fallback is invalid")
    warm_budget=max(0.0,deadline-warm_started)
    if warm_iar_seconds is not None:
        warm_budget=min(warm_budget,warm_iar_seconds)
    iar = warm_solver(instance, warm_budget)
    warm_seconds = time.monotonic() - warm_started
    if iar.objective_upper_bound is not None and iar.status in (
        "OPTIMAL", "FEASIBLE_NOT_PROVEN"
    ):
        candidate = set(iar.deletions)
        if not iar_accepts(instance, candidate):
            raise AssertionError("IAR warm start is not a feasible explanation")
        if warm_best is None or len(candidate)<len(warm_best):
            warm_best=candidate
    if bound_from_warm and warm_best is not None:
        # A failed capped warm search still leaves a valid support fallback.
        backend.master.add(backend.deletion_cost < len(warm_best))

    def timeout(details: dict) -> SolveResult:
        return result("FEASIBLE_NOT_PROVEN" if warm_best is not None else "TIMEOUT",
                      warm_best, details)

    def stats() -> dict:
        return {"relevant_facts": local.n, "iterations": iterations,
                "separator_calls": separator_calls, "cuts": len(backend.cut_keys),
                "cut_literals": backend.cut_literals}

    iterations = 0
    while time.monotonic() < deadline:
        master_status, kept = backend.optimum_candidate(deadline)
        if master_status == "unknown":
            return timeout(stats())
        if master_status == "unsat":
            if bound_from_warm and warm_best is not None:
                return result("OPTIMAL", warm_best,
                              {**stats(), "certificate": "strict_warm_bound_exhausted"})
            # Without a strict incumbent bound, a consistent provenance set
            # can always be retained on its own.
            raise AssertionError("AR master excluded every feasible explanation")
        assert kept is not None
        iterations += 1
        separator_calls += 1
        separator_status, repair = backend.bad_repair(kept, deadline)
        if separator_status == "unknown":
            return timeout(stats())
        if separator_status == "unsat":
            deletions = {original_ids[i] for i in full - kept}
            return result("OPTIMAL", deletions, stats())
        assert repair is not None
        if cut_mode == "interval":
            backend.add_interval_cut(repair)
        else:
            backend.add_point_cut(kept)
    return timeout(stats())


def solve_ar_qbf_z3(instance: ExplicitInstance, time_limit: float = 60.0) -> SolveResult:
    """Direct quantified AR formula with bounded-cardinality Z3 checks.

    This is a generic quantified pilot baseline, not a dedicated QBF engine.
    """
    started = time.monotonic()
    deadline = started + time_limit
    best: set[int] | None = None
    original_ids: list[int] = []
    checks = sat_checks = unsat_checks = 0

    def result(status: str, details: dict) -> SolveResult:
        optimum = status == "OPTIMAL"
        deletions = {original_ids[i] for i in best} if best is not None else None
        return SolveResult(
            "AR", instance.target, status,
            sorted(deletions) if deletions is not None else [],
            len(deletions) if deletions is not None else None,
            len(deletions) if optimum and deletions is not None else None,
            time.monotonic() - started,
            {"method": "qbf_z3", "quantified_checks": checks,
             "quantified_sat": sat_checks, "quantified_unsat": unsat_checks,
             **details},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", {})
    if not instance.consistent_provenance():
        return result("INFEASIBLE", {"consistent_provenance": 0})
    local, original_ids = _relevant_subinstance(instance)
    n = local.n
    backend = _ARBackend(local)
    initial_status, _ = backend.bad_repair(set(range(n)), deadline)
    if initial_status == "unsat":
        best = set()
        return result("INVALID_ALREADY_ACCEPTED", {"relevant_facts": n})
    if initial_status == "unknown":
        return result("TIMEOUT", {"relevant_facts": n, "phase": "precondition"})

    # Keeping only one consistent provenance set is always an AR-feasible
    # candidate. This upper bound requires no call to another exact solver.
    support = max(local.provenance, key=lambda item: (len(item), tuple(sorted(item))))
    best = set(range(n)) - support
    z3 = _z3()
    x = [z3.Bool(f"qbf_keep_{i}") for i in range(n)]
    y = [z3.Bool(f"qbf_repair_{i}") for i in range(n)]
    incident: dict[int, list[frozenset[int]]] = defaultdict(list)
    for conflict in local.conflicts:
        for vertex in conflict:
            incident[vertex].append(conflict)
    repair_terms = [z3.Implies(y[i], x[i]) for i in range(n)]
    repair_terms.extend(z3.Or(*[z3.Not(y[i]) for i in conflict])
                        for conflict in local.conflicts)
    for i in range(n):
        blockers = [z3.And(*[y[j] for j in conflict if j != i])
                    for conflict in incident[i]]
        repair_terms.append(z3.Implies(z3.And(x[i], z3.Not(y[i])),
                                       z3.Or(*blockers)))
    bad_repair = z3.And(
        *repair_terms,
        *[z3.Or(*[z3.Not(y[i]) for i in provenance])
          for provenance in local.provenance],
    )
    quantified = z3.ForAll(y, z3.Not(bad_repair))
    solver = z3.Solver()
    solver.add(quantified)
    cost = z3.Sum(*[z3.If(z3.Not(variable), 1, 0) for variable in x])
    lower, upper = 1, len(best)
    initial_upper = upper
    while lower < upper:
        if time.monotonic() >= deadline:
            return result("FEASIBLE_NOT_PROVEN", {"relevant_facts": n,
                                                   "initial_upper_bound": initial_upper,
                                                   "proven_lower_bound": lower})
        midpoint = (lower + upper) // 2
        solver.push()
        solver.add(cost <= midpoint)
        solver.set(timeout=max(1, int((deadline - time.monotonic()) * 1000)))
        status = solver.check()
        checks += 1
        if status == z3.sat:
            sat_checks += 1
            model = solver.model()
            candidate = {i for i, variable in enumerate(x)
                         if not z3.is_true(model.eval(variable, model_completion=True))}
            if len(candidate) > midpoint:
                raise AssertionError("quantified model violates deletion budget")
            separator_status, _ = backend.bad_repair(set(range(n)) - candidate,
                                                      deadline)
            if separator_status == "sat":
                raise AssertionError("quantified AR model has a bad repair")
            if len(candidate) < len(best):
                best = candidate
            upper = len(candidate)
        elif status == z3.unsat:
            unsat_checks += 1
            lower = midpoint + 1
        else:
            reason = solver.reason_unknown()
            solver.pop()
            return result("FEASIBLE_NOT_PROVEN", {"relevant_facts": n,
                                                   "initial_upper_bound": initial_upper,
                                                   "proven_lower_bound": lower,
                                                   "unknown_reason": reason})
        solver.pop()
    if len(best) != lower:
        raise AssertionError("QBF bounds closed without a matching feasible solution")
    return result("OPTIMAL", {"relevant_facts": n,
                              "initial_upper_bound": initial_upper,
                              "proven_lower_bound": lower})


def _incidence_components(
    instance: ExplicitInstance,
) -> list[tuple[ExplicitInstance, list[int]]]:
    """Split a relevant instance by conflict/provenance incidence connectivity."""
    parent = list(range(instance.n))

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for support in (*instance.conflicts, *instance.provenance):
        first = next(iter(support))
        for vertex in support:
            parent[root(vertex)] = root(first)
    groups: dict[int, list[int]] = defaultdict(list)
    for vertex in range(instance.n):
        groups[root(vertex)].append(vertex)
    output = []
    for original_ids in sorted(groups.values(), key=lambda ids: ids[0]):
        remap = {vertex: index for index, vertex in enumerate(original_ids)}
        conflicts = tuple(frozenset(remap[v] for v in edge)
                          for edge in instance.conflicts if next(iter(edge)) in remap)
        provenance = tuple(frozenset(remap[v] for v in support)
                           for support in instance.provenance
                           if next(iter(support)) in remap)
        if not provenance:
            continue
        output.append((ExplicitInstance(len(original_ids), conflicts, provenance,
                                        instance.target), original_ids))
    return output


def solve_ar_components(instance: ExplicitInstance, time_limit: float = 60.0,
                        cut_mode: str = "interval",
                        warm_iar_method: str = "residual",
                        bound_from_warm: bool = False,
                        warm_iar_seconds: float | None = None) -> SolveResult:
    """Exact AR optimization by the incidence-component factorization theorem."""
    if cut_mode not in ("interval", "point"):
        raise ValueError("cut_mode must be 'interval' or 'point'")
    if warm_iar_method not in ("residual", "rc2"):
        raise ValueError("warm_iar_method must be 'residual' or 'rc2'")
    if warm_iar_seconds is not None and warm_iar_seconds<=0:
        raise ValueError("warm IAR limit must be positive")
    started = time.monotonic()
    deadline = started + time_limit
    best: set[int] | None = None
    processed = 0
    component_statuses: list[str] = []
    decomposition_seconds = 0.0
    premise_seconds = 0.0
    component_solve_seconds = 0.0

    def result(status: str, details: dict) -> SolveResult:
        optimum = status == "OPTIMAL"
        return SolveResult(
            "AR", instance.target, status,
            sorted(best) if best is not None else [],
            len(best) if best is not None else None,
            len(best) if optimum and best is not None else None,
            time.monotonic() - started,
            {"method": "components", "cut_mode": cut_mode,
             "processed_components": processed,
             "component_statuses": component_statuses,
             "decomposition_seconds": decomposition_seconds,
             "premise_seconds": premise_seconds,
             "component_solve_seconds": component_solve_seconds,
             "iar_feasible_fallback_method": warm_iar_method,
             "iar_feasible_fallback_limit_seconds": warm_iar_seconds,
             "bound_from_warm": bound_from_warm,
             **details},
        )

    if not instance.provenance:
        return result("INVALID_NOT_ORDINARY", {})
    if not instance.consistent_provenance():
        return result("INFEASIBLE", {"consistent_provenance": 0})
    local, original_ids = _relevant_subinstance(instance)
    components = _incidence_components(local)
    decomposition_seconds = time.monotonic() - started
    sizes = sorted((component.n for component, _ in components))
    if len(components) == 1:
        inner = solve_ar(instance, max(0.0, deadline - time.monotonic()), cut_mode,
                         warm_iar_method, bound_from_warm, warm_iar_seconds)
        component_solve_seconds = time.monotonic() - started - decomposition_seconds
        inner.elapsed_seconds = time.monotonic() - started
        inner.details = {**inner.details, "method": "components",
                         "component_count": 1, "component_sizes": sizes,
                         "processed_components": 1,
                         "decomposition_seconds": decomposition_seconds,
                         "premise_seconds": 0.0,
                         "component_solve_seconds": component_solve_seconds}
        return inner

    premise_started = time.monotonic()
    premise_status, _ = _ARBackend(local).bad_repair(set(range(local.n)), deadline)
    premise_seconds = time.monotonic() - premise_started
    if premise_status == "unsat":
        best = set()
        return result("INVALID_ALREADY_ACCEPTED",
                      {"component_count": len(components), "component_sizes": sizes})
    if premise_status == "unknown":
        return result("TIMEOUT", {"component_count": len(components),
                                  "component_sizes": sizes, "phase": "precondition"})

    all_proved = True
    for component, local_ids in sorted(components, key=lambda pair: pair[0].n):
        if time.monotonic() >= deadline:
            all_proved = False
            break
        component_started = time.monotonic()
        inner = solve_ar(component, max(0.0, deadline - component_started), cut_mode,
                         warm_iar_method, bound_from_warm, warm_iar_seconds)
        component_solve_seconds += time.monotonic() - component_started
        processed += 1
        component_statuses.append(inner.status)
        if inner.status == "INVALID_ALREADY_ACCEPTED":
            raise AssertionError("global AR premise disagrees with component premise")
        if inner.objective_upper_bound is not None:
            candidate = {original_ids[local_ids[i]] for i in inner.deletions}
            if best is None or (len(candidate), sorted(candidate)) < (len(best), sorted(best)):
                best = candidate
        if inner.status not in ("OPTIMAL", "INFEASIBLE"):
            all_proved = False
        if best is not None and len(best) == 1:
            return result("OPTIMAL", {"component_count": len(components),
                                      "component_sizes": sizes,
                                      "certificate": "global Why-Not premise and one deletion"})

    details = {"component_count": len(components), "component_sizes": sizes}
    if processed == len(components) and all_proved:
        return result("OPTIMAL" if best is not None else "INFEASIBLE", details)
    return result("FEASIBLE_NOT_PROVEN" if best is not None else "TIMEOUT", details)
