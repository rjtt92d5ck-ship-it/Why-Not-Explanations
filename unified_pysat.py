from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from exact_solvers import ExplicitInstance, SolveResult, _minimal_edges, _relevant_subinstance, iar_accepts


class DeadlineReached(Exception):
    pass


class Engine:
    def __init__(self, instance, semantics, objective, deadline, checkpoint, cut_mode="interval", request_started=None,
                 subset_seed="consistent_provenance", repair_batch_size=1, iar_encoding="residual_simplified",
                 ar_single_provenance=False, indexed_families=False):
        self.original = instance
        self.semantics, self.objective = semantics, objective
        self.effective_semantics = semantics
        if type(ar_single_provenance) is not bool or type(indexed_families) is not bool:
            raise ValueError("implementation options must be Boolean")
        if ar_single_provenance and semantics != "AR":
            raise ValueError("single-provenance AR reduction requires AR semantics")
        self.ar_single_provenance = ar_single_provenance
        self.indexed_families = indexed_families
        self.deadline, self.checkpoint, self.cut_mode = deadline, checkpoint, cut_mode
        if subset_seed not in ("consistent_provenance", "greedy_iar"):
            raise ValueError("unsupported subset initialization")
        self.subset_seed = subset_seed
        if type(repair_batch_size) is not int or repair_batch_size < 1:
            raise ValueError("repair batch size must be a positive integer")
        if repair_batch_size != 1 and (semantics != "AR" or cut_mode != "interval"):
            raise ValueError("repair batches require AR interval cuts")
        self.repair_batch_size = repair_batch_size
        if iar_encoding not in ("direct_formula", "residual_simplified"):
            raise ValueError("unsupported IAR encoding")
        if semantics != "IAR" and iar_encoding != "residual_simplified":
            raise ValueError("direct IAR formula requires IAR semantics")
        self.iar_encoding = iar_encoding
        self.started = time.monotonic() if request_started is None else request_started
        self.best = None
        self.lower = None
        self.master = self.separator = None
        self.ids = list(range(instance.n))
        self.details = {
            "method": "unified_pysat", "objective": objective, "cut_mode": cut_mode,
            "sat_backend": "g4", "optimization_backend": "RC2" if objective == "minimum_cardinality" else None,
            "master_calls": 0, "separator_calls": 0, "add_back_tests": 0,
            "cuts": 0, "cut_literals": 0, "phase_seconds": {}, "timeline": [],
            "subset_minimal_certified": False,
            "subset_seed": subset_seed if objective == "subset_minimal" else None,
            "repair_batch_size": repair_batch_size, "repair_batches": 0,
            "iar_encoding": iar_encoding if semantics == "IAR" else None,
            "indexed_families": indexed_families,
            "ar_single_provenance_enabled": ar_single_provenance,
            "ar_single_provenance_reduced": False,
        }
        self.phase = "initialization"
        self.phase_started = time.monotonic()

    def save(self, status="FEASIBLE_NOT_PROVEN"):
        if self.best is None and status == "FEASIBLE_NOT_PROVEN":
            status = "TIMEOUT"
        result = SolveResult(self.semantics, self.original.target, status,
            sorted(self.ids[v] for v in self.best) if self.best is not None else [],
            len(self.best) if self.best is not None else None, self.lower,
            time.monotonic() - self.started,
            {**self.details, "phase": self.phase, "active_phase_started": self.phase_started}).as_dict()
        temporary = self.checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(result))
        temporary.replace(self.checkpoint)
        return result

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise DeadlineReached

    def call(self, phase, function):
        self.check_deadline()
        self.phase, self.phase_started = phase, time.monotonic()
        self.save()
        result = function()
        elapsed = time.monotonic() - self.phase_started
        totals = self.details["phase_seconds"]
        totals[phase] = totals.get(phase, 0.) + elapsed
        self.phase, self.phase_started = "between_calls", time.monotonic()
        self.check_deadline()
        return result

    def trace(self, event):
        # Record bound changes, not a growing copy of the same bound after
        # every master solve. Counters still record every call and cut.
        if event == "proved_master_lower_bound" and self.details["timeline"]:
            previous = self.details["timeline"][-1]
            if (previous["event"] == event and previous["lower_bound"] == self.lower
                    and previous["upper_bound"] == (len(self.best) if self.best is not None else None)):
                return
        self.details["timeline"].append({"seconds": time.monotonic() - self.started,
            "event": event, "upper_bound": len(self.best) if self.best is not None else None,
            "lower_bound": self.lower})
        self.save()

    def build_separator(self):
        from pysat.solvers import Solver
        n = self.local.n
        # d_i=i+1, y_i=n+i+1. d_i means deletion, y_i repair membership.
        self.y = lambda i: n + i + 1
        fresh = 2 * n
        clauses = [[-self.y(i), -(i + 1)] for i in range(n)]
        blockers = defaultdict(list)
        for edge in self.local.conflicts:
            clauses.append([-self.y(i) for i in sorted(edge)])
            for i in sorted(edge):
                fresh += 1
                others = [self.y(j) for j in sorted(edge - {i})]
                clauses.extend([[-fresh, literal] for literal in others])
                clauses.append([fresh, *[-literal for literal in others]])
                blockers[i].append(fresh)
        clauses.extend([[i + 1, self.y(i), *blockers[i]] for i in range(n)])
        clauses.extend([[-self.y(i) for i in sorted(s)] for s in self.local.provenance])
        self.separator = Solver(name="g4", bootstrap_with=clauses)
        self.separator_next_variable = fresh + 1
        self.details["separator_clauses"] = len(clauses)

    def bad_repair(self, deleted, phase="separator", extra_assumptions=()):
        self.details["separator_calls"] += 1
        assumptions = [i + 1 if i in deleted else -(i + 1) for i in range(self.local.n)]
        assumptions.extend(extra_assumptions)
        sat = self.call(phase, lambda: self.separator.solve(assumptions=assumptions))
        if not sat:
            return None
        model = set(self.separator.get_model())
        repair = {i for i in range(self.local.n) if self.y(i) in model}
        kept = set(range(self.local.n)) - deleted
        if (not repair <= kept or any(e <= repair for e in self.local.conflicts)
                or any(s <= repair for s in self.local.provenance)
                or any(not any(e - {i} <= repair for e in self.incident[i]) for i in kept - repair)):
            raise AssertionError("invalid bad repair from SAT separator")
        return repair

    def add_repair_batch(self, deleted):
        """Add up to b cuts for one candidate; zero means a full UNSAT check.

        Enumeration blocks are guarded and disabled before another candidate
        or add-back test. Exhausting a nonempty batch NEVER proves feasibility.
        """
        self.details["repair_batches"] += 1
        repair = self.bad_repair(deleted)
        if repair is None:
            return 0
        self.add_cut(repair, deleted)
        count = 1
        if self.repair_batch_size == 1:
            return count
        guard = self.separator_next_variable
        self.separator_next_variable += 1
        try:
            while count < self.repair_batch_size:
                # Block exactly this repair membership vector, only in this batch.
                self.separator.add_clause([-guard, *[-self.y(i) if i in repair else self.y(i)
                                                    for i in range(self.local.n)]])
                repair = self.bad_repair(deleted, extra_assumptions=(guard,))
                if repair is None:
                    break
                self.add_cut(repair, deleted)
                count += 1
            return count
        finally:
            self.separator.add_clause([-guard])

    def touching_conflicts(self, support):
        if not self.indexed_families:
            return [e for e in self.local.conflicts if e & support]
        # Preserve the old conflict order, including for the direct-encoding
        # ablation, while visiting only incident edges and emitting each once.
        indices = {self.edge_order[e] for v in support for e in self.incident[v]}
        return [self.local.conflicts[i] for i in sorted(indices)]

    def accepts_iar(self, deleted):
        return iar_accepts(self.local, deleted, indexed=self.indexed_families)

    def master_clauses(self):
        n = self.local.n
        selectors = [n + j + 1 for j in range(len(self.local.provenance))]
        clauses = [selectors]
        raw_residual_count = encoded_residual_count = 0
        for selector, support in zip(selectors, self.local.provenance):
            clauses.extend([[-selector, -(i + 1)] for i in sorted(support)])
            if self.effective_semantics == "IAR":
                # Both arms encode the paper's exact Phi_IAR. The direct arm
                # retains every conjunct; only the simplified arm removes
                # duplicate/subsumed residual edges. This is a simplification
                # ablation, not a comparison of different solver backends.
                raw = [e - support for e in self.touching_conflicts(support)]
                raw_residual_count += len(raw)
                residual = (_minimal_edges(set(raw)) if self.iar_encoding == "residual_simplified"
                            else sorted(raw, key=lambda e: (len(e), tuple(sorted(e)))))
                encoded_residual_count += len(residual)
                clauses.extend([[-selector, *[i + 1 for i in sorted(e)]] for e in residual])
        if self.effective_semantics == "IAR":
            self.details["residual_conjuncts_before_simplification"] = raw_residual_count
            self.details["residual_conjuncts_encoded"] = encoded_residual_count
        return clauses

    def build_master(self):
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF
        from pysat.solvers import Solver
        clauses = self.master_clauses()
        self.details["master_initial_clauses"] = len(clauses)
        self.details["master_initial_literals"] = sum(map(len, clauses))
        self.details["master_initial_variables"] = max((abs(v) for c in clauses for v in c), default=0)
        self.details["master_soft_clauses"] = self.local.n if self.objective == "minimum_cardinality" else 0
        if self.objective == "minimum_cardinality":
            formula = WCNF()
            for clause in clauses:
                formula.append(clause)
            for i in range(self.local.n):
                formula.append([-(i + 1)], weight=1)
            self.master = RC2(formula, solver="g4", incr=True)
        else:
            self.master = Solver(name="g4", bootstrap_with=clauses)

    def add_cut(self, repair, deleted):
        full = set(range(self.local.n))
        if self.cut_mode == "interval":
            blocked = {i for i in full - repair if any(e - {i} <= repair for e in self.incident[i])}
            outside = full - repair - blocked
            clause = [i + 1 for i in sorted(repair)] + [-(i + 1) for i in sorted(outside)]
        else:
            clause = [i + 1 if i not in deleted else -(i + 1) for i in range(self.local.n)]
        key = tuple(clause)
        if key in self.cut_keys:
            raise AssertionError("duplicate cut failed to exclude candidate")
        self.cut_keys.add(key)
        self.master.add_clause(clause)
        self.details["cuts"] += 1
        self.details["cut_literals"] += len(clause)

    def feasible_model(self, forced_kept=frozenset()):
        while True:
            self.details["master_calls"] += 1
            if self.objective == "minimum_cardinality":
                model = self.call("master", self.master.compute)
            else:
                sat = self.call("master", lambda: self.master.solve(
                    assumptions=[-(i + 1) for i in sorted(forced_kept)]))
                model = self.master.get_model() if sat else None
            if model is None:
                return None
            positive = set(model)
            deleted = {i for i in range(self.local.n) if i + 1 in positive}
            if self.objective == "minimum_cardinality":
                if self.master.cost != len(deleted):
                    raise AssertionError("RC2 cost disagrees with deletion cardinality")
                self.lower = max(self.lower or 0, len(deleted))
                self.trace("proved_master_lower_bound")
                if self.best is not None and self.lower == len(self.best):
                    self.details["certificate"] = "master_lower_bound_meets_verified_upper_bound"
                    return self.best
            if self.effective_semantics == "IAR":
                if not self.accepts_iar(deleted):
                    raise AssertionError("IAR encoding returned invalid candidate")
                return deleted
            if self.add_repair_batch(deleted) == 0:
                return deleted

    def greedy_incumbent(self):
        # A sound IAR (hence AR) upper bound, with no optimality claim.
        best = None
        for support in self.local.provenance:
            edges = [e - support for e in self.touching_conflicts(support)]
            deleted = set()
            while edges:
                self.check_deadline()
                frequencies = Counter(i for e in edges for i in e)
                i = max(frequencies, key=lambda v: (frequencies[v], -v))
                deleted.add(i)
                edges = [e for e in edges if i not in e]
            if best is None or len(deleted) < len(best):
                best = deleted
        if not self.accepts_iar(best):
            raise AssertionError("invalid greedy upper bound")
        return best

    def maximize_kept(self):
        # Crucially, other deleted facts remain FREE in each test. Testing only
        # the exact set K+{t} is unsound for this nonmonotone feasibility family.
        full = set(range(self.local.n))
        for i in sorted(self.best):
            if i not in self.best:
                continue
            self.details["add_back_tests"] += 1
            candidate = self.feasible_model((full - self.best) | {i})
            if candidate is not None:
                if not candidate < self.best:
                    raise AssertionError("add-back model did not strictly reduce deletions")
                self.best = candidate
                self.trace("feasible_strict_subset")
        self.details["subset_minimal_certified"] = True

    def run(self):
        try:
            self.save()
            if not self.original.provenance:
                return self.save("INVALID_NOT_ORDINARY")
            if not self.original.consistent_provenance(indexed=self.indexed_families):
                return self.save("INFEASIBLE")
            self.local, self.ids = self.call("preprocessing", lambda: _relevant_subinstance(
                self.original, indexed_provenance=self.indexed_families))
            self.details["relevant_facts"] = self.local.n
            if self.ar_single_provenance and len(self.local.provenance) == 1:
                # The only repair-usable provenance set must belong to every
                # repair iff it belongs to their intersection, after any deletion.
                # Keep the requested semantics in SolveResult; change only the
                # exact encoding used to solve this equivalent subproblem.
                self.effective_semantics = "IAR"
                self.details["ar_single_provenance_reduced"] = True
                self.details["iar_encoding"] = self.iar_encoding
            self.incident = defaultdict(list)
            if self.indexed_families:
                self.edge_order = {e: i for i, e in enumerate(self.local.conflicts)}
            for edge in self.local.conflicts:
                for i in edge:
                    self.incident[i].append(edge)
            self.cut_keys = set()
            if self.effective_semantics == "AR":
                self.call("separator_encoding", self.build_separator)
                accepted = self.bad_repair(set(), "why_not_premise") is None
            else:
                accepted = self.call("why_not_premise", lambda: self.accepts_iar(set()))
            if accepted:
                self.best = set()
                return self.save("INVALID_ALREADY_ACCEPTED")
            # Certified fallback retains one consistent provenance set and all
            # irrelevant facts. It is published only AFTER the premise check.
            support = max(self.local.provenance, key=lambda s: (len(s), tuple(sorted(s))))
            self.best = set(range(self.local.n)) - support
            if not self.accepts_iar(self.best):
                raise AssertionError("invalid consistent-provenance fallback")
            self.trace("feasible_fallback")
            if self.objective == "minimum_cardinality" or self.subset_seed == "greedy_iar":
                self.best = self.call("initialization", self.greedy_incumbent)
                self.trace("greedy_upper_bound")
            self.call("master_encoding", self.build_master)
            if self.objective == "minimum_cardinality":
                candidate = self.feasible_model()
                if candidate is None:
                    raise AssertionError("master excluded known feasible incumbent")
                self.best = candidate
                self.trace("cardinality_optimum")
                return self.save("OPTIMAL")
            self.maximize_kept()
            self.trace("subset_minimality_proved")
            return self.save("SUBSET_MINIMAL")
        except DeadlineReached:
            return self.save()
        finally:
            if self.master is not None:
                self.master.delete()
            if self.separator is not None:
                self.separator.delete()


def solve_pysat(instance, semantics, objective="minimum_cardinality", time_limit=60., cut_mode="interval",
                subset_seed="consistent_provenance", repair_batch_size=1, iar_encoding="residual_simplified",
                ar_single_provenance=False, indexed_families=False):
    if semantics not in ("IAR", "AR") or objective not in ("minimum_cardinality", "subset_minimal"):
        raise ValueError("unsupported semantics or objective")
    if type(ar_single_provenance) is not bool or type(indexed_families) is not bool:
        raise ValueError("implementation options must be Boolean")
    if ar_single_provenance and semantics != "AR":
        raise ValueError("single-provenance AR reduction requires AR semantics")
    if cut_mode not in ("interval", "point") or not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("invalid cut mode or time budget")
    if subset_seed not in ("consistent_provenance", "greedy_iar"):
        raise ValueError("unsupported subset initialization")
    if type(repair_batch_size) is not int or repair_batch_size < 1:
        raise ValueError("repair batch size must be a positive integer")
    if repair_batch_size != 1 and (semantics != "AR" or cut_mode != "interval"):
        raise ValueError("repair batches require AR interval cuts")
    if iar_encoding not in ("direct_formula", "residual_simplified") or (
            semantics != "IAR" and iar_encoding != "residual_simplified"):
        raise ValueError("unsupported encoding for semantics")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sigmod-unified-") as directory:
        request, checkpoint = Path(directory)/"request.json", Path(directory)/"checkpoint.json"
        request.write_text(json.dumps({"n": instance.n, "conflicts": [sorted(e) for e in instance.conflicts],
            "provenance": [sorted(s) for s in instance.provenance], "target": instance.target,
            "semantics": semantics, "objective": objective, "deadline": started + time_limit,
            "request_started": started,
            "subset_seed": subset_seed,
            "repair_batch_size": repair_batch_size,
            "iar_encoding": iar_encoding,
            "ar_single_provenance": ar_single_provenance,
            "indexed_families": indexed_families,
            "cut_mode": cut_mode}))
        terminated = False
        try:
            remaining = started + time_limit - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("serialization", time_limit)
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(request), str(checkpoint)],
                capture_output=True, text=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            terminated = True
        if not terminated and done.returncode:
            return SolveResult(semantics, instance.target, "SOLVER_ERROR", elapsed_seconds=time.monotonic()-started,
                details={"method": "unified_pysat", "objective": objective, "stderr_tail": done.stderr[-3000:]})
        if checkpoint.exists():
            data = json.loads(checkpoint.read_text())
            active = data["details"].pop("active_phase_started")
            data["details"]["deadline_terminated"] = terminated
            if terminated:
                data["details"]["unfinished_phase_seconds"] = max(0., time.monotonic() - active)
            data["elapsed_seconds"] = time.monotonic() - started
            data["target"] = instance.target
            return SolveResult(**data)
        return SolveResult(semantics, instance.target, "TIMEOUT", elapsed_seconds=time.monotonic()-started,
            details={"method": "unified_pysat", "objective": objective, "phase": "worker_startup"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    data = json.loads(args.request.read_text())
    instance = ExplicitInstance(data["n"], tuple(map(frozenset, data["conflicts"])),
        tuple(map(frozenset, data["provenance"])), tuple(data["target"]))
    Engine(instance, data["semantics"], data["objective"], data["deadline"], args.checkpoint,
           data["cut_mode"], request_started=data["request_started"],
           subset_seed=data.get("subset_seed", "consistent_provenance"),
           repair_batch_size=data.get("repair_batch_size", 1),
           iar_encoding=data.get("iar_encoding", "residual_simplified"),
           ar_single_provenance=data.get("ar_single_provenance", False),
           indexed_families=data.get("indexed_families", False)).run()


if __name__ == "__main__":
    main()
