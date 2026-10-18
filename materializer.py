from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Iterator


Value = Hashable


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Const:
    value: Value


Term = Var | Const | None  # None is an ignored relational position.


@dataclass(frozen=True)
class Atom:
    relation: str
    terms: tuple[Term, ...]


@dataclass(frozen=True)
class Builtin:
    op: str  # eq, ne, lt, le, gt, ge
    left: Var | Const
    right: Var | Const


@dataclass(frozen=True)
class DenialConstraint:
    name: str
    atoms: tuple[Atom, ...]
    builtins: tuple[Builtin, ...] = ()


@dataclass(frozen=True)
class ConjunctiveQuery:
    name: str
    head: tuple[Var, ...]
    atoms: tuple[Atom, ...]
    builtins: tuple[Builtin, ...] = ()


@dataclass(frozen=True)
class Fact:
    id: int
    relation: str
    values: tuple[Value, ...]


class BudgetExceeded(RuntimeError):
    """No exact result is available because the grounding budget was reached."""

    def __init__(self, states: int, matches: int, limit: str):
        self.states = states
        self.matches = matches
        super().__init__(f"{limit} exceeded (join_states={states}, raw_matches={matches})")


@dataclass(frozen=True)
class Limits:
    max_join_states: int = 10_000_000
    max_raw_matches: int = 2_000_000


@dataclass
class EvaluationStats:
    join_states: int = 0
    raw_matches: int = 0


class FactStore:
    def __init__(self) -> None:
        self.schemas: dict[str, int] = {}
        self.facts: list[Fact] = []
        self._ids: dict[tuple[str, tuple[Value, ...]], int] = {}
        self._by_relation: dict[str, list[int]] = defaultdict(list)
        self._indexes: dict[tuple[str, int], dict[Value, list[int]]] = {}

    def register(self, relation: str, arity: int) -> None:
        if arity < 1:
            raise ValueError("relations must have positive arity")
        if relation in self.schemas and self.schemas[relation] != arity:
            raise ValueError(f"inconsistent arity for {relation}")
        self.schemas[relation] = arity

    def add(self, relation: str, values: Iterable[Value]) -> int:
        row = tuple(values)
        self.register(relation, len(row))
        try:
            existing = self._ids.get((relation, row))
        except TypeError as exc:
            raise TypeError("fact values must be hashable, fixed constants") from exc
        if existing is not None:
            return existing
        fact_id = len(self.facts)
        self.facts.append(Fact(fact_id, relation, row))
        self._ids[(relation, row)] = fact_id
        self._by_relation[relation].append(fact_id)
        for (indexed_relation, position), index in self._indexes.items():
            if indexed_relation == relation:
                index.setdefault(row[position], []).append(fact_id)
        return fact_id

    def relation_ids(self, relation: str) -> list[int]:
        return self._by_relation[relation]

    def lookup(self, relation: str, position: int, value: Value) -> list[int]:
        key = (relation, position)
        if key not in self._indexes:
            index: dict[Value, list[int]] = defaultdict(list)
            for fact_id in self._by_relation[relation]:
                index[self.facts[fact_id].values[position]].append(fact_id)
            self._indexes[key] = index
        return self._indexes[key].get(value, [])


class _Antichain:
    def __init__(self) -> None:
        self.sets: set[frozenset[int]] = set()
        self.postings: dict[int, set[frozenset[int]]] = defaultdict(set)

    def add(self, support: frozenset[int]) -> None:
        if not support:
            raise ValueError("a DC/CQ support must contain a relational fact")
        if support in self.sets:
            return
        if len(support) <= 12:
            for size in range(1, len(support)):
                for subset in itertools.combinations(support, size):
                    if frozenset(subset) in self.sets:
                        return
        elif any(old <= support for old in self.sets):
            return

        postings = [self.postings[fact_id] for fact_id in support]
        supersets = set.intersection(*postings) if postings else set()
        for old in supersets:
            self.sets.remove(old)
            for fact_id in old:
                self.postings[fact_id].remove(old)
        self.sets.add(support)
        for fact_id in support:
            self.postings[fact_id].add(support)

    def sorted(self) -> tuple[frozenset[int], ...]:
        return tuple(sorted(self.sets, key=lambda s: (len(s), tuple(sorted(s)))))


def _validate(
    store: FactStore,
    atoms: tuple[Atom, ...],
    builtins: tuple[Builtin, ...],
    head: tuple[Var, ...] = (),
) -> None:
    if not atoms:
        raise ValueError("a DC/CQ must have at least one relational atom")
    variables: set[str] = set()
    for atom in atoms:
        if atom.relation not in store.schemas:
            raise ValueError(f"unregistered relation {atom.relation}")
        if len(atom.terms) != store.schemas[atom.relation]:
            raise ValueError(f"arity mismatch for {atom.relation}")
        variables.update(term.name for term in atom.terms if isinstance(term, Var))
    for condition in builtins:
        if condition.op not in {"eq", "ne", "lt", "le", "gt", "ge"}:
            raise ValueError(f"unsupported built-in comparison {condition.op}")
        for term in (condition.left, condition.right):
            if isinstance(term, Var) and term.name not in variables:
                raise ValueError(f"unbound built-in variable {term.name}")
    for term in head:
        if not isinstance(term, Var) or term.name not in variables:
            raise ValueError("CQ head terms must be variables in relational atoms")


def _value(term: Var | Const, bindings: dict[str, Value]) -> tuple[bool, Value | None]:
    if isinstance(term, Const):
        return True, term.value
    if isinstance(term, Var):
        return term.name in bindings, bindings.get(term.name)
    raise TypeError("built-in terms must be Var or Const")


def _test(condition: Builtin, bindings: dict[str, Value]) -> bool | None:
    left_ready, left = _value(condition.left, bindings)
    right_ready, right = _value(condition.right, bindings)
    if not left_ready or not right_ready:
        return None
    if condition.op == "eq":
        return left == right
    if condition.op == "ne":
        return left != right
    try:
        if condition.op == "lt":
            return left < right
        if condition.op == "le":
            return left <= right
        if condition.op == "gt":
            return left > right
        if condition.op == "ge":
            return left >= right
    except TypeError as exc:
        raise TypeError(f"incompatible values in built-in {condition.op}: {left!r}, {right!r}") from exc
    raise AssertionError("unvalidated built-in")


def _candidates(store: FactStore, atom: Atom, bindings: dict[str, Value]) -> list[int]:
    selections = []
    for position, term in enumerate(atom.terms):
        if isinstance(term, Const):
            selections.append(store.lookup(atom.relation, position, term.value))
        elif isinstance(term, Var) and term.name in bindings:
            selections.append(store.lookup(atom.relation, position, bindings[term.name]))
    return min(selections, key=len) if selections else store.relation_ids(atom.relation)


def _extend(
    atom: Atom, fact: Fact, bindings: dict[str, Value]
) -> dict[str, Value] | None:
    extended = bindings.copy()
    for term, value in zip(atom.terms, fact.values):
        if isinstance(term, Const):
            if term.value != value:
                return None
        elif isinstance(term, Var):
            if term.name in extended:
                if extended[term.name] != value:
                    return None
            else:
                extended[term.name] = value
    return extended


def _ground(
    store: FactStore,
    atoms: tuple[Atom, ...],
    builtins: tuple[Builtin, ...],
    initial: dict[str, Value],
    limits: Limits,
    stats: EvaluationStats,
) -> Iterator[tuple[frozenset[int], dict[str, Value]]]:
    def visit(
        remaining: tuple[int, ...],
        bindings: dict[str, Value],
        support: frozenset[int],
    ) -> Iterator[tuple[frozenset[int], dict[str, Value]]]:
        stats.join_states += 1
        if stats.join_states > limits.max_join_states:
            raise BudgetExceeded(stats.join_states, stats.raw_matches, "join-state budget")
        tests = [_test(condition, bindings) for condition in builtins]
        if False in tests:
            return
        if not remaining:
            if None in tests:
                raise AssertionError("built-in variable remained unbound")
            stats.raw_matches += 1
            if stats.raw_matches > limits.max_raw_matches:
                raise BudgetExceeded(stats.join_states, stats.raw_matches, "raw-match budget")
            yield support, bindings
            return
        chosen = min(remaining, key=lambda i: len(_candidates(store, atoms[i], bindings)))
        atom = atoms[chosen]
        for fact_id in _candidates(store, atom, bindings):
            extended = _extend(atom, store.facts[fact_id], bindings)
            if extended is not None:
                yield from visit(
                    tuple(i for i in remaining if i != chosen),
                    extended,
                    support | {fact_id},
                )

    yield from visit(tuple(range(len(atoms))), initial, frozenset())


@dataclass(frozen=True)
class ConflictResult:
    mis: tuple[frozenset[int], ...]
    stats: EvaluationStats


@dataclass(frozen=True)
class QueryResult:
    provenance: dict[tuple[Value, ...], tuple[frozenset[int], ...]]
    stats: EvaluationStats


def materialize_conflicts(
    store: FactStore,
    constraints: Iterable[DenialConstraint],
    limits: Limits = Limits(),
) -> ConflictResult:
    family = _Antichain()
    stats = EvaluationStats()
    for constraint in constraints:
        _validate(store, constraint.atoms, constraint.builtins)
        for support, _ in _ground(
            store, constraint.atoms, constraint.builtins, {}, limits, stats
        ):
            family.add(support)
    return ConflictResult(family.sorted(), stats)


def materialize_query(
    store: FactStore,
    query: ConjunctiveQuery,
    target: tuple[Value, ...] | None = None,
    limits: Limits = Limits(),
) -> QueryResult:
    _validate(store, query.atoms, query.builtins, query.head)
    initial: dict[str, Value] = {}
    if target is not None:
        if len(target) != len(query.head):
            raise ValueError("target arity does not match CQ head")
        for variable, value in zip(query.head, target):
            if variable.name in initial and initial[variable.name] != value:
                return QueryResult({}, EvaluationStats())
            initial[variable.name] = value
    families: dict[tuple[Value, ...], _Antichain] = defaultdict(_Antichain)
    stats = EvaluationStats()
    for support, bindings in _ground(
        store, query.atoms, query.builtins, initial, limits, stats
    ):
        answer = tuple(bindings[variable.name] for variable in query.head)
        families[answer].add(support)
    return QueryResult(
        {answer: family.sorted() for answer, family in families.items()}, stats
    )
