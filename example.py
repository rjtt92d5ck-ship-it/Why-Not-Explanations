import itertools
import json
from exact_solvers import ExplicitInstance
from materializer import (Atom, Builtin, ConjunctiveQuery, Const, DenialConstraint,
                          FactStore, Var, materialize_conflicts, materialize_query)
from unified_pysat import solve_pysat


def main():
    store = FactStore()
    store.register('R', 1)
    for value in range(3):
        store.add('R', (value,))
    variables = [Var('x'+str(i)) for i in range(3)]
    dc = DenialConstraint('at_most_two', tuple(Atom('R', (x,)) for x in variables),
        tuple(Builtin('ne', a, b) for a, b in itertools.combinations(variables, 2)))
    cq = ConjunctiveQuery('both_present', (), (Atom('R', (Const(0),)), Atom('R', (Const(1),))))
    mis = materialize_conflicts(store, (dc,)).mis
    provenance = materialize_query(store, cq).provenance[()]
    relational = ExplicitInstance(3, tuple(mis), tuple(provenance), ())
    assert set(mis) == {frozenset({0, 1, 2})}
    assert set(provenance) == {frozenset({0, 1})}
    multiple = ExplicitInstance(3, (frozenset({0, 2}), frozenset({1, 2})),
                                  (frozenset({0}), frozenset({1})), ('multiple',))
    for name, instance in [('relational', relational), ('multiple_provenance', multiple)]:
        for semantics in ('IAR', 'AR'):
            for objective in ('minimum_cardinality', 'subset_minimal'):
                result = solve_pysat(instance, semantics, objective, 10,
                    indexed_families=True, ar_single_provenance=(semantics == 'AR'))
                expected = 'OPTIMAL' if objective == 'minimum_cardinality' else 'SUBSET_MINIMAL'
                assert result.status == expected, result
                assert set(result.deletions) == {2}, result
                print(json.dumps({'example': name, 'semantics': semantics,
                    'objective': objective, 'status': result.status, 'deletions': result.deletions}))


if __name__ == '__main__':
    main()
