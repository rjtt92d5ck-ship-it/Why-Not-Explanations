# Exact Counterfactual Explanations under IAR and AR

Implementation of tuple-deletion Why-Not explanations for conjunctive
queries under denial constraints. Repairs are recomputed after deletion.
Both **subset-minimal** and **minimum-cardinality** explanations are supported.

This is a methods-only release. Dataset adapters, experiment configurations,
benchmark runners, archived results, statistical analysis, and plotting scripts
are not included. Reproducing the full experimental evaluation requires the
separate experiment package and its data.

## Files

| File | Role |
| --- | --- |
| `unified_pysat.py` | Primary IAR encoding and repair-guided AR search, RC2/Glucose4, both objectives |
| `exact_solvers.py` | Shared instance representation, residual hitting sets, structural preprocessing, Z3 AR-Benders and component methods |
| `materializer.py` | Compute MISs from DCs and provenance sets from CQs |
| `rc2_worker.py` | Isolated MaxSAT worker used by the structural solver module |
| `solve.py` | Command-line interface to the primary solver |
| `example.py` | Executable small examples without datasets |
| `README.md` | Installation and usage |

The four implementation modules retain the algorithms from the experimental
source package; their introductory module descriptions have been removed. `solve.py` and `example.py` are lightweight entry points.
The primary implementation uses PySAT's RC2 and Glucose4; the structural methods
in `exact_solvers.py` additionally use Z3. These solver engines are dependencies,
not implementations distributed in this repository.

## Install and try

The recorded environment used Python 3.14.2. Run from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install python-sat==1.9.dev15 z3-solver==5.1.0.0
python example.py
```

The example first grounds a three-tuple DC and a two-tuple CQ provenance set.
It then exercises a target with multiple provenance sets, including general AR
separation. It prints eight successful requests across two examples, both
semantics, and both objectives. Each returned deletion set is `{2}`.

## Solve your own instance

Tuple IDs are consecutive integers from `0` to `n-1`. `mis` contains the complete
family of minimal inconsistent subsets; `provenance_sets` contains the complete
minimal provenance sets for the selected target. Provenance sets need not be
consistent. These are distinct families. Input generation is the caller's
responsibility; `materializer.py` provides a relational implementation.

A compact instance can be passed directly without any JSON files:

```bash
python solve.py \
  --instance '{"n":3,"mis":[[0,1,2]],"provenance_sets":[[0,1]],"target":[]}' \
  --semantics AR --objective minimum_cardinality --seconds 60
```

Use `--semantics IAR` for IAR or `--objective subset_minimal` for inclusion
minimality. `--indexed` enables indexed family operations.
`--single-provenance-reduction` enables the conditional AR-to-IAR reduction and
is valid only with AR. `--cut point` selects point cuts instead of interval cuts.

For a compact instance stored in a file, use `--input input.json` in place of
`--instance`. Existing complete version-2 relational materializations are also
accepted with `--input`, together with `--target '[...]'`. No input or result JSON
files are bundled; JSON remains an optional runtime interchange format.

The Python API is:

```python
from exact_solvers import ExplicitInstance
from unified_pysat import solve_pysat

instance = ExplicitInstance(
    n=3,
    conflicts=(frozenset({0, 1, 2}),),
    provenance=(frozenset({0, 1}),),
    target=(),
)
result = solve_pysat(instance, "AR", "subset_minimal", time_limit=60)
print(result.as_dict())
```

`deletions` contains the IDs of tuples to remove, not the retained database.
`OPTIMAL` certifies minimum cardinality; `SUBSET_MINIMAL` certifies that no strict
subset is feasible. `FEASIBLE_NOT_PROVEN` denotes an incumbent whose requested
guarantee is unfinished. A timeout does not establish nonexistence.
`INVALID_ALREADY_ACCEPTED` denotes a target already accepted under the requested
semantics. Do not count unproved candidates as exact completed solutions.

Only the two solver dependencies above are needed; plotting, dataframe, and
process-monitoring packages are not required. Third-party solver dependencies retain their upstream licenses.
