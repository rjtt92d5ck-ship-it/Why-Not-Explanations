import argparse
import json
from pathlib import Path

from exact_solvers import ExplicitInstance
from unified_pysat import solve_pysat


def load_instance(data, target):
    if data.get('format_version') == 2:
        if target is None:
            raise ValueError('--target is required for a relational materialization')
        return ExplicitInstance.from_data(data, tuple(target))
    n = data['n']
    if type(n) is not int or n < 0:
        raise ValueError('n must be a nonnegative integer')
    families = []
    for key in ('mis', 'provenance_sets'):
        family = []
        for members in data[key]:
            if not members or any(type(v) is not int or not 0 <= v < n for v in members):
                raise ValueError('Each set must contain valid tuple IDs in [0, n)')
            if len(set(members)) != len(members):
                raise ValueError('Repeated tuple ID in a set')
            family.append(frozenset(members))
        if len(set(family)) != len(family):
            raise ValueError('Duplicate sets in '+key)
        families.append(tuple(family))
    return ExplicitInstance(n, families[0], families[1],
                            tuple(target if target is not None else data.get('target', [])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path, help='JSON input file')
    inputs.add_argument('--instance', help='Inline JSON instance; no input file needed')
    parser.add_argument('--target', help='JSON answer tuple, required for version-2 materializations')
    parser.add_argument('--semantics', choices=('IAR', 'AR'), required=True)
    parser.add_argument('--objective', choices=('minimum_cardinality', 'subset_minimal'),
                        default='minimum_cardinality')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--cut', choices=('interval', 'point'), default='interval')
    parser.add_argument('--indexed', action='store_true')
    parser.add_argument('--single-provenance-reduction', action='store_true',
                        help='Enable the conditional AR-to-IAR reduction (AR only)')
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text() if args.input else args.instance)
        instance = load_instance(data, json.loads(args.target) if args.target else None)
        result = solve_pysat(instance, args.semantics, args.objective, args.seconds,
                             cut_mode=args.cut, indexed_families=args.indexed,
                             ar_single_provenance=args.single_provenance_reduction)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == '__main__':
    main()
