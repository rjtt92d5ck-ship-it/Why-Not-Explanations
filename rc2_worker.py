from __future__ import annotations

import argparse
import json

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formula")
    parser.add_argument("--deletion-variables", type=int, required=True)
    args = parser.parse_args()
    formula = WCNF(from_file=args.formula)
    with RC2(formula, solver="g4") as rc2:
        model = rc2.compute()
        if model is None:
            print(json.dumps({"status": "unsat"}))
            return
        positive = {literal for literal in model if literal > 0}
        deletion = sorted(variable for variable in positive
                          if 1 <= variable <= args.deletion_variables)
        print(json.dumps({"status": "optimal", "deletion_variables": deletion,
                          "cost": rc2.cost}))


if __name__ == "__main__":
    main()
