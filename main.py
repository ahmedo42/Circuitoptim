import argparse
import pickle
import random
from collections import OrderedDict

import numpy as np
from scipy.optimize import differential_evolution

from blackbox import BlackBox
from interface.eval_engines.ngspice.TwoStageClass import *

parser = argparse.ArgumentParser()
parser.add_argument("--combpb", type=float, default=0.5)
parser.add_argument("--mutpb", type=float, default=0.5)
parser.add_argument("--pop", type=int, default=10)
parser.add_argument("--seed", type=int, default=15)
parser.add_argument("--ngen", type=int, default=50)
parser.add_argument("--env", type=str, default="two_stage_opamp")
args = parser.parse_args()


def load_valid_specs():
    with open("specs_valid_two_stage_opamp", "rb") as f:
        specs = pickle.load(f)

    specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
    return specs


def stopping_criteria(xk, convergence):
    cost = box.simulate(xk)
    return cost <= 0.02


def evaluate(box, bounds):
    specs = load_valid_specs()
    designs_met = 0
    n_evals = 0
    n_specs = len(list(specs.values())[0])
    random.seed(args.seed)
    for i in range(n_specs):
        target_specs = [spec[i] for spec in specs.values()]
        setattr(box, "target_specs", target_specs)
        result = differential_evolution(
            box.simulate,
            bounds,
            seed=args.seed,
            maxiter=args.ngen,
            disp=True,
            popsize=args.pop,
            mutation=args.mutpb,
            recombination=args.combpb,
            callback=stopping_criteria,
        )
        n_evals += result.nfev
        print(result.nfev)
        if result.fun <= 0.02:
            designs_met += 1
        print(f"total achieved designs {designs_met} / {i+1}")

    print(designs_met)
    print(n_evals / 100)


if __name__ == "__main__":
    CIR_YAML = (
        f"interface/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"
    )
    sim_env = TwoStageClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd())
    box = BlackBox(sim_env, CIR_YAML)
    bounds = [(0, len(param)) for param in box.params]
    evaluate(box, bounds)
