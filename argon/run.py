import argparse
import sys
import tomllib
from pathlib import Path

from .argon import simulate


def main():
    parser = argparse.ArgumentParser(description="MD - Argon Simulation")
    parser.add_argument("parameters", metavar="path", type=Path)
    parser.add_argument("--tau", type=float)
    parser.add_argument("-T", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("-a", type=float)
    args = parser.parse_args()

    with open(str(args.parameters), mode="rb") as fp:
        params = tomllib.load(fp)

        T_avg, P_avg, H_avg = simulate(
            params["n"],
            args.a or params["a"],
            args.T or params["T0"],
            params["m"],
            params["L"],
            params["f"],
            params["e"],
            params["R"],
            args.tau or params["tau"],
            params["S_o"],
            args.steps or params["S_d"],
            params["S_out"],
            params["S_xyz"],
        )

        print("avg", T_avg, P_avg, H_avg, file=sys.stderr)


if __name__ == "__main__":
    main()
