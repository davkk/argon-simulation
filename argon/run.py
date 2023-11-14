import sys
import tomllib

from .argon import simulate


def main():
    with open(sys.argv[1], mode="rb") as fp:
        params = tomllib.load(fp)

        T_avg, P_avg, H_avg = simulate(
            params["n"],
            params["a"],
            int(sys.argv[2]) if len(sys.argv) == 3 else params["T0"],
            params["m"],
            params["L"],
            params["f"],
            params["e"],
            params["R"],
            params["tau"],
            params["S_o"],
            params["S_d"],
            params["S_out"],
            params["S_xyz"],
        )

        print("avg", T_avg, P_avg, H_avg, file=sys.stderr)


if __name__ == "__main__":
    main()
