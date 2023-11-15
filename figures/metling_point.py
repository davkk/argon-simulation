import tomllib

import numpy as np
from matplotlib import pyplot as plt

with open("gas.toml", "rb") as fp:
    params = tomllib.load(fp)
    n = params["n"]
    N = n * n * n
    L = params["L"]
    k_B = 8.31e-3  # nm^2 kg s^-2 K^-1
    V = 4 / 3 * np.pi * (L**3)

    T, P = np.loadtxt("./data/processed/P_vs_T_1700042688").T

    def ideal_P(t):
        return 3 / 2 * N * k_B * t / V

    plt.plot(T, P, "-o")
    plt.plot(T, ideal_P(T))

    plt.xlabel("$T$ [$K$]")
    plt.ylabel("$P(T)$ [$u\,nm^{-1}\,ps^{-2}$]")

    plt.show()
