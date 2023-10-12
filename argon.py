# %%
import tomllib
from math import sqrt

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def simulate(*, n, a, T_0, m):
    N = n * n * n  # total number of atoms

    b0 = np.array([a, 0, 0], dtype=np.float64)
    b1 = np.array([a / 2, a * sqrt(3) / 2, 0], dtype=np.float64)
    b2 = np.array([a / 2, a * sqrt(3) / 6, a * sqrt(2 / 3)], dtype=np.float64)

    r0 = np.empty((N, 3))
    for i0 in range(n):
        for i1 in range(n):
            for i2 in range(n):
                r0[i0 + i1 * n + i2 * n * n] = (
                    (i0 - (n - 1) / 2) * b0
                    + (i1 - (n - 1) / 2) * b1
                    + (i2 - (n - 1) / 2) * b2
                )

    k = 8.31e-3  # nm^2 kg s^-2 K^-1

    E_k = np.empty((N, 3))
    for i in range(N):
        E_k[i] = -0.5 * k * T_0 * np.log(np.random.random(3))

    p_0 = np.empty((N, 3))
    for i in range(N):
        signs = np.random.choice(np.array([-1, 1]), 3)
        p_0[i] = signs * np.sqrt(2 * m * E_k[i])

    P = np.sum(p_0, axis=0)
    p_0 = p_0 - P / N

    return p_0


# %%
with open("parameters.toml", mode="rb") as fp:
    params = tomllib.load(fp)

    result = simulate(
        n=params["n"],
        a=params["a"],  # nm
        T_0=params["T_0"],  # K
        m=params["m"],  # kg
    )

    np.savetxt("p0", result)
