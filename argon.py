# %%
import tomllib
from math import sqrt

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def simulate(*, n, a, T0, m):
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

    E_k = -0.5 * k * T0 * np.log(np.random.random((N, 3)))

    signs = np.random.choice(np.array([-1, 1]), (N, 3))
    p0 = signs * np.sqrt(2 * m * E_k)

    P = np.sum(p0, axis=0)
    p0 = p0 - P / N

    return r0, p0


# %%
with open("parameters.toml", mode="rb") as fp:
    params = tomllib.load(fp)

    result_r0, result_p0 = simulate(
        n=params["n"],
        a=params["a"],  # nm
        T0=params["T0"],  # K
        m=params["m"],  # kg
    )

    np.savetxt("r0.out", result_r0)
    np.savetxt("p0.out", result_p0)
