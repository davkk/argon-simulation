# %%
import tomllib
from math import sqrt

import numpy as np
from numba import njit


# %%
@njit(parallel=True, fastmath=True, cache=True)
def calc_static(r: np.ndarray, N: int, f: float, L: float, e: float, R: float):
    F_s = np.zeros((N, 3))
    F_p = np.zeros((N, 3))

    P = 0
    V_s = 0
    V_p = 0

    for i in range(N):
        r_i = np.linalg.norm(r[i])
        if r_i >= L:
            V_s += 0.5 * f * (r_i - L) * (r_i - L)
            F_s[i] = f * (L - r_i) * r[i] / r_i

        for j in range(i):
            r_ij = np.linalg.norm(r[i] - r[j])
            V_p += e * (np.power(R / r_ij, 12) - 2 * np.power(R / r_ij, 6))
            F_p[i] += (
                12
                * e
                * (np.power(R / r_ij, 12) - np.power(R / r_ij, 6))
                * (r[i] - r[j])
                / r_ij
                / r_ij
            )

    P = np.sum(np.sqrt(np.sum(F_s * F_s, axis=1))) / (4 * np.pi * L * L)

    return F_s + F_p, V_s + V_p, P


# %%
@njit(parallel=True, fastmath=True, cache=True)
def simulate(
    *,
    n: int,
    a: float,
    T0: int,
    m: int,
    L: float,
    f: float,
    e: float,
    R: float,
    tau: float,
    S_o: int,
    S_d: int,
    S_out: int,
    S_xyz: int,
):
    N = n * n * n  # total number of atoms

    b0 = np.array([a, 0, 0], dtype=np.float64)
    b1 = np.array([a / 2, a * sqrt(3) / 2, 0], dtype=np.float64)
    b2 = np.array([a / 2, a * sqrt(3) / 6, a * sqrt(2 / 3)], dtype=np.float64)

    r = np.empty((N, 3))
    for i0 in range(n):
        for i1 in range(n):
            for i2 in range(n):
                r[i0 + i1 * n + i2 * n * n] = (
                    (i0 - (n - 1) / 2) * b0
                    + (i1 - (n - 1) / 2) * b1
                    + (i2 - (n - 1) / 2) * b2
                )

    k = 8.31e-3  # nm^2 kg s^-2 K^-1

    E_k0 = -0.5 * k * T0 * np.log(np.random.random((N, 3)))

    signs = np.random.choice(np.array([-1, 1]), (N, 3))
    p = signs * np.sqrt(2 * m * E_k0)

    p = np.sum(p, axis=0)
    p = p - p / N

    stats_out = np.array([])
    stats_xyz = np.array([])

    F, V_s, P_s = calc_static(r=r, N=N, f=f, L=L, e=e, R=R)

    for s in np.arange(S_o + S_d):
        p_half = p + 0.5 * F * tau
        r = r + p_half * tau / m

        # F, V_s, P_s = calc_static(r=r, N=N, f=f, L=L, e=e, R=R)
        p = p_half + 0.5 * F * tau

        E_k = np.sum(p * p) / (2 * m)
        H_s =  E_k + V_s
        T_s = 2 / (3 * N * k) * E_k

        if s % S_out:
            stats_out.append(s, V_s, P_s, H_s, T_s)

        if s % S_xyz:
            stats_xyz.append(*(np.column_stack((r, E_k))))

    return stats_out, stats_xyz



# %%
def main():
    with open("parameters.toml", mode="rb") as fp:
        params = tomllib.load(fp)

        simulate(
            n=params["n"],
            a=params["a"],  # nm
            T0=params["T0"],  # K
            m=params["m"],  # kg
            L=params["L"],  # nm
            f=params["f"],
            e=params["e"],
            R=params["R"],
            tau=params["tau"],
            S_o=params["S_o"],
            S_d=params["S_d"],
            S_out=params["S_out"],
            S_xyz=params["S_xyz"],
        )


if __name__ == "__main__":
    main()
