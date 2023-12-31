import numba as nb
import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC("argon")


@njit(cache=True)
@cc.export("calc_static", "Array(f8, 2, 'C'), i8, f8, f8, f8, f8")
def calc_static(r: np.ndarray, N: int, f: float, L: float, e: float, R: float):
    F = np.zeros((N, 3))
    P = 0
    V = 0

    for i in nb.prange(N):
        r_i = np.sqrt(r[i] @ r[i])
        if r_i >= L:
            V += 0.5 * f * (r_i - L) * (r_i - L)
            F[i] = f * (L - r_i) * r[i] / r_i
            P += np.sqrt(F[i] @ F[i])

        for j in nb.prange(i):
            dr = r[i] - r[j]
            r_ij = np.sqrt(dr @ dr)

            R_r_ij = R / r_ij
            R_r_ij_6 = R_r_ij * R_r_ij * R_r_ij * R_r_ij * R_r_ij * R_r_ij
            R_r_ij_12 = R_r_ij_6 * R_r_ij_6

            V += e * (R_r_ij_12 - 2 * R_r_ij_6)

            coeff = 12 * e * (R_r_ij_12 - R_r_ij_6) * dr / r_ij / r_ij
            F[i] += coeff
            F[j] -= coeff

    P /= 4 * np.pi * L * L

    return F, V, P


@njit(cache=True)
@cc.export("simulate", "i8, f8, i8, i8, f8, f8, f8, f8, f8, i8, i8, i8, i8")
def simulate(
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
    b1 = np.array([a / 2, a * np.sqrt(3) / 2, 0], dtype=np.float64)
    b2 = np.array(
        [a / 2, a * np.sqrt(3) / 6, a * np.sqrt(2 / 3)], dtype=np.float64
    )

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

    p = p - np.sum(p, axis=0) / N

    F, V, P = calc_static(r, N, f, L, e, R)

    T_avg, P_avg, H_avg = 0, 0, 0

    E_k = np.sum(p * p) / (2 * m)

    H = E_k + V
    T = 2 / (3 * N * k) * E_k
    print("out", 0, H, V, T, P)

    print("xyz", N)
    print("xyz")
    for i in range(N):
        print("xyz", "atom", r[i][0], r[i][1], r[i][2], E_k)

    for s in range(1, S_o + S_d):
        p += 0.5 * F * tau
        r += p * tau / m

        F, V, P = calc_static(r, N, f, L, e, R)
        p += 0.5 * F * tau

        E_k = np.sum(p * p) / (2 * m)
        H = E_k + V
        T = 2 / (3 * N * k) * E_k

        if s % S_out == 0:
            print("out", s * tau, H, V, T, P)

        if s % S_xyz == 0:
            print("xyz", N)
            print("xyz")
            for i in range(N):
                print("xyz", "atom", r[i][0], r[i][1], r[i][2], E_k)

        if s >= S_o:
            T_avg += T
            P_avg += P
            H_avg += H

    return T_avg / S_d, P_avg / S_d, H_avg / S_d


def main():
    cc.compile()


if __name__ == "__main__":
    main()
