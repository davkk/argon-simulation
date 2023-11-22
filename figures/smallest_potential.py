from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from figures import common

common.setup_pyplot()

ROOT = Path("data/")

a, V = np.loadtxt(ROOT / "crystal_V_vs_a_1700644142.out").T

a_min_idx = np.argmin(V)
a_min = a[a_min_idx]

plt.plot(a, V, "-o")
plt.annotate(
    f"$a\\approx{a_min}$",
    xy=(a_min, V[a_min_idx]),
    xytext=(10, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->"),
)

plt.plot([a_min], [V[a_min_idx]], "o", color="black")
plt.xlabel("$a$ [nm]")
plt.ylabel("$V$ [kJ/mol]")
plt.tight_layout()

# plt.show()
plt.savefig("./figures/plot_V_vs_a.pdf")
