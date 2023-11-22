import numpy as np
from matplotlib import pyplot as plt

from figures import common

common.setup_pyplot()

file = "./data/crystal_stability_1700645352.out"

steps, Hs = np.loadtxt(file).T
plt.plot(steps, Hs)
plt.xlim(0, 2)

plt.xlabel("$t$ [ps]")
plt.ylabel("$H$ [kJ/mol]")
plt.tight_layout()

# plt.show()
plt.savefig("./figures/crystal_stability.pdf")
