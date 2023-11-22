import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from figures import common

common.setup_pyplot()

root_path = Path("data/processed/gas_1700042688/")
files = [int(file) for file in os.listdir(root_path)]
files.sort()

fig = plt.figure(figsize=(14, 14))
subfigs = fig.subfigures(len(files) // 2, 2)

for idx, file in enumerate(files):
    steps, Hs, Ps = np.loadtxt(root_path / str(file)).T
    subfig = subfigs.flat[idx]
    axs = subfig.subplots(2, 1, sharex=True)

    subfig.suptitle(f"$T=${file}")

    axs[0].plot(steps, Hs, "-+", mew=2, linewidth=0.7)
    axs[0].set_ylabel("$H$ [kJ/mol]")

    axs[1].plot(steps, Ps, "-+", mew=2, linewidth=0.7)
    axs[1].set_ylabel("$P$ [u nm$^{-1}$ ps$^{-2}$]")

    axs[1].set_xlabel("$t$ [ps]")

# plt.show()

plt.savefig("./figures/gas_stability.pdf", dpi=300, bbox_inches='tight')
