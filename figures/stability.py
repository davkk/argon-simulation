import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from figures import common

common.setup_pyplot()

root_path = Path("./data/processed/stability_1700677133/")
files = os.listdir(root_path)
files.sort()

fig, axs = plt.subplots(len(files), 1, figsize=(9, 10))

for idx, file in enumerate(files):
    steps, Hs = np.loadtxt(root_path / file).T
    axs[idx].plot(steps, Hs)
    axs[idx].set_xlim(0, 1.4)
    axs[idx].set_title(f"$\\tau=${file.split('=')[1]}")

fig.supxlabel("$t$ [ps]")
fig.supylabel("$H$ [kJ/mol]")
fig.tight_layout()
# plt.show()

plt.savefig("./figures/stability.pdf")
