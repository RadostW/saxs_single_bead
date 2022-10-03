import matplotlib.pyplot as plt
import numpy as np
from saxs_single_bead.scattering_curve import scattering_curve
from saxs_single_bead.read_pdb import read_c_alpha_pdb

plt.rcParams.update(
    {
        "legend.frameon": False,
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 15,
        "figure.subplot.left": 0.125,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.95,
        "figure.subplot.wspace": 0.2,
        "figure.subplot.hspace": 0.2,
    }
)

sequence = list("T")

(qs, Is) = scattering_curve(sequence, np.array([[0, 0, 0]]), points=50)

plt.plot(qs, Is / Is[0], label="saxs_single_bead")
# plt.yscale("log")
plt.xlabel("$q [\AA]$")
plt.ylabel("$I(q)$")
plt.legend()
plt.show()
