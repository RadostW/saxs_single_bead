import numpy as np
import saxs_single_bead.scattering_curve
import sarw_spheres
import matplotlib.pyplot as plt

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
