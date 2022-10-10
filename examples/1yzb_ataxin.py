import numpy as np
import sarw_spheres
import matplotlib.pyplot as plt

from saxs_single_bead.scattering_curve import scattering_curve_ensemble
from saxs_single_bead.read_pdb import read_c_alpha_pdb
from saxs_single_bead.replace_bead import replace_bead

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


# Read domain shape from 2nmo record in protein database
locations = read_c_alpha_pdb("1yzb.pdb")

sequence = list(
    "MESIFHEKQEGSLCAQHCLNNLLQGEYFSPVELSSIAHQLDEEERMRMAEGGVTSEDYRT\
FLQQPSGNMDDSGFFSIQVISNALKVWGLELILFNSPEYQRLRIDPINERSFICNYKEHW\
FTVRKLGKQWFNLNSLLTGPELISDTYLALFLAQLQQEGYSIFVVKGDLPDCEADQLLQM\
IRVQQMHRPKLIGEELAQLKEQRVHKTDLERVLEANDGSGMLDEDEEDLQRALALSRQEI\
DMEDEEADLRRAIQLSMQGSSRNISQDMTQTSGTNLTSEELRKRREAYFEKQQQKQQQQQ\
QQQQQGDLSGQSSHPCERPATSSGALGSDLGDAMSEEDMLQAAVTMSLETVRNDLKTEGK\
K"
)

# SAXS data from:
# The intrinsically disordered N-terminal domain of galectin-3 dynamically
# mediates multisite self-association of the protein through fuzzy interactions
# Yu-Hao Lin De-Chen Qiu Wen-Han Chang Yi-Qi Yeh U-Ser
# Jeng Fu-Tong Liu Jie-rong Huang
# Protein Structure and Folding, (2017)
# 10.1074/jbc.M117.802793
experiment = np.transpose(
    np.array(
        [
            [7.11e-3, 1.0, 1.0],
            [1.019e-2, 1.0, 1.0]
        ]
    )
)


ensemble_size = 10
n = len(sequence) - len(locations)
domain_ball_size = 20.0

ensemble = list()
sizes = np.hstack([1.9025 * np.ones(n), np.array([domain_ball_size])])

for i in range(ensemble_size):
    chain = sarw_spheres.generateChain(sizes)
    attachment = locations[-1] + (locations[-1] - locations[-2])
    chain_with_domain = replace_bead(locations, attachment, chain, sizes, 0)
    ensemble.append(chain_with_domain)

ensemble = np.array(ensemble)

# Generate average curves from conformers
(qvals, saxs_mean) = scattering_curve_ensemble(sequence, ensemble, points=50)

# Plotting
plt.plot(qvals, saxs_mean / saxs_mean[0], label="sarw_spheres $\\to$ saxs_single_bead")

#plt.errorbar(
#    experiment[0],
#    experiment[1] / experiment[1][0],
#    experiment[2] / experiment[1][0],
#    label="experiment",
#    capsize=2,
#)

plt.ylim([0.001, 1.2])
plt.yscale("log")
plt.xlabel("$q [\AA]$")
plt.ylabel("$I(q)$")
plt.legend()
plt.show()

residuals = np.interp(experiment[0], qvals, saxs_mean / saxs_mean[0]) - (
    experiment[1] / experiment[1][0]
)
scaled_residuals = residuals / (experiment[2] / experiment[1][0])
chi_stat = np.mean(scaled_residuals ** 2)
