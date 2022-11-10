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
            [8.091e-3, 0.9999999999999999],
            [1.3589e-2, 1.0160567941437804],
            [1.9088e-2, 0.886896573498171],
            [2.4586e-2, 0.8157276148293814],
            [3.0084e-2, 0.6695025400705716],
            [3.5582e-2, 0.6143212170689134],
            [4.108e-2, 0.4862382148693731],
            [4.6578e-2, 0.4254497932011203],
            [5.2076e-2, 0.3708318110951244],
            [5.7575e-2, 0.32128641532070734],
            [6.3073e-2, 0.281928514190267],
            [6.8571e-2, 0.24726647362431006],
            [7.4069e-2, 0.21167252258550467],
            [7.9567e-2, 0.18591414726846567],
            [8.5065e-2, 0.16664430292014148],
            [9.0563e-2, 0.1428565307401911],
            [9.6061e-2, 0.12902731609483128],
            [0.101559, 0.11284381911149398],
            [0.107058, 0.10384600061039871],
            [0.112556, 8.814830433939372e-2],
            [0.118054, 7.908771700351354e-2],
            [0.123552, 6.801232524329501e-2],
            [0.12905, 6.348428674307174e-2],
            [0.134548, 5.282392101500588e-2],
            [0.140046, 4.8543086482675045e-2],
            [0.145544, 4.49765011523907e-2],
            [0.151042, 3.765423467876639e-2],
            [0.156541, 3.2813141313319465e-2],
            [0.162039, 3.194809656327474e-2],
            [0.167537, 2.7798212103034098e-2],
            [0.173035, 2.433168104711947e-2],
            [0.178533, 2.2725325082426376e-2],
            [0.184031, 2.021291790137099e-2],
            [0.189529, 1.7901032716469787e-2],
            [0.195027, 1.8147635306304394e-2],
            [0.200526, 1.7588804746075628e-2],
            [0.206024, 1.732074047680259e-2],
            [0.211522, 1.563831018779533e-2],
            [0.21702, 1.4116071978939737e-2],
            [0.222518, 1.342264549214525e-2],
            [0.228016, 1.3484437087586202e-2],
            [0.233514, 1.2081422078693324e-2],
            [0.239012, 1.2096193427238517e-2],
            [0.24451, 1.033385502879849e-2],
            [0.250009, 1.1580474156529687e-2],
            [0.255507, 9.486400586944983e-3],
            [0.261005, 1.0830142271014026e-2],
            [0.266503, 1.0626463040056287e-2],
            [0.272001, 1.0604324810302808e-2],
            [0.277499, 1.0011854664964764e-2],
            [0.282997, 1.0686300156809327e-2],
            [0.288495, 7.481330969917565e-3],
            [0.293993, 8.135329607796263e-3],
            [0.299492, 8.837701593200818e-3],
            [0.30499, 9.896502836249264e-3],
            [0.310488, 7.34128505470285e-3],
            [0.315986, 6.56516908495818e-3],
            [0.321484, 8.562082512076422e-3],
            [0.326982, 8.40083802032357e-3],
            [0.33248, 6.413847331159351e-3],
            [0.337978, 9.20717323747367e-3],
            [0.343477, 8.233241472835e-3],
            [0.348975, 6.105565904269633e-3],
            [0.354473, 6.561560816611264e-3],
            [0.359971, 6.0437367227000665e-3],
            [0.365469, 7.684145719917189e-3],
            [0.370967, 7.849148824531412e-3],
            [0.376465, 6.9770754684359195e-3],
            [0.381963, 5.937067289694334e-3],
            [0.387461, 7.886246333473153e-3],
            [0.39296, 7.19188019346332e-3],
            [0.398458, 7.212364633557797e-3],
            [0.403956, 6.4965744002381445e-3],
            [0.409454, 5.525536767502732e-3],
            [0.414952, 6.887658068463884e-3],
            [0.42045, 6.072753213989856e-3],
            [0.425948, 6.344951957410405e-3],
            [0.431446, 5.617171749062977e-3],
            [0.436945, 7.1790633236060416e-3],
            [0.442443, 6.425085583614853e-3],
            [0.447941, 5.935150397135033e-3],
            [0.453439, 6.784145870261704e-3],
            [0.458937, 7.665803689153695e-3],
            [0.464435, 6.422266623968824e-3],
            [0.469933, 5.166889928270631e-3],
            [0.475431, 5.753684568188003e-3],
            [0.480929, 6.201974324163821e-3],
            [0.486428, 7.1680130017936095e-3],
            [0.491926, 5.871742598163691e-3],
            [0.497424, 6.51269884941343e-3],
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

# Plotting
plt.plot(qvals, saxs_mean / saxs_mean[0], label="sarw_spheres $\\to$ saxs_one_two_bead_blend")

plt.plot(
    experiment[0],
    experiment[1] / experiment[1][0],
    label="experiment",
)

plt.ylim([0.001, 1.2])
plt.yscale("log")
plt.xlabel("q [\AA]")
plt.ylabel("I(q)")
plt.legend()
plt.show()
