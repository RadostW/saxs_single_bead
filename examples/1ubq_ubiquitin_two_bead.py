import matplotlib.pyplot as plt
import numpy as np
from saxs_single_bead.scattering_curve import scattering_curve_two_bead
from saxs_single_bead.scattering_curve import scattering_curve
from saxs_single_bead.read_pdb import read_backbone_and_coe_pdb
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


# Data from 1ubq record in protein database
locations_two = read_backbone_and_coe_pdb("1ubq.pdb")

# Data from 1ubq record in protein database
locations_one = read_c_alpha_pdb("1ubq.pdb")

sequence = list(
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

# SAXS data from:
# Accurate optimization of amino acid form factors for computing
# small-angle X-ray scattering intensity of atomistic protein structures
# D. Tong, S. Yang and L. Lu
# Journal of Applied Crystalography (2016)
# 10.1107/S1600576716007962
experiment = np.transpose(
    np.array(
        [
            [8.869179600887039e-4, 7.0050401369931325],
            [1.4926050856253259e-2, 6.815823042420152],
            [3.0820399113082056e-2, 6.601547198842251],
            [4.212860310421289e-2, 6.314254191339679],
            [5.390544652545173e-2, 5.901641270805373],
            [6.26037461500617e-2, 5.551626196904059],
            [7.176280841628535e-2, 5.201474477383095],
            [8.179282445629102e-2, 4.789172974975425],
            [8.920061408060892e-2, 4.4810245732281],
            [0.10143767068715179, 3.9325017347069613],
            [0.11216382540656025, 3.4508834513283198],
            [0.12364713381396172, 2.97442988679637],
            [0.13453890880785022, 2.5340026237338122],
            [0.144005146168483, 2.173469385298053],
            [0.15705072258652958, 1.7240046475459663],
            [0.16915229749492858, 1.3547546355916011],
            [0.17851364579893383, 1.10776923108362],
            [0.1867782864667309, 0.9143093774117921],
            [0.19611253108593538, 0.7266560913203429],
            [0.2029933213104771, 0.6054961415709961],
            [0.21024914099479494, 0.4922675931868642],
            [0.2208363801481342, 0.36162578123894557],
            [0.22791639147049117, 0.28613114808591145],
            [0.23519306977402468, 0.22520893976208844],
            [0.24076530090736112, 0.1865078002226231],
            [0.2458786424179523, 0.15627400341176637],
            [0.25370161789142076, 0.1224266260103032],
            [0.25949038135756786, 0.10273073954836516],
            [0.2683314561022787, 8.282232541398155e-2],
            [0.2791809178971868, 7.180844203391934e-2],
            [0.2843598150681701, 7.088293532841035e-2],
            [0.2998178397886494, 8.036274482019534e-2],
            [0.30627725752596024, 8.880398563079683e-2],
            [0.31874077914113064, 0.1088437993923878],
            [0.32760867211054745, 0.12554020184313225],
            [0.3375052846738109, 0.1445272007444786],
            [0.346211580648205, 0.16085106849462724],
            [0.35571059583903386, 0.17834964774687995],
            [0.36507756319964424, 0.19226121334435414],
            [0.37561329433410395, 0.2067521746515708],
            [0.38922701980888286, 0.22053975601485543],
            [0.3979153299995283, 0.22669513811755873],
            [0.4019666698117659, 0.22844143660187421],
            [0.4166297117516631, 0.23812861947655833],
            [0.43295576698859006, 0.23020118733056352],
            [0.43874339529178663, 0.22919396224785113],
            [0.45278541774779457, 0.22177892432571855],
            [0.460835652843956, 0.21681081484889428],
            [0.4715586858652505, 0.2107403858815581],
            [0.4827967990753409, 0.20261886792034903],
            [0.4938101500212295, 0.19523471017950794],
        ]
    )
)

(qs, Is_two) = scattering_curve_two_bead(sequence, np.array(locations_two), points=100)
(qs, Is_one) = scattering_curve(sequence, np.array(locations_one), points=100)

plt.plot(qs, Is_two / Is_two[0], label="saxs_two_bead")
plt.plot(qs, Is_one / Is_one[0], label="saxs_single_bead")
plt.plot(experiment[0], experiment[1] / experiment[1][0], label="experiment")
plt.yscale("log")
plt.xlabel("q [\AA]")
plt.ylabel("I(q)")
plt.legend()
plt.show()
