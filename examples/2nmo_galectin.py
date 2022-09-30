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
locations = read_c_alpha_pdb("2nmo.pdb")

sequence = list(
    "MADNFSLHDALSGSGNPNPQGWPGAWGNQPAGAGGYPGASYPGAYPGQAPPGAYPGQAPP\
GAYPGAPGAYPGAPAPGVYPGPPSGPGAYPSSGQPSATGAYPATGPYGAPAGPLIVPYNL\
PLPGGVVPRMLITILGTVKPNANRIALDFQRGNDVAFHFNPRFNENNRRVIVCNTKLDNN\
WGREERQSVFPFESGKPFKIQVLVEPDHFKVAVNDAHLLQYNHRVKKLNEISKLGISGDI\
DLTSASYTMI"
)


experiment = np.transpose(
    np.array(
        [
            [7.11e-3, 1.0, 4.280055302689016e-2],
            [1.019e-2, 1.0232240224939027, 2.026315380672021e-2],
            [1.326e-2, 1.006571077936402, 1.3916548863654017e-2],
            [1.634e-2, 0.9427710375467976, 1.0564833082192844e-2],
            [1.941e-2, 0.9147002625324282, 8.925481179997825e-3],
            [2.249e-2, 0.9055659981669334, 8.047939353455642e-3],
            [2.556e-2, 0.8554052164727448, 7.369083311326177e-3],
            [2.864e-2, 0.8128873906762152, 6.768986997654296e-3],
            [3.171e-2, 0.75544094573812, 6.470103925558852e-3],
            [3.479e-2, 0.717692200146024, 6.339769779255278e-3],
            [3.786e-2, 0.6901185279542665, 5.930592018392805e-3],
            [4.094e-2, 0.6435928106504281, 5.4171780094138845e-3],
            [4.401e-2, 0.600593416494493, 5.118916315846706e-3],
            [4.709e-2, 0.5686390256784677, 5.051341400897892e-3],
            [5.016e-2, 0.5317446755627359, 4.979727525515356e-3],
            [5.324e-2, 0.495269755953583, 4.589812499029096e-3],
            [5.631e-2, 0.4633930374535908, 4.316250601960449e-3],
            [5.938e-2, 0.4291550805461918, 4.084010377021422e-3],
            [6.246e-2, 0.4040358535410809, 3.948083823963463e-3],
            [6.553e-2, 0.3707765678156991, 3.727494446429404e-3],
            [6.861e-2, 0.34770788995386265, 3.5601882776940645e-3],
            [7.168e-2, 0.3241265748062076, 3.4284560297018936e-3],
            [7.476e-2, 0.3048172370403741, 3.3405309679524027e-3],
            [7.783e-2, 0.28268062697093504, 3.2078666521678347e-3],
            [8.11e-2, 0.2613518089882404, 2.529476643934569e-3],
            [8.437e-2, 0.2422288847808864, 2.9846364158886488e-3],
            [8.783e-2, 0.22135056623118388, 2.8614481226601216e-3],
            [9.128e-2, 0.2056296894660805, 2.8421853882839077e-3],
            [9.513e-2, 0.18829322852748823, 2.7267643266586923e-3],
            [9.897e-2, 0.17115871561058207, 2.651732869370699e-3],
            [0.10301, 0.15463781399033757, 2.0865890979137216e-3],
            [0.10723, 0.13856430491044383, 2.026936759200286e-3],
            [0.11146, 0.12048529663057493, 1.984061640750004e-3],
            [0.11607, 0.11325711090053284, 2.0005281717490253e-3],
            [0.12068, 0.10381215727090551, 1.9082534603016795e-3],
            [0.12568, 8.875615553104563e-2, 1.8605626582573439e-3],
            [0.13087, 7.70229754710826e-2, 1.6469637891662655e-3],
            [0.13605, 7.139794634396408e-2, 1.765025709536607e-3],
            [0.14162, 5.734391748093145e-2, 1.4996970779674707e-3],
            [0.14739, 5.038447796436394e-2, 1.5644757895390926e-3],
            [0.15353, 4.741584204557812e-2, 1.4122380501141783e-3],
            [0.15968, 3.4719525266804406e-2, 1.3539838130893387e-3],
            [0.16621, 3.1317477824553776e-2, 1.381479812965063e-3],
            [0.17294, 2.652975626427229e-2, 1.2074938250508755e-3],
            [0.17985, 2.422288847808864e-2, 1.191648672580119e-3],
            [0.18715, 1.4678203594674788e-2, 9.569850713808585e-3],
            [0.19483, 1.6878194273996863e-2, 1.3544498469855375e-3],
            [0.2027, 1.527457163717708e-2, 1.8201730539201219e-3],
            [0.21096, 1.2025383312879624e-2, 1.8995541609059699e-3],
            [0.2196, 1.1677566681683314e-2, 2.0514812110667517e-3],
            [0.22843, 1.1615118139592687e-2, 2.616469637891663e-3],
            [0.23765, 1.3457194786634147e-2, 1.3177574448914918e-2],
            [0.24744, 1.234150963913442e-2, 7.157348577819894e-2],
            [0.25742, 1.6820716760132356e-2, 6.855203268451059e-3],
            [0.26791, 1.9576530532987434e-2, 1.0884376990353098e-2],
            [0.27872, 5.140198530439781e-2, 3.733397542447921e-2],
        ]
    )
)


ensemble_size = 100
n = 112
ensemble = list()
for i in range(ensemble_size):
    sizes = np.hstack([1.9025 * np.ones(n), np.array([20.0])])
    chain = sarw_spheres.generateChain(sizes)
    attachment = locations[0] + (locations[0] - locations[1])
    chain_with_domain = replace_bead(locations, attachment, chain, sizes, -1)
    ensemble.append(chain_with_domain)

ensemble = np.array(ensemble)

# Generate average curves from conformers
(qvals, saxs_mean) = scattering_curve_ensemble(sequence, ensemble, points=50)

# Plotting
plt.plot(qvals, saxs_mean / saxs_mean[0], label="sarw_spheres $\\to$ saxs_single_bead")
plt.errorbar(
    experiment[0],
    experiment[1] / experiment[1][0],
    experiment[2] / experiment[1][0],
    label="experiment",
    capsize=2,
)

plt.ylim([0.005, 1.2])
plt.yscale("log")
plt.xlabel("$q [\AA]$")
plt.ylabel("$I(q)$")
plt.legend()
plt.show()
