import numpy as np
import sarw_spheres
import matplotlib.pyplot as plt
from tqdm import tqdm

from saxs_single_bead.read_pdb import read_c_alpha_pdb
from saxs_single_bead.read_pdb import read_backbone_and_coe_pdb

from saxs_single_bead.replace_bead import replace_bead

from saxs_single_bead.scattering_curve import scattering_curve_ensemble
from saxs_single_bead.scattering_curve import scattering_curve_one_two_blend_ensemble

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

# Read domain shape from 2nmo record in protein database (two bead model variant)
locations_two_raw = read_backbone_and_coe_pdb("1yzb.pdb")
locations_two = np.vstack((locations_two_raw[:, 0], locations_two_raw[:, 1]))

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
            [8.091e-3, 0.9999999999999999, 1.4112426124464207e-2],
            [1.3589e-2, 1.0160567941437804, 6.443202097606665e-3],
            [1.9088e-2, 0.886896573498171, 4.866614346775636e-3],
            [2.4586e-2, 0.8157276148293814, 4.438621100251225e-3],
            [3.0084e-2, 0.6695025400705716, 3.6841171544594433e-3],
            [3.5582e-2, 0.6143212170689134, 3.368731949261733e-3],
            [4.108e-2, 0.4862382148693731, 3.116529026263683e-3],
            [4.6578e-2, 0.4254497932011203, 2.9707700194996833e-3],
            [5.2076e-2, 0.3708318110951244, 2.6694044402748897e-3],
            [5.7575e-2, 0.32128641532070734, 2.4330628635518287e-3],
            [6.3073e-2, 0.281928514190267, 2.2616701170732733e-3],
            [6.8571e-2, 0.24726647362431006, 2.0974187350313238e-3],
            [7.4069e-2, 0.21167252258550467, 1.942902160300328e-3],
            [7.9567e-2, 0.18591414726846567, 1.8578071651188695e-3],
            [8.5065e-2, 0.16664430292014148, 1.8758860929820682e-3],
            [9.0563e-2, 0.1428565307401911, 1.764217704870711e-3],
            [9.6061e-2, 0.12902731609483128, 1.6283814360607328e-3],
            [0.101559, 0.11284381911149398, 1.5201333856532243e-3],
            [0.107058, 0.10384600061039871, 1.4617621279161195e-3],
            [0.112556, 8.814830433939372e-2, 1.3786591975511883e-3],
            [0.118054, 7.908771700351354e-2, 1.3194986311131959e-3],
            [0.123552, 6.801232524329501e-2, 1.2712004558445675e-3],
            [0.12905, 6.348428674307174e-2, 1.2197826319010011e-3],
            [0.134548, 5.282392101500588e-2, 1.1658089512117014e-3],
            [0.140046, 4.8543086482675045e-2, 1.2129419564933042e-3],
            [0.145544, 4.49765011523907e-2, 1.1531048397402646e-3],
            [0.151042, 3.765423467876639e-2, 1.09180186397129e-3],
            [0.156541, 3.2813141313319465e-2, 1.0617329610803155e-3],
            [0.162039, 3.194809656327474e-2, 1.0399330064843587e-3],
            [0.167537, 2.7798212103034098e-2, 9.99866193382135e-4],
            [0.173035, 2.433168104711947e-2, 9.74909003982626e-4],
            [0.178533, 2.2725325082426376e-2, 9.571683512769511e-4],
            [0.184031, 2.021291790137099e-2, 9.504404342550954e-4],
            [0.189529, 1.7901032716469787e-2, 9.292042715883446e-4],
            [0.195027, 1.8147635306304394e-2, 9.146208536862219e-4],
            [0.200526, 1.7588804746075628e-2, 9.674293643884962e-4],
            [0.206024, 1.732074047680259e-2, 9.158236098018609e-4],
            [0.211522, 1.563831018779533e-2, 8.923698655469006e-4],
            [0.21702, 1.4116071978939737e-2, 8.731257676966768e-4],
            [0.222518, 1.342264549214525e-2, 8.665106090606624e-4],
            [0.228016, 1.3484437087586202e-2, 8.595571752671245e-4],
            [0.233514, 1.2081422078693324e-2, 8.442596209213411e-4],
            [0.239012, 1.2096193427238517e-2, 8.401627329024459e-4],
            [0.24451, 1.033385502879849e-2, 8.196407066793556e-4],
            [0.250009, 1.1580474156529687e-2, 8.111462416126552e-4],
            [0.255507, 9.486400586944983e-3, 8.326455071797021e-4],
            [0.261005, 1.0830142271014026e-2, 8.419668670759043e-4],
            [0.266503, 1.0626463040056287e-2, 8.0866555712415e-4],
            [0.272001, 1.0604324810302808e-2, 7.950969646945976e-4],
            [0.277499, 1.0011854664964764e-2, 7.906618015181789e-4],
            [0.282997, 1.0686300156809327e-2, 7.837835399818684e-4],
            [0.288495, 7.481330969917565e-3, 7.680349520927204e-4],
            [0.293993, 8.135329607796263e-3, 7.629984108584822e-4],
            [0.299492, 8.837701593200818e-3, 7.599163483121572e-4],
            [0.30499, 9.896502836249264e-3, 7.573980776950381e-4],
            [0.310488, 7.34128505470285e-3, 7.411984562625255e-4],
            [0.315986, 6.56516908495818e-3, 7.878804280007637e-4],
            [0.321484, 8.562082512076422e-3, 7.694256388514279e-4],
            [0.326982, 8.40083802032357e-3, 7.47926373284381e-4],
            [0.33248, 6.413847331159351e-3, 7.361243288996736e-4],
            [0.337978, 9.20717323747367e-3, 7.371015682436302e-4],
            [0.343477, 8.233241472835e-3, 7.206012577822079e-4],
            [0.348975, 6.105565904269633e-3, 7.122947233585761e-4],
            [0.354473, 6.561560816611264e-3, 7.14437132689558e-4],
            [0.359971, 6.0437367227000665e-3, 7.152264413904461e-4],
            [0.365469, 7.684145719917189e-3, 7.118436898152114e-4],
            [0.370967, 7.849148824531412e-3, 7.257129712736736e-4],
            [0.376465, 6.9770754684359195e-3, 7.582625586531536e-4],
            [0.381963, 5.937067289694334e-3, 7.302984789645472e-4],
            [0.387461, 7.886246333473153e-3, 7.212778080972548e-4],
            [0.39296, 7.19188019346332e-3, 7.27780208347428e-4],
            [0.398458, 7.212364633557797e-3, 7.298474454211825e-4],
            [0.403956, 6.4965744002381445e-3, 7.271036580323811e-4],
            [0.409454, 5.525536767502732e-3, 7.251491793444677e-4],
            [0.414952, 6.887658068463884e-3, 7.263519354601067e-4],
            [0.42045, 6.072753213989856e-3, 7.290205505916808e-4],
            [0.425948, 6.344951957410405e-3, 7.28456758662475e-4],
            [0.431446, 5.617171749062977e-3, 7.788597571334713e-4],
            [0.436945, 7.1790633236060416e-3, 7.690873636939044e-4],
            [0.442443, 6.425085583614853e-3, 7.533011896761428e-4],
            [0.447941, 5.935150397135033e-3, 7.449946552525111e-4],
            [0.453439, 6.784145870261704e-3, 7.469115478118106e-4],
            [0.458937, 7.665803689153695e-3, 7.467987894259695e-4],
            [0.464435, 6.422266623968824e-3, 7.491667155286337e-4],
            [0.469933, 5.166889928270631e-3, 7.457463778247855e-4],
            [0.475431, 5.753684568188003e-3, 7.452953442814208e-4],
            [0.480929, 6.201974324163821e-3, 7.415367314200489e-4],
            [0.486428, 7.1680130017936095e-3, 7.458967223392402e-4],
            [0.491926, 5.871742598163691e-3, 8.232865611548863e-4],
            [0.497424, 6.51269884941343e-3, 7.744245939570525e-4],
        ]
    )
)


ensemble_size = 20
n = len(sequence) - len(locations)
domain_ball_size = 20.0

ensemble = list()
sizes = np.hstack([1.9025 * np.ones(n), np.array([domain_ball_size])])

ensemble_two = list()
model_types = [2] * len(locations_two) + [1] * n
sequence_two = ["BB"] * (len(locations_two) // 2) + sequence

ensemble_denatured = list()
sizes_denatured = np.array([1.9025] * len(sequence))

deleted_residues = 90+45
ensemble_chunk = list()
sizes_chunk = np.hstack([1.9025 * np.ones(n), np.array([domain_ball_size])])

for i in tqdm(range(ensemble_size)):
    chain = sarw_spheres.generateChain(sizes)

    attachment = locations[-1] + (locations[-1] - locations[-2])
    chain_with_domain = replace_bead(locations, attachment, chain, sizes, 0)
    ensemble.append(chain_with_domain)
    
    chain_chunk = replace_bead(locations[deleted_residues:], attachment, chain, sizes_chunk, 0)
    ensemble_chunk.append(chain_chunk)

    #chain_with_domain_two = replace_bead(locations_two, attachment, chain, sizes, 0)
    #ensemble_two.append(chain_with_domain_two)
    
    chain_denatured = sarw_spheres.generateChain(sizes_denatured)
    ensemble_denatured.append(chain_denatured)


ensemble = np.array(ensemble)
ensemble_two = np.array(ensemble_two)
ensemble_denatured = np.array(ensemble_denatured)
ensemble_chunk = np.array(ensemble_chunk)

# Generate average curves from conformers
(qvals, saxs_mean) = scattering_curve_ensemble(sequence, ensemble, points=50)

(qvals_denatured, saxs_mean_denatured) = scattering_curve_ensemble(sequence, ensemble_denatured, points=50)

(qvals_chunk, saxs_mean_chunk) = scattering_curve_ensemble(sequence[deleted_residues:], ensemble_chunk, points=50)

#(qvals_two, saxs_mean_two) = scattering_curve_one_two_blend_ensemble(
#    sequence_two, model_types, ensemble_two, points=50, maximal_q = 0.26
#)

# Plotting
plt.plot(qvals, qvals**2 * saxs_mean / saxs_mean[0], label="sarw_spheres+domain $\\to$ saxs_single_bead")

plt.plot(qvals_denatured, qvals_denatured**2 * saxs_mean_denatured / saxs_mean_denatured[0], label="sarw_spheres $\\to$ saxs_single_bead")

plt.plot(qvals_chunk, qvals_chunk**2 * saxs_mean_chunk / saxs_mean_chunk[0], label="sarw_spheres+domain_chunk $\\to$ saxs_single_bead")

# Plotting
# plt.plot(qvals, saxs_mean_two / saxs_mean_two[0], label="sarw_spheres $\\to$ saxs_one_two_bead_blend")

plt.errorbar(
    experiment[0],
    experiment[0]**2 * experiment[1] / experiment[1][0],
    experiment[0]**2 * experiment[2] / experiment[1][0],
    label="experiment",
    capsize = 2,
)

#plt.ylim([0.001, 1.2])
#plt.xlim([0.0, 0.3])
#plt.yscale("log")
plt.xlabel("q [\AA]")
plt.ylabel("I(q)")
plt.legend()
plt.show()

