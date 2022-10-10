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


# Read domain shape from 1r4g record in protein database
locations = read_c_alpha_pdb("1r4g.pdb")

sequence = list(
"MKYKPDLIREDEFRDEIRNPVYQERDTEPRASN\
ASRLLPSKEKPTMHSLRLVIESSPLSRAEKAAYV\
KSLSKCKTDQEVKAVMELVEEDIESLTN"
)

# SAXS data from:
# A structural model for unfolded proteins from
# residual dipolar couplings and small-angle x-ray scattering
# Pau Bernado, Laurence Blanchard, Peter Timmins, Dominique Marion,
# Rob W. H. Ruigrok, and Martin Blackledge
# PNAS, (2005)
# 10.1073/pnas.0506202102

experiment = np.transpose(
    np.array(
[[2.587296769425676e-2,0.8240929756945915],[3.2294293086744835e-2,0.7499098109054555],[4.0283862964527045e-2,0.6541263066857846],[4.751702385979731e-2,0.5685573728508687],[5.401924751900339e-2,0.49318272322506784],[6.1539933488175685e-2,0.4278754673235181],[6.792190139358109e-2,0.36504324606465416],[7.592080605996623e-2,0.31653014447684524],[8.279600401182435e-2,0.27421746617693327],[9.000145164695947e-2,0.23852471992855495],[9.756717166385137e-2,0.2069806533798463],[0.10549316406250003,0.17966205082034423],[0.11305888407939191,0.15728771357258617],[0.12134514885979733,0.13717684895424603],[0.1297439987595017,0.11905353677374156],[0.13893491808674488,0.1035457364883442],[0.1480053051097973,9.005101732449919e-2],[0.15766908193809623,7.823406349448005e-2],[0.1667394689611487,6.837343022168786e-2],[0.17632671734234237,5.94579430157951e-2],[0.1860540716497748,5.212273217720255e-2],[0.1957233820734798,4.529136410348364e-2],[0.20312697951858114,4.1062935346761906e-2],[0.21338414615212645,3.593219217639452e-2],[0.22690495671452707,3.1117643207063495e-2],[0.2398747624577703,2.707302212000206e-2],[0.24834116342905407,2.5067064148733303e-2],[0.25701772328969597,2.21121252361985e-2],[0.2698974609375001,1.9787335798229888e-2],[0.27970287426097973,1.7842537014058028e-2],[0.28677374956528223,1.687050603798004e-2],[0.29604122677364875,1.56732931192667e-2],[0.30323766759923987,1.4534767442981163e-2],[0.3126497835726352,1.449840693882869e-2],[0.32375272333768434,1.4046307712115852e-2],[0.3295825855152027,1.3285712518326583e-2],[0.3377112311285896,1.2622764131888823e-2],[0.3440385148331926,1.1018515098977616e-2],[0.35516192461993246,1.1282124483709455e-2],[0.3668024495063492,1.1429013175273972e-2],[0.3720947265625001,1.024147140443267e-2],[0.3812216269003379,1.0574267906066123e-2],[0.39171897512172105,1.1030171468864038e-2],[0.39659324852195954,1.1480421838996153e-2],[0.4055775410420187,1.0884886896833068e-2],[0.41368045291385136,1.0640007676718119e-2],[0.42262292810388513,1.025545539148211e-2],[0.4347306535050677,1.0732945120034497e-2],[0.4430028816991708,1.0868540784930653e-2]]
    )
)


ensemble_size = 100
n = 42
ensemble = list()
for i in range(ensemble_size):
    sizes = np.hstack([1.9025 * np.ones(n), np.array([16.0])])
    chain = sarw_spheres.generateChain(sizes)
    attachment = locations[0] + (locations[0] - locations[1])
    chain_with_domain = replace_bead(locations, attachment, chain, sizes, -1)
    ensemble.append(chain_with_domain)

ensemble = np.array(ensemble)

# Generate average curves from conformers
(qvals, saxs_mean) = scattering_curve_ensemble(sequence, ensemble, points=50)

# Plotting

plt.plot(qvals, saxs_mean / saxs_mean[0], label="sarw_spheres $\\to$ saxs_single_bead")
plt.plot(
    experiment[0],
    experiment[1],
    label="experiment"
)



plt.ylim([0.005, 1.2])
plt.yscale("log")
plt.xlabel("q [1/\AA]")
plt.ylabel("$I(q) / I(0)$")
plt.legend()
plt.show()

