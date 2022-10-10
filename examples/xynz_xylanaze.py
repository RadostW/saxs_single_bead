import numpy as np
import sarw_spheres
import matplotlib.pyplot as plt
import tqdm #progress bar

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

# MD simulation data from:
# Large conformational fluctuations of the multi-domain xylanase Z of
# Clostridium thermocellum
# B. Rozycki, M. Cieplak and Mirjam Czjzek
# Journal of Structural Biology (2015)
# 10.1016/j.jsb.2015.05.004

l1_exp = np.array([[6,6.230677248875299e-4],[7,8.813497890375055e-4],[8,1.4815033853714799e-3],[9,3.4538391479713576e-3],[10,7.545027044107103e-3],[11,1.0148510250738957e-2],[12,1.349584580212275e-2],[13,2.0213762290663843e-2],[14,2.669095989541715e-2],[15,3.578887516453741e-2],[16,4.456230765758477e-2],[17,4.957268452259258e-2],[18,6.0677337383532474e-2],[19,7.286041513736169e-2],[20,8.017206506807073e-2],[21,8.036688353931529e-2],[22,8.75936827804823e-2],[23,8.453374472593447e-2],[24,7.635136893366296e-2],[25,6.544153454396762e-2],[26,5.3878763096101116e-2],[27,4.4422070559688887e-2],[28,3.127239877568479e-2],[29,2.0596254547482318e-2],[30,1.27271983792124e-2],[31,6.2827321560431845e-3],[32,2.8339621940113857e-3],[33,8.680667114526613e-4],[34,6.024051597555286e-4]])

l2_exp = np.array([[2,-8.932779710499883e-4],[3,-8.932779710499883e-4],[4,1.8285877516013294e-3],[5,1.0093161854924426e-2],[6,1.5833824106334493e-2],[7,3.045766630712471e-2],[8,4.1923763589118784e-2],[9,6.1326098025374864e-2],[10,9.871923071878058e-2],[11,0.1440089554093691],[12,0.18407693978112583],[13,0.1751093643198907],[14,0.14009936646228816],[15,0.10006616541353264],[16,2.893837034920843e-2],[17,1.5564011793361976e-3]])

l3_exp = np.array([[6,4.616502313050108e-4],[7,7.143044457536829e-4],[8,7.46696011708653e-4],[9,9.604803470113776e-4],[10,7.867805745779138e-4],[11,1.815617688222304e-3],[12,3.0983237000387076e-3],[13,3.1594049386965972e-3],[14,3.240846590240515e-3],[15,4.6660754922587255e-3],[16,5.23616705306601e-3],[17,6.590134509983303e-3],[18,8.585454972808798e-3],[19,1.0762168204982075e-2],[20,1.0295729655230651e-2],[21,1.588651393905663e-2],[22,1.8904739488666097e-2],[23,2.0486116304660823e-2],[24,2.2695221102789043e-2],[25,2.4904325900917262e-2],[26,2.8211504784918615e-2],[27,3.1828432467755186e-2],[28,3.544454146266102e-2],[29,3.578141374859259e-2],[30,3.5736065556255664e-2],[31,3.880030769559481e-2],[32,3.9441660701502984e-2],[33,4.279094862124577e-2],[34,4.330921367652513e-2],[35,4.179328838983304e-2],[36,4.538875221083352e-2],[37,4.2941974297510774e-2],[38,4.3108385967604385e-2],[39,4.236337995064032e-2],[40,3.9441660701502984e-2],[41,3.622301876444521e-2],[42,3.438209809933834e-2],[43,3.3598222203228334e-2],[44,2.8078699364503273e-2],[45,2.3512352339946113e-2],[46,2.2267652432183593e-2],[47,1.9345933183046254e-2],[48,1.4651585487023805e-2],[49,1.286114167886343e-2],[50,8.56318577121476e-3],[51,7.516533296295147e-3],[52,6.447611619781496e-3],[53,2.6804724992197243e-3],[54,2.314447803928671e-3],[55,6.754345666077355e-4],[56,6.624779402257808e-4],[57,6.358448748849943e-4],[58,4.616502313050108e-4],[59,4.616502313050108e-4]])

ensemble_size = 10000

ensemble = list()
sizes = np.hstack([
np.array([25.0]), #pdb code: 1jjf
1.9025 * np.ones(12), 
np.array([20.0]), #pdb code: 1gmm
1.9025 * np.ones(6),
np.array([14.0]), #pdb code: 1ohz 
1.9025 * np.ones(24),
np.array([28.0]), #pdb code: 1xyz 
])

l1_lengths = list()
l2_lengths = list()
l3_lengths = list()

def normalized(vec):
    return vec * (1.0 /  np.sqrt(np.sum(vec**2)))

adjust_ends = False
if adjust_ends == True:
    end_shift = 1.9025
else:
    end_shift = 0.0    

for i in tqdm.tqdm(range(ensemble_size)):
    chain = sarw_spheres.generateChain(sizes)
    
    l1_end_a = chain[1]  + end_shift*normalized(chain[0] - chain[1])
    l1_end_b = chain[12] + end_shift*normalized(chain[13] - chain[12])
    l1_length = np.sqrt(np.sum((l1_end_a - l1_end_b)**2))
    
    l2_end_a = chain[14] + end_shift*normalized(chain[13] - chain[14])
    l2_end_b = chain[19] + end_shift*normalized(chain[20] - chain[19])
    l2_length = np.sqrt(np.sum((l2_end_a - l2_end_b)**2))
    
    l3_end_a = chain[21] + end_shift*normalized(chain[21] - chain[22])
    l3_end_b = chain[44] + end_shift*normalized(chain[45] - chain[44])
    l3_length = np.sqrt(np.sum((l3_end_a - l3_end_b)**2))
    
    l1_lengths.append(l1_length)
    l2_lengths.append(l2_length)
    l3_lengths.append(l3_length)


# Plotting
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

#L1
bins = np.arange(0,60)+0.5
ax1.hist(np.array(l1_lengths),bins=bins, label="L1, sarw_spheres",density=True)
exp_values = l1_exp[:,1]
exp_edges = [0.]+list(l1_exp[:,0]+0.5) 
ax1.stairs(exp_values, exp_edges, label="L1, MD simulation")
#ax1.set_xlabel("distance $\\mathrm{[\AA]}$")
#ax1.set_ylabel("probability density $\\mathrm{[1/\AA]}$")
ax1.legend()

#L2
bins = np.arange(0,60)+0.5
ax2.hist(np.array(l2_lengths),bins=bins, label="L2, sarw_spheres",density=True)
exp_values = l2_exp[:,1]
exp_edges = [0.]+list(l2_exp[:,0]+0.5) 
ax2.stairs(exp_values, exp_edges, label="L2, MD simulation")
#ax2.set_xlabel("distance $\\mathrm{[\AA]}$")
ax2.set_ylabel("probability density $\\mathrm{[1/\AA]}$",labelpad = 8.0)
ax2.legend()


#L2
bins = np.arange(0,60)+0.5
ax3.hist(np.array(l3_lengths),bins=bins, label="L3, sarw_spheres",density=True)
exp_values = l3_exp[:,1]
exp_edges = [0.]+list(l3_exp[:,0]+0.5) 
ax3.stairs(exp_values, exp_edges, label="L3, MD simulation")
ax3.set_xlabel("distance $\\mathrm{[\AA]}$")
#ax3.set_ylabel("probability density $\\mathrm{[1/\AA]}$")
ax3.legend()

plt.show()
