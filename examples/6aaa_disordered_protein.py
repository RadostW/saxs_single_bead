from tqdm import tqdm  # progress bars
import numpy as np
import saxs_single_bead.scattering_curve
import sarw_spheres
import matplotlib.pyplot as plt

# Sequence of the 6AAA protein
sequence = "VRTKADSVPGTYRKVVAARAPRKVLGSSTSATNSTSVSSRKAENKYAGGNPVCVRPTPKWQKGIGEFFRLSPKDSEKENQIPEEAGSSGLGKAKRKACPLQPDHTNDEKE"
ensemble_size = 100

# Generate ensemble from sequence
n = len(sequence)
ensemble = list()
for i in tqdm(range(ensemble_size)):
    item = sarw_spheres.generateChain(1.9025 * np.ones(n))
    ensemble.append(item)

ensemble = np.array(ensemble)

# Generate average curves from conformers
(qvals, saxs_mean) = saxs_single_bead.scattering_curve.scattering_curve_ensemble(
    sequence, ensemble, points=50
)

# Plotting
plt.plot(qvals, saxs_mean / saxs_mean[0], label="sarw_spheres $\\to$ saxs_single_bead")

plt.yscale("log")
plt.xlabel("$q [\AA]$")
plt.ylabel("$I(q)$")
plt.legend()
plt.show()