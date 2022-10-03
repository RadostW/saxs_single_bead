from tqdm import tqdm  # progress bars
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

# Sequence of the 6AAA protein
sequence = "VRTKADSVPGTYRKVVAARAPRKVLGSSTSATNSTSV\
SSRKAENKYAGGNPVCVRPTPKWQKGIGEFFRLSPKDSEKENQIPEEAG\
SSGLGKAKRKACPLQPDHTNDEKE"

ensemble_size = 100

# SAXS data from:
# p15PAF Is an Intrinsically Disordered Protein with Nonrandom Structural
# Preferences at Sites of Interaction with Other Proteins
# Alfredo De Biasio, Alain Ibanez de Opakua, Tiago N. Cordeiro,
# Maider Villate, Nekane Merino, Nathalie Sibille, Moreno Lelli,
# Tammo Diercks, Pau Bernado and Francisco J. Blanco
# Biophysical Journal (2014)
# 10.1016/j.bpj.2013.12.046
experiment = np.transpose(
    np.array(
        [
            [2.40448e-2, 1.0, 5.809601677392472e-2],
            [3.7753300000000004e-2, 0.8394524236611284, 1.6746457911211418e-2],
            [5.14618e-2, 0.6258047673871342, 1.2063986280709241e-2],
            [6.51704e-2, 0.46643878654666165, 1.0594566237828154e-2],
            [7.88789e-2, 0.3265832840137226, 9.226115128157667e-3],
            [9.25874e-2, 0.24347931108043566, 8.801728524387684e-3],
            [0.106296, 0.19123527570976664, 8.5984293537042e-3],
            [0.120004, 0.15230280387174855, 8.571028955107673e-3],
            [0.133713, 0.1237333437367968, 8.7873435226404e-3],
            [0.14742200000000003, 9.894913951459543e-2, 8.22259150828362e-3],
            [0.16113, 8.35975287347258e-2, 7.985325306065192e-3],
            [0.17483900000000002, 6.314462944716936e-2, 7.899461039721034e-3],
            [0.18854700000000002, 6.96277662906202e-2, 7.904981792555492e-3],
            [0.202256, 5.618700336258754e-2, 7.763132228297366e-3],
            [0.21596400000000002, 5.167704384478808e-2, 7.68256044730466e-3],
            [0.22967300000000002, 4.1568056497447145e-2, 7.626197470464263e-3],
            [0.243381, 5.0200385647536754e-2, 7.58923888980657e-3],
            [0.25709, 4.345111382080754e-2, 7.510574594908224e-3],
            [0.27079800000000004, 3.8574407313856175e-2, 7.869567130146499e-3],
            [0.284507, 3.744602719122258e-2, 7.536542303775047e-3],
            [0.298215, 2.7365668736568538e-2, 7.393042573130677e-3],
            [0.31192400000000003, 3.2084497151636845e-2, 7.455068243676784e-3],
            [0.32563200000000003, 2.7024496006149114e-2, 7.335295880311474e-3],
            [0.339341, 3.617908455608616e-2, 7.3638293156872885e-3],
            [0.353049, 2.5095361853764382e-2, 7.248448818695024e-3],
            [0.36675800000000003, 2.6099913092343746e-2, 7.200080184141304e-3],
            [0.380466, 1.9804470224374485e-2, 7.168039730162934e-3],
            [0.394175, 3.713043702017138e-2, 7.2221914176406775e-3],
            [0.407883, 1.7439976027763968e-2, 7.455566281821526e-3],
            [0.42159199999999997, 2.6488175329348478e-2, 7.102784282248178e-3],
            [0.43530100000000005, 2.2805913704930663e-2, 7.095154337870738e-3],
            [0.44900900000000005, 2.105803223468886e-2, 6.970621559905274e-3],
            [0.462718, 2.184930693841841e-2, 6.894162743924558e-3],
            [0.476426, 1.5008562105771683e-2, 6.829607039603163e-3],
            [0.490135, 1.1278870648702984e-2, 6.8448976407103025e-3],
        ]
    )
)

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
plt.errorbar(
    experiment[0],
    experiment[1] / experiment[1][0],
    experiment[2] / experiment[1][0],
    label="experiment",
    capsize=2,
)


plt.yscale("log")
plt.xlabel("$q [\AA]$")
plt.ylabel("$I(q)$")
plt.legend()
plt.show()
