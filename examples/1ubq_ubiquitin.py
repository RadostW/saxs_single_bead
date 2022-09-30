import matplotlib.pyplot as plt
import numpy as np
from saxs_single_bead.scattering_curve import scattering_curve
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
locations = read_c_alpha_pdb("1ubq.pdb")

sequence = list(
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

experiment = np.transpose(
    np.array(
        [
            [2.017036112070518e-2, 6.73870123170992],
            [3.237814796118682e-2, 6.498620630410662],
            [5.272971002299623e-2, 5.902095375139478],
            [6.943993867668777e-2, 5.267649125205302],
            [8.733069094670978e-2, 4.521474828870395],
            [0.1017962620200547, 3.9041441747810555],
            [0.11999461551405013, 3.105285005568141],
            [0.13696256411215185, 2.423964320104153],
            [0.15250889624270067, 1.8583303660891963],
            [0.1719833791388561, 1.269356906417914],
            [0.18628683605362048, 0.9167587408258417],
            [0.20147568568063265, 0.6233149115254721],
            [0.21303907321575163, 0.4495169286679442],
            [0.223494707659271, 0.3264043014679449],
            [0.23430380279320204, 0.22830012809305456],
            [0.2460481719189442, 0.15378513609463682],
            [0.2585541212951611, 0.10467959480966023],
            [0.27313206228400577, 7.638000636886096e-2],
            [0.2933403755429669, 7.498987793874658e-2],
            [0.30731992389236207, 8.999417059782844e-2],
            [0.3239030512087049, 0.11829701294629157],
            [0.34485318862527337, 0.15777422358206802],
            [0.3639072705857158, 0.1895412193688944],
            [0.37892833772692425, 0.20896426051116115],
            [0.3945162375904425, 0.22311015153684072],
            [0.4125566336991542, 0.23080614887980494],
            [0.430730046553368, 0.2291608174234481],
            [0.44798513226075054, 0.22289498826916265],
            [0.4625361642676129, 0.21470744931968222],
            [0.47504332828537765, 0.20650343820026537],
            [0.49118615738403715, 0.1964804894655716],
        ]
    )
)

(qs, Is) = scattering_curve(sequence, np.array(locations), points=50)

plt.plot(qs, Is / Is[0], label="saxs_single_bead")
plt.plot(experiment[0], experiment[1] / experiment[1][0], label="experiment")
plt.yscale("log")
plt.xlabel("$q [\AA]$")
plt.ylabel("$I(q)$")
plt.legend()
plt.show()
