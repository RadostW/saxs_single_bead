import matplotlib.pyplot as plt
import numpy as np
from saxs_single_bead.scattering_curve import scattering_curve

plt.rcParams.update({
    "legend.frameon": False,
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 15,
    "figure.subplot.left":   0.125,
    "figure.subplot.right":  0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top":    0.95,
    "figure.subplot.wspace": 0.2,
    "figure.subplot.hspace": 0.2,
})


# Data from 1ubq record in protein database
# These are locaitons of C_alpha atoms (note, units are Angstroms)
locations = [[2.6266e1,2.5413000000000004e1,2.842],[2.685e1,2.9021e1,3.898],[2.6235e1,3.0058000000000003e1,7.497000000000001],[2.6772e1,3.3436e1,9.197],[2.8605e1,3.3965e1,1.2503e1],[2.7691e1,3.7315e1,1.4143000000000002e1],[3.0225e1,3.8643e1,1.6662e1],[2.9607e1,4.118e1,1.9467e1],[3.1422000000000004e1,4.394e1,1.7553e1],[2.8978e1,4.396e1,1.4678000000000003e1],[3.1191e1,4.2012e1,1.2331e1],[2.9542e1,3.902e1,1.0653e1],[3.1720000000000002e1,3.6289e1,9.176],[3.0505e1,3.3884e1,6.512],[3.1677e1,3.0275000000000002e1,6.639],[3.1220000000000002e1,2.7341e1,4.275],[3.0288000000000004e1,2.4245e1,6.193],[2.8468000000000004e1,2.094e1,5.98],[2.5829e1,1.9825e1,8.494],[2.8054e1,1.6835e1,9.21],[3.0796e1,1.9083e1,1.0566e1],[3.1398000000000003e1,1.9064e1,1.4286e1],[3.1288000000000004e1,2.2201e1,1.6417e1],[3.5031e1,2.1722000000000005e1,1.7069e1],[3.559e1,2.1945e1,1.3302000000000001e1],[3.3533e1,2.5097000000000005e1,1.2978e1],[3.5596e1,2.6715e1,1.5736000000000002e1],[3.8794e1,2.5761e1,1.388e1],[3.7471e1,2.7391e1,1.0668e1],[3.6731e1,3.057e1,1.2645e1],[4.0269e1,3.0508e1,1.4115e1],[4.1718e1,3.0022e1,1.0643000000000002e1],[3.9808e1,3.2994e1,9.233],[3.9676e1,3.5547e1,1.2072e1],[4.2345e1,3.4269e1,1.4431e1],[4.0226e1,3.3716e1,1.7509e1],[4.1461e1,3.0751e1,1.9594e1],[3.8817e1,2.802e1,1.9889e1],[3.9063e1,2.8063e1,2.3695e1],[3.7738e1,3.1637e1,2.3712e1],[3.4738e1,3.0875e1,2.1473e1],[3.12e1,3.0329e1,2.278e1],[2.8762e1,2.9573000000000004e1,1.9906e1],[2.5034000000000002e1,3.017e1,2.0401e1],[2.2126e1,2.9062000000000005e1,1.8183e1],[1.8443e1,2.9143e1,1.9083e1],[1.9399e1,2.9894e1,2.2655e1],[2.155e1,2.6796e1,2.3133e1],[2.5349e1,2.6872e1,2.3643e1],[2.6826e1,2.4521e1,2.1011999999999997e1],[2.9015e1,2.1657e1,2.2288000000000004e1],[3.2262e1,2.067e1,2.0514000000000003e1],[3.1568e1,1.6962e1,1.9825e1],[2.8108000000000004e1,1.7439e1,1.8276e1],[2.7574e1,1.8192e1,1.4563e1],[2.5594e1,2.1109e1,1.3072e1],[2.2924000000000003e1,1.8583e1,1.2025e1],[2.2418e1,1.7638e1,1.5693e1],[2.1079e1,2.1149e1,1.6251e1],[1.9065e1,2.1352e1,1.2999e1],[2.1184e1,2.4263e1,1.169e1],[2.0081e1,2.4773000000000003e1,8.033],[2.1656e1,2.6847000000000005e1,5.24],[2.1907e1,3.0563e1,5.881],[2.1419e1,3.0253000000000004e1,9.62],[2.3212e1,3.2762e1,1.1891e1],[2.5149e1,3.1609e1,1.498e1],[2.6179e1,3.4127e1,1.765e1],[2.9801e1,3.4145e1,1.8829e1],[3.0479000000000003e1,3.5369e1,2.2374000000000002e1],[3.4145e1,3.5472e1,2.3481000000000005e1],[3.5161e1,3.4174e1,2.6896e1],[3.8668e1,3.5502e1,2.768e1],[4.0873e1,3.3802e1,3.0253000000000004e1],[4.1845e1,3.655e1,3.2686e1],[4.0373e1,3.9813e1,3.3944e1]]
sequence = list(
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

experiment = np.transpose(np.array([[2.017036112070518e-2,6.73870123170992],[3.237814796118682e-2,6.498620630410662],[5.272971002299623e-2,5.902095375139478],[6.943993867668777e-2,5.267649125205302],[8.733069094670978e-2,4.521474828870395],[0.1017962620200547,3.9041441747810555],[0.11999461551405013,3.105285005568141],[0.13696256411215185,2.423964320104153],[0.15250889624270067,1.8583303660891963],[0.1719833791388561,1.269356906417914],[0.18628683605362048,0.9167587408258417],[0.20147568568063265,0.6233149115254721],[0.21303907321575163,0.4495169286679442],[0.223494707659271,0.3264043014679449],[0.23430380279320204,0.22830012809305456],[0.2460481719189442,0.15378513609463682],[0.2585541212951611,0.10467959480966023],[0.27313206228400577,7.638000636886096e-2],[0.2933403755429669,7.498987793874658e-2],[0.30731992389236207,8.999417059782844e-2],[0.3239030512087049,0.11829701294629157],[0.34485318862527337,0.15777422358206802],[0.3639072705857158,0.1895412193688944],[0.37892833772692425,0.20896426051116115],[0.3945162375904425,0.22311015153684072],[0.4125566336991542,0.23080614887980494],[0.430730046553368,0.2291608174234481],[0.44798513226075054,0.22289498826916265],[0.4625361642676129,0.21470744931968222],[0.47504332828537765,0.20650343820026537],[0.49118615738403715,0.1964804894655716]]))

(qs, Is) = scattering_curve(sequence, np.array(locations), points=50)

plt.plot(qs, Is / Is[0], label='saxs_single_bead')
plt.plot(experiment[0], experiment[1] / experiment[1][0], label='experiment')
plt.yscale('log')
plt.xlabel('$q [\AA]$')
plt.ylabel('$I(q)$')
plt.legend()
plt.show()
