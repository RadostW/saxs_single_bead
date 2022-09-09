# form_factors.py
# raw data and helper functions for working with tabulated form factors

import numpy as np
import json

"""
Raw data of fitted form factors.
Source: D. Tong, S. Yang and L. Lu  *Accurate optimization of amino acid form factors for computing small-angle X-ray scattering intensity of atomistic protein structures*; J. Appl. Cryst. (2016)
Supplemental data, tabulated form factors for single bead approximation located at C_alpha sites.
"""
_raw_data = np.transpose(
    np.array(
        [
            [
                0.0,
                9.031605,
                18.753885,
                20.176972,
                19.219688,
                9.168,
                9.985664,
                21.371629,
                6.211923,
                10.985094,
                6.228658,
                16.526707,
                19.968169,
                8.641675,
                19.033278,
                23.318231,
                14.012906,
                13.084412,
                7.142596,
                15.696784,
                14.149632,
            ],
            [
                0.01,
                9.032873,
                18.758167,
                20.175863,
                19.209075,
                9.169585,
                9.986057,
                21.366358,
                6.210098,
                10.985704,
                6.226842,
                16.52892,
                19.970212,
                8.637247,
                19.030661,
                23.315152,
                14.008752,
                13.08747,
                7.141364,
                15.703052,
                14.143818,
            ],
            [
                0.02,
                9.036791,
                18.770413,
                20.172925,
                19.178863,
                9.173406,
                9.987564,
                21.351259,
                6.20372,
                10.98785,
                6.220754,
                16.534341,
                19.975624,
                8.625259,
                19.022988,
                23.30544,
                13.997234,
                13.096045,
                7.136393,
                15.72059,
                14.126724,
            ],
            [
                0.03,
                9.043628,
                18.788663,
                20.169295,
                19.133694,
                9.176551,
                9.991123,
                21.328385,
                6.1902,
                10.992488,
                6.208426,
                16.53956,
                19.98247,
                8.609469,
                19.010785,
                23.287901,
                13.981071,
                13.108469,
                7.124318,
                15.745601,
                14.099197,
            ],
            [
                0.04,
                9.053617,
                18.809334,
                20.166787,
                19.080542,
                9.174037,
                9.998078,
                21.300848,
                6.165751,
                11.001047,
                6.186607,
                16.539751,
                19.988326,
                8.595518,
                18.994908,
                23.261208,
                13.964275,
                13.122416,
                7.100916,
                15.772021,
                14.062074,
            ],
            [
                0.05,
                9.066604,
                18.827417,
                20.167832,
                19.027618,
                9.15926,
                10.009681,
                21.272284,
                6.126404,
                11.014686,
                6.151336,
                16.530061,
                19.99142,
                8.589561,
                18.976488,
                23.224462,
                13.951202,
                13.135587,
                7.062673,
                15.792488,
                14.015666,
            ],
            [
                0.06,
                9.081629,
                18.83755,
                20.175379,
                18.982981,
                9.125331,
                10.026386,
                21.2462,
                6.06942,
                11.033075,
                6.099216,
                16.507321,
                19.9916,
                8.596348,
                18.956761,
                23.177281,
                13.945345,
                13.146403,
                7.008124,
                15.799933,
                13.959579,
            ],
            [
                0.07,
                9.096634,
                18.835728,
                20.192525,
                18.953043,
                9.066979,
                10.0474,
                21.225457,
                5.994376,
                11.053622,
                6.028911,
                16.471323,
                19.990231,
                8.617691,
                18.936712,
                23.119444,
                13.948425,
                13.154309,
                6.938438,
                15.789061,
                13.893043,
            ],
            [
                0.08,
                9.108509,
                18.820755,
                20.221683,
                18.941213,
                8.982089,
                10.070983,
                21.212014,
                5.903093,
                11.072202,
                5.941821,
                16.424712,
                19.988847,
                8.652353,
                18.916545,
                23.05068,
                13.96026,
                13.159564,
                6.857019,
                15.756924,
                13.815514,
            ],
            [
                0.09,
                9.11357,
                18.794529,
                20.263615,
                18.947024,
                8.872014,
                10.095418,
                21.206769,
                5.798346,
                11.085185,
                5.841437,
                16.371257,
                19.987582,
                8.697436,
                18.895243,
                22.970991,
                13.979364,
                13.162761,
                6.768438,
                15.702477,
                13.727142,
            ],
            [
                0.1,
                9.108373,
                18.760947,
                20.316985,
                18.966103,
                8.740612,
                10.119848,
                21.209284,
                5.682322,
                11.091325,
                5.73182,
                16.313191,
                19.984761,
                8.750126,
                18.870692,
                22.881113,
                14.003767,
                13.164397,
                6.677192,
                15.625672,
                13.628906,
            ],
            [
                0.11,
                9.090638,
                18.723944,
                20.378738,
                18.991204,
                8.59273,
                10.144288,
                21.217502,
                5.555909,
                11.092329,
                5.616192,
                16.248797,
                19.978045,
                8.808436,
                18.840554,
                22.782469,
                14.031546,
                13.164665,
                6.586648,
                15.526801,
                13.522499,
            ],
            [
                0.12,
                9.059841,
                18.685575,
                20.445062,
                19.014121,
                8.432989,
                10.168777,
                21.227744,
                5.419112,
                11.092055,
                5.496327,
                16.171382,
                19.966156,
                8.870744,
                18.803492,
                22.676463,
                14.06088,
                13.163519,
                6.498245,
                15.406436,
                13.410109,
            ],
            [
                0.13,
                9.017147,
                18.644909,
                20.512468,
                19.027868,
                8.265263,
                10.192343,
                21.235179,
                5.272021,
                11.095108,
                5.372836,
                16.070381,
                19.949818,
                8.935084,
                18.760098,
                22.563624,
                14.089763,
                13.160856,
                6.411049,
                15.26577,
                13.293968,
            ],
            [
                0.14,
                8.964777,
                18.598131,
                20.578605,
                19.028336,
                8.092754,
                10.212608,
                21.234772,
                5.115494,
                11.10569,
                5.245884,
                15.934559,
                19.93136,
                8.999217,
                18.713027,
                22.443197,
                14.115742,
                13.156509,
                6.321892,
                15.106835,
                13.17558,
            ],
            [
                0.15,
                8.905241,
                18.539726,
                20.642402,
                19.014933,
                7.918461,
                10.226525,
                21.222462,
                4.951128,
                11.127044,
                5.115745,
                15.756075,
                19.913541,
                9.061529,
                18.666225,
                22.313365,
                14.136103,
                13.149919,
                6.22626,
                14.932343,
                13.054953,
            ],
            [
                0.16,
                8.840775,
                18.464214,
                20.703389,
                18.990164,
                7.745807,
                10.23194,
                21.196137,
                4.780725,
                11.161265,
                4.982932,
                15.533623,
                19.898612,
                9.121961,
                18.62348,
                22.171839,
                14.148718,
                13.139763,
                6.119805,
                14.745376,
                12.930423,
            ],
            [
                0.17,
                8.773065,
                18.367817,
                20.760564,
                18.958463,
                7.579082,
                10.229001,
                21.155995,
                4.605769,
                11.20903,
                4.84811,
                15.273475,
                19.888143,
                9.182212,
                18.58672,
                22.016365,
                14.153202,
                13.124045,
                5.999953,
                14.54928,
                12.799355,
            ],
            [
                0.18,
                8.703146,
                18.249608,
                20.811623,
                18.924719,
                7.423303,
                10.220526,
                21.104129,
                4.42736,
                11.269185,
                4.712162,
                14.988272,
                19.883362,
                9.24497,
                18.554663,
                21.844993,
                14.151583,
                13.100816,
                5.866883,
                14.347721,
                12.659368,
            ],
            [
                0.19,
                8.631502,
                18.111941,
                20.853118,
                18.892815,
                7.283308,
                10.21124,
                21.043636,
                4.246674,
                11.33859,
                4.576403,
                14.694112,
                19.885413,
                9.312394,
                18.522584,
                21.656269,
                14.147774,
                13.069285,
                5.723455,
                14.144479,
                12.509341,
            ],
            [
                0.2,
                8.558451,
                17.960073,
                20.881486,
                18.864519,
                7.162439,
                10.206461,
                20.977759,
                4.065633,
                11.412582,
                4.442691,
                14.406807,
                19.895154,
                9.38445,
                18.483673,
                21.449581,
                14.145878,
                13.030827,
                5.574248,
                13.942857,
                12.349671,
            ],
            [
                0.21,
                8.484819,
                17.801153,
                20.894242,
                18.839063,
                7.061673,
                10.210934,
                20.909473,
                3.88745,
                11.485834,
                4.313319,
                14.138496,
                19.912561,
                9.458028,
                18.431523,
                21.225554,
                14.148261,
                12.989365,
                5.424329,
                13.745102,
                12.181789,
            ],
            [
                0.22,
                8.412586,
                17.643036,
                20.890731,
                18.813527,
                6.97986,
                10.228041,
                20.841475,
                3.716802,
                11.553196,
                4.190806,
                13.896001,
                19.93595,
                9.527511,
                18.36237,
                20.986135,
                14.154653,
                12.95079,
                5.278393,
                13.552368,
                12.007422,
            ],
            [
                0.23,
                8.345044,
                17.493458,
                20.872237,
                18.783741,
                6.914905,
                10.259209,
                20.776157,
                3.55952,
                11.610445,
                4.077836,
                13.681752,
                19.961417,
                9.586643,
                18.275878,
                20.734111,
                14.162871,
                12.921549,
                5.140554,
                13.365325,
                11.828268,
            ],
            [
                0.24,
                8.286158,
                17.359654,
                20.841605,
                18.745174,
                6.865001,
                10.303321,
                20.715065,
                3.421808,
                11.655041,
                3.977385,
                13.496626,
                19.982939,
                9.630701,
                18.174467,
                20.47221,
                14.170535,
                12.906959,
                5.014649,
                13.185093,
                11.646429,
            ],
            [
                0.25,
                8.239396,
                17.247965,
                20.802525,
                18.693549,
                6.829244,
                10.356417,
                20.657864,
                3.309372,
                11.686641,
                3.892916,
                13.342622,
                19.993432,
                9.657807,
                18.062238,
                20.202351,
                14.176544,
                12.909904,
                4.904576,
                13.014102,
                11.465417,
            ],
            [
                0.26,
                8.206731,
                17.16293,
                20.758625,
                18.625383,
                6.807737,
                10.412247,
                20.601548,
                3.227001,
                11.70687,
                3.828429,
                13.2234,
                19.986486,
                9.668988,
                17.944139,
                19.925529,
                14.181473,
                12.930266,
                4.814344,
                12.856477,
                11.290967,
            ],
            [
                0.27,
                8.188461,
                17.105947,
                20.712587,
                18.53868,
                6.801812,
                10.463809,
                20.540629,
                3.178724,
                11.718396,
                3.788319,
                13.142377,
                19.958059,
                9.667418,
                17.825054,
                19.642357,
                14.187007,
                12.965178,
                4.747861,
                12.717663,
                11.130658,
            ],
            [
                0.28,
                8.183869,
                17.074256,
                20.665539,
                18.433519,
                6.814505,
                10.505249,
                20.468465,
                3.16811,
                11.723965,
                3.777139,
                13.099953,
                19.907443,
                9.657571,
                17.708479,
                19.353807,
                14.195094,
                13.009986,
                4.708709,
                12.603263,
                10.991937,
            ],
            [
                0.29,
                8.192058,
                17.061181,
                20.616817,
                18.312023,
                6.850747,
                10.533222,
                20.379145,
                3.198159,
                11.72594,
                3.799323,
                13.091956,
                19.83721,
                9.644531,
                17.595495,
                19.061611,
                14.207319,
                13.059712,
                4.700017,
                12.517394,
                10.879351,
            ],
            [
                0.3,
                8.212318,
                17.057996,
                20.564082,
                18.177463,
                6.916602,
                10.547263,
                20.269114,
                3.270779,
                11.726183,
                3.858742,
                13.110306,
                19.752331,
                9.633143,
                17.485161,
                18.768095,
                14.22454,
                13.110615,
                4.724289,
                12.461229,
                10.792655,
            ],
            [
                0.31,
                8.243895,
                17.056755,
                20.503724,
                18.032974,
                7.017557,
                10.54928,
                20.137911,
                3.386343,
                11.72584,
                3.958113,
                13.145268,
                19.658781,
                9.626794,
                17.37625,
                18.475658,
                14.246625,
                13.161261,
                4.783161,
                12.432428,
                10.727172,
            ],
            [
                0.32,
                8.285626,
                17.052722,
                20.431583,
                17.880719,
                7.156573,
                10.54256,
                19.987814,
                3.543739,
                11.725022,
                4.098439,
                13.187995,
                19.561986,
                9.626278,
                17.268603,
                18.186178,
                14.272364,
                13.212479,
                4.877204,
                12.425751,
                10.676296,
            ],
            [
                0.33,
                8.335937,
                17.045098,
                20.343851,
                17.721904,
                7.332764,
                10.530643,
                19.822657,
                3.740664,
                11.722743,
                4.278644,
                13.232428,
                19.465376,
                9.629594,
                17.162565,
                17.900496,
                14.299776,
                13.266116,
                5.005851,
                12.434448,
                10.634446,
            ],
            [
                0.34,
                8.393207,
                17.035801,
                20.237814,
                17.557224,
                7.541132,
                10.516429,
                19.646439,
                3.973579,
                11.717316,
                4.495354,
                13.276293,
                19.369527,
                9.63309,
                17.05708,
                17.61804,
                14.326851,
                13.32331,
                5.167339,
                12.451614,
                10.598569,
            ],
            [
                0.35,
                8.45617,
                17.027228,
                20.112093,
                17.387081,
                7.773345,
                10.501753,
                19.462302,
                4.237204,
                11.706876,
                4.742771,
                13.321097,
                19.272375,
                9.63345,
                16.9487,
                17.336813,
                14.352276,
                13.383471,
                5.35853,
                12.470974,
                10.567571,
            ],
            [
                0.36,
                8.524049,
                17.020484,
                19.966466,
                17.211377,
                8.019225,
                10.48749,
                19.272171,
                4.524226,
                11.689606,
                5.012893,
                13.370888,
                19.170609,
                9.629425,
                16.832917,
                17.053963,
                14.375568,
                13.44453,
                5.574724,
                12.487316,
                10.540748,
            ],
            [
                0.37,
                8.596422,
                17.015049,
                19.801707,
                17.029362,
                8.268432,
                10.473946,
                19.077046,
                4.826,
                11.663698,
                5.296431,
                13.429804,
                19.061524,
                9.622359,
                16.706811,
                16.766756,
                14.396414,
                13.503873,
                5.809707,
                12.497093,
                10.516773,
            ],
            [
                0.38,
                8.672926,
                17.009873,
                19.619732,
                16.839993,
                8.511914,
                10.46113,
                18.877646,
                5.134104,
                11.627504,
                5.584328,
                13.49923,
                18.944199,
                9.615388,
                16.570749,
                16.473304,
                14.413794,
                13.558998,
                6.056054,
                12.49918,
                10.493835,
            ],
            [
                0.39,
                8.75286,
                17.004966,
                19.423815,
                16.642665,
                8.742964,
                10.448747,
                18.674982,
                5.441768,
                11.580177,
                5.869275,
                13.575982,
                18.819337,
                9.612001,
                16.427979,
                16.172582,
                14.425767,
                13.607593,
                6.305638,
                12.495174,
                10.470388,
            ],
            [
                0.4,
                8.834757,
                17.002378,
                19.218398,
                16.437826,
                8.957937,
                10.436077,
                18.470443,
                5.744296,
                11.522532,
                6.146599,
                13.65261,
                18.688178,
                9.614948,
                16.282818,
                15.863951,
                14.430311,
                13.647553,
                6.55037,
                12.488715,
                10.445622,
            ],
            [
                0.41,
                8.916219,
                17.005802,
                19.008305,
                16.227183,
                9.1565,
                10.422082,
                18.265374,
                6.038543,
                11.457456,
                6.414272,
                13.719731,
                18.551571,
                9.625965,
                16.138928,
                15.546844,
                14.426636,
                13.677462,
                6.783262,
                12.484048,
                10.419318,
            ],
            [
                0.42,
                8.99434,
                17.018868,
                18.797717,
                16.013563,
                9.341111,
                10.405891,
                18.060523,
                6.3222,
                11.389445,
                6.672324,
                13.769167,
                18.410033,
                9.646118,
                15.998539,
                15.220999,
                14.415834,
                13.697299,
                6.999548,
                12.484591,
                10.391434,
            ],
            [
                0.43,
                9.066667,
                17.042944,
                18.589552,
                15.800596,
                9.515619,
                10.387457,
                17.855869,
                6.593423,
                11.32348,
                6.921987,
                13.796387,
                18.264621,
                9.67618,
                15.862491,
                14.887018,
                14.40022,
                13.708667,
                7.197349,
                12.492118,
                10.361877,
            ],
            [
                0.44,
                9.132242,
                17.075664,
                18.385563,
                15.592158,
                9.683553,
                10.367975,
                17.651002,
                6.850827,
                11.263859,
                7.164828,
                13.801407,
                18.117729,
                9.716628,
                15.730635,
                14.546745,
                14.381811,
                13.714088,
                7.37752,
                12.506583,
                10.330718,
            ],
            [
                0.45,
                9.192113,
                17.111059,
                18.186906,
                15.391553,
                9.847041,
                10.349771,
                17.445816,
                7.093587,
                11.213469,
                7.401998,
                13.788236,
                17.972993,
                9.767263,
                15.602371,
                14.203245,
                14.361146,
                13.715798,
                7.542887,
                12.526225,
                10.29869,
            ],
            [
                0.46,
                9.249012,
                17.141346,
                17.994597,
                15.20066,
                10.006879,
                10.33564,
                17.241021,
                7.321621,
                11.173612,
                7.633771,
                13.763415,
                17.834172,
                9.826713,
                15.477267,
                13.860517,
                14.337359,
                13.714967,
                7.6974,
                12.547752,
                10.26765,
            ],
            [
                0.47,
                9.30641,
                17.159499,
                17.80955,
                15.019543,
                10.163316,
                10.32796,
                17.038214,
                7.535875,
                11.144103,
                7.859642,
                13.734161,
                17.703637,
                9.892132,
                15.355556,
                13.523146,
                14.309396,
                13.711944,
                7.845503,
                12.566731,
                10.240593,
            ],
            [
                0.48,
                9.36742,
                17.161381,
                17.632368,
                14.846822,
                10.316617,
                10.328033,
                16.839534,
                7.738545,
                11.123457,
                8.078941,
                13.706533,
                17.581341,
                9.95935,
                15.238137,
                13.195885,
                14.2773,
                13.707166,
                7.99154,
                12.578402,
                10.221008,
            ],
            [
                0.49,
                9.434072,
                17.146594,
                17.463217,
                14.680573,
                10.466856,
                10.335932,
                16.647114,
                7.932805,
                11.10912,
                8.291632,
                13.684143,
                17.464926,
                10.023606,
                15.126013,
                12.883081,
                14.24251,
                13.70185,
                8.138959,
                12.578789,
                10.21169,
            ],
            [
                0.5,
                9.507167,
                17.117942,
                17.301927,
                14.519166,
                10.613276,
                10.350773,
                16.462505,
                8.121897,
                11.097896,
                8.498707,
                13.667817,
                17.351017,
                10.080735,
                15.019529,
                12.588045,
                14.207045,
                13.697879,
                8.289462,
                12.565615,
                10.213577,
            ],
        ]
    )
)

def _raw_form_factor(residue_id, q):
    """
    Returns linearly interpolated form factor for a given `residue_id`.

    Parameters
    ----------
    residue_id: int
        Residue id
    q: float
        Scattering vector, units: Angstrom^(-1)

    Returns
    -------
    float:
        Form factor value.
    """
    if residue_id not in range(0, 20):
        raise IndexError("Wrong residue id value (has to be 0-19)")
    return np.interp(q, _raw_data[0,:], _raw_data[residue_id + 1,:])


_three_letter_dict = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 4,
    "GLY": 5,
    "HIS": 6,
    "ILE": 7,
    "LYS": 8,
    "LEU": 9,
    "MET": 10,
    "ASN": 11,
    "PRO": 12,
    "GLN": 13,
    "ARG": 14,
    "SER": 15,
    "THR": 16,
    "VAL": 17,
    "TRP": 18,
    "TYR": 19,
}
_single_letter_dict = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}


def form_factor(residue_name, q):
    """
    Returns linearly interpolated form factor for a given `residue_name`.

    Parameters
    ----------
    residue_name: string
        Name of the residue. Either in single letter format or three letter format.
    q: float
        Scattering vector, units: Angstrom^(-1)

    Returns
    -------
    float:
        Form factor value.
    """

    if len(residue_name) == 3:
        return _raw_form_factor(_three_letter_dict[residue_name], q)
    elif len(residue_name) == 1:
        return _raw_form_factor(_single_letter_dict[residue_name], q)
    else:
        raise IndexError("Wrong length of residue name (has to be 3 or 1)")
