import pytest
import numpy as np
import neutral_density as nd


def test_dummy():
    """Are internal tests working?"""
    assert 1 == 1


def imports():
    """Are the data files importing?"""
    nd.init_fdt()


def test_gamma_n():
    """Are neutral densities and surface consistent with original product?"""
    # adapted from eos80_legacy_gamma_n example data
    # https://www.teos-10.org/preteos10_software/neutral_density.html
    long = 187.317
    lat = -41.6667

    SP = [
        35.066,
        35.086,
        35.089,
        35.078,
        35.025,
        34.851,
        34.696,
        34.572,
        34.531,
        34.509,
        34.496,
        34.452,
        34.458,
        34.456,
        34.488,
        34.536,
        34.579,
        34.612,
        34.642,
        34.657,
        34.685,
        34.707,
        34.72,
        34.729,
    ]

    t = [
        12.25,
        12.21,
        12.09,
        11.99,
        11.69,
        10.54,
        9.35,
        8.36,
        7.86,
        7.43,
        6.87,
        6.04,
        5.5,
        4.9,
        4.04,
        3.29,
        2.78,
        2.45,
        2.211,
        2.011,
        1.894,
        1.788,
        1.554,
        1.38,
    ]

    p = [
        1.0,
        48.0,
        97.0,
        145.0,
        194.0,
        291.0,
        388.0,
        485.0,
        581.0,
        678.0,
        775.0,
        872.0,
        969.0,
        1066.0,
        1260.0,
        1454.0,
        1647.0,
        1841.0,
        2020.0,
        2216.0,
        2413.0,
        2611.0,
        2878.0,
        3000.0,
    ]

    gamma_n_known = [
        26.657202583296442,
        26.682830469203406,
        26.710963096615604,
        26.723242299110460,
        26.741488538021695,
        26.825445912051336,
        26.918689217252997,
        26.989761790054338,
        27.039067923101946,
        27.089143151019517,
        27.166567035269665,
        27.260376554533835,
        27.343619695291586,
        27.421578895148251,
        27.557338511940429,
        27.698188932980081,
        27.798443363873236,
        27.866285802482334,
        27.920185440895871,
        27.959264296723756,
        27.997866000490600,
        28.031679411184577,
        28.079958980601589,
        28.117372360538731,
    ]

    gamman, dgl, dgh = nd.gamma_n(SP, t, p, long, lat)

    # backed off precision to 1e-4 because data was originally stored 32-bit
    assert np.allclose(gamma_n_known, gamman, rtol=0, atol=1e-4)

    SP_ns_known = [34.906414899221204, 34.630793170239990, 34.724828416099236]

    t_ns_known = [10.906418170693556, 2.300294994057337, 1.460659249137126]

    p_ns_known = [260.107786066340, 1953.132582431939, 2943.451862678507]

    gamma_n_surfaces = [26.8, 27.9, 28.1]
    SP_ns, t_ns, p_ns, _, _, _ = nd.neutral_surfaces(
        SP, t, p, gamma_n_known, gamma_n_surfaces
    )

    assert (  # similar to above, backed off tolerance
        np.allclose(SP_ns, SP_ns_known, rtol=0, atol=1e-5)
        | np.allclose(t_ns, t_ns_known, rtol=0, atol=1e-4)
        | np.allclose(p_ns, p_ns_known, rtol=0, atol=1e-2)
    )
