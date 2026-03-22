import numpy as np
import math
from numba import njit
from scipy.io import FortranFile
import os

along_d = np.zeros(90, dtype=np.float32)
alat_d = np.zeros(45, dtype=np.float32)
p0_s_global = np.zeros(33, dtype=np.float32)
n_global = np.zeros((90, 45), dtype=np.int32)
iocean_global = np.zeros((90, 45), dtype=np.int32)
stga_data = np.zeros((4050, 4, 33), dtype=np.float32)


@njit
def atg(s, t, p):
    ds = s - 35.0
    atg = (
        (
            ((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p
            + (
                (2.7759e-12 * t - 1.1351e-10) * ds
                + ((-5.4481e-14 * t + 8.733e-12) * t - 6.7795e-10) * t
                + 1.8741e-8
            )
        )
        * p
        + (-4.2393e-8 * t + 1.8932e-6) * ds
        + ((6.6228e-10 * t - 6.836e-8) * t + 8.5258e-6) * t
        + 3.5803e-5
    )
    return atg


@njit
def depth_ns(s, t, p, n, s0, t0, p0):
    nmax = 100
    e = np.zeros(nmax)
    n2 = 2
    if n > nmax:
        print("\nparameter nmax in depth-ns.f <", n, "\n")
        raise ValueError("n > nmax")

    ncr = 0
    sns, tns, pns = -99.0, -99.0, -99.0

    for k in range(n):
        sigl, sigu = sig_vals(s0, t0, p0, s[k], t[k], p[k])
        e[k] = sigu - sigl

        if k > 0:
            if e[k - 1] == 0.0:
                ncr += 1
                sns = s[k - 1]
                tns = t[k - 1]
                pns = p[k - 1]
            elif e[k] * e[k - 1] < 0.0:
                ncr += 1
                pc0 = p[k - 1] - e[k - 1] * (p[k] - p[k - 1]) / (
                    e[k] - e[k - 1]
                )
                iter = 0
                isuccess = 0
                pc_0 = 0.0
                ec_0 = 0.0
                while isuccess == 0:
                    iter += 1
                    sc0, tc0 = stp_interp(
                        s[k - 1 : k + 1],
                        t[k - 1 : k + 1],
                        p[k - 1 : k + 1],
                        n2,
                        pc0,
                    )
                    sigl, sigu = sig_vals(s0, t0, p0, sc0, tc0, pc0)
                    ec0 = sigu - sigl
                    p1 = (p[k - 1] + pc0) / 2
                    ez1 = (e[k - 1] - ec0) / (pc0 - p[k - 1])
                    p2 = (pc0 + p[k]) / 2
                    ez2 = (ec0 - e[k]) / (p[k] - pc0)
                    r = (pc0 - p1) / (p2 - p1)
                    ecz_0 = ez1 + r * (ez2 - ez1)
                    if iter == 1:
                        ecz0 = ecz_0
                    else:
                        ecz0 = -(ec0 - ec_0) / (pc0 - pc_0)
                        if ecz0 == 0:
                            ecz0 = ecz_0
                    pc1 = pc0 + ec0 / ecz0
                    if pc1 <= p[k - 1] or pc1 >= p[k]:
                        sns, tns, pns, niter = e_solve(
                            s, t, p, e, n, k, s0, t0, p0
                        )
                        if pns < p[k - 1] or pns > p[k]:
                            print("ERROR 1 in depth-ns.f")
                            raise ValueError("ERROR 1 in depth-ns.f")
                        else:
                            isuccess = 1
                    else:
                        eps = abs(pc1 - pc0)
                        if abs(ec0) <= 5e-5 and eps <= 5e-3:
                            sns = sc0
                            tns = tc0
                            pns = pc0
                            isuccess = 1
                            niter = iter
                        elif iter > 10:
                            sns, tns, pns, niter = e_solve(
                                s, t, p, e, n, k, s0, t0, p0
                            )
                            isuccess = 1
                        else:
                            pc_0 = pc0
                            ec_0 = ec0
                            pc0 = pc1
                            isuccess = 0

            if k == n - 1 and e[k] == 0.0:
                ncr += 1
                sns = s[k]
                tns = t[k]
                pns = p[k]

    if ncr == 0:
        sns, tns, pns = -99.0, -99.0, -99.0
    elif ncr >= 2:
        sns, tns, pns = -99.2, -99.2, -99.2

    return sns, tns, pns


@njit
def derthe(s, t, p0):
    a0, a1, a2 = -0.36504e-4, -0.83198e-5, +0.54065e-7
    a3, b0, b1 = -0.40274e-9, -0.17439e-5, +0.29778e-7
    d0, c0, c1 = +0.41057e-10, -0.89309e-8, +0.31628e-9
    c2, e0, e1 = -0.21987e-11, +0.16056e-12, -0.50484e-14
    ds = s - 35.0
    p = p0
    pp = p * p
    ppp = pp * p
    tt = t * t
    ttt = tt * t
    part = 1.0 + p * (a1 + 2.0 * a2 * t + 3.0 * a3 * tt + ds * b1)
    dthedt = part + pp * (c1 + 2.0 * c2 * t) + ppp * e1
    dtheds = p * (b0 + b1 * t) + pp * d0
    part = a0 + a1 * t + a2 * tt + a3 * ttt + ds * (b0 + b1 * t)
    dthedp = (
        part
        + 2.0 * p * (ds * d0 + c0 + c1 * t + c2 * tt)
        + 3.0 * pp * (e0 + e1 * t)
    )
    return dthedt, dtheds, dthedp


@njit
def eos8d(s, t, p0):
    drv = np.zeros((3, 8))
    r3500, r4 = 1028.1063, 4.8314e-4
    dr350 = 28.106331
    p = p0 / 10.0
    sal = s
    sr = math.sqrt(abs(s))
    r1 = (
        (((6.536332e-9 * t - 1.120083e-6) * t + 1.001685e-4) * t - 9.095290e-3)
        * t
        + 6.793952e-2
    ) * t - 28.263737
    r2 = (
        ((5.3875e-9 * t - 8.2467e-7) * t + 7.6438e-5) * t - 4.0899e-3
    ) * t + 8.24493e-1
    r3 = (-1.6546e-6 * t + 1.0227e-4) * t - 5.72466e-3
    sig = (r4 * s + r3 * sr + r2) * s + r1
    v350p = 1.0 / r3500
    sva = -sig * v350p / (r3500 + sig)
    sigma = sig + dr350
    drv[0, 2] = sigma
    v0 = 1.0 / (1000.0 + sigma)
    drv[0, 1] = v0
    r4s = 9.6628e-4
    rhos = r4s * sal + 1.5 * r3 * sr + r2
    r1 = (
        ((3.268166e-8 * t - 4.480332e-6) * t + 3.005055e-4) * t - 1.819058e-2
    ) * t + 6.793952e-2
    r2 = ((2.155e-8 * t - 2.47401e-6) * t + 1.52876e-4) * t - 4.0899e-3
    r3 = -3.3092e-6 * t + 1.0227e-4
    rhot = (r3 * sr + r2) * sal + r1
    drv[1, 2] = rhot
    rho1 = 1000.0 + sigma
    rho2 = rho1 * rho1
    v0t = -rhot / rho2
    v0s = -rhos / rho2
    drv[0, 7] = rhos
    drv[1, 1] = v0t
    r1 = ((1.3072664e-7 * t - 1.3440996e-5) * t + 6.01011e-4) * t - 1.819058e-2
    r2 = (6.465e-8 * t - 4.94802e-6) * t + 1.52876e-4
    r3 = -3.3092e-6
    rhott = (r3 * sr + r2) * sal + r1
    drv[2, 2] = rhott
    v0tt = (2.0 * rhot * rhot / rho1 - rhott) / rho2
    drv[2, 1] = v0tt
    svan_val = sva * 1.0e8
    eos8d = svan_val
    e = (9.1697e-10 * t + 2.0816e-8) * t - 9.9348e-7
    bw = (5.2787e-8 * t - 6.12293e-6) * t + 3.47718e-5
    b = bw + e * s
    dbds = e
    drv[0, 5] = b + 5.03217e-5
    bw = 1.05574e-7 * t - 6.12293e-6
    e = 1.83394e-9 * t + 2.0816e-8
    bt = bw + e * sal
    drv[1, 5] = bt
    e = 1.83394e-9
    bw = 1.05574e-7
    btt = bw + e * sal
    drv[2, 5] = btt
    d = 1.91075e-4
    c = (-1.6078e-6 * t - 1.0981e-5) * t + 2.2838e-3
    aw = ((-5.77905e-7 * t + 1.16092e-4) * t + 1.43713e-3) * t - 0.1194975
    a = (d * sr + c) * s + aw
    drv[0, 4] = a + 3.3594055
    dads = 2.866125e-4 * sr + c
    c = -3.2156e-6 * t - 1.0981e-5
    aw = (-1.733715e-6 * t + 2.32184e-4) * t + 1.43713e-3
    at = c * sal + aw
    drv[1, 4] = at
    c = -3.2156e-6
    aw = -3.46743e-6 * t + 2.32184e-4
    att = c * sal + aw
    drv[2, 4] = att
    b1 = (-5.3009e-4 * t + 1.6483e-2) * t + 7.944e-2
    a1 = ((-6.1670e-5 * t + 1.09987e-2) * t - 0.603459) * t + 54.6746
    kw = (
        ((-5.155288e-5 * t + 1.360477e-2) * t - 2.327105) * t + 148.4206
    ) * t - 1930.06
    k0 = (b1 * sr + a1) * s + kw
    drv[0, 3] = k0 + 21582.27
    k0s = 1.5 * b1 * sr + a1
    ks = (dbds * p + dads) * p + k0s
    b1 = -1.06018e-3 * t + 1.6483e-2
    a1 = (-1.8501e-4 * t + 2.19974e-2) * t - 0.603459
    kw = ((-2.0621152e-4 * t + 4.081431e-2) * t - 4.65421) * t + 148.4206
    k0t = (b1 * sr + a1) * sal + kw
    drv[1, 3] = k0t
    b1 = -1.06018e-3
    a1 = -3.7002e-4 * t + 2.19974e-2
    kw = (-6.1863456e-4 * t + 8.162862e-2) * t - 4.65421
    k0tt = (b1 * sr + a1) * sal + kw
    drv[2, 3] = k0tt
    dk = (b * p + a) * p + k0
    k35 = (5.03217e-5 * p + 3.359406) * p + 21582.27
    gam = p / k35
    pk = 1.0 - gam
    sva = sva * pk + (v350p + sva) * p * dk / (k35 * (k35 + dk))
    svan_val = sva * 1.0e8
    eos8d = svan_val
    v350p = v350p * pk
    dr35p = gam / v350p
    dvan = sva / (v350p * (v350p + sva))
    sigma = dr350 + dr35p - dvan
    drv[0, 2] = sigma
    k = k35 + dk
    vp = 1.0 - p / k
    kt = (bt * p + at) * p + k0t
    ktt = (btt * p + att) * p + k0tt
    v = 1.0 / (sigma + 1000.0)
    drv[0, 0] = v
    v2 = v * v
    vs = v0s * vp + v0 * p * ks / (k * k)
    rhos = -vs / v2
    drv[2, 7] = vs
    drv[0, 7] = rhos
    vt = v0t * vp + v0 * p * kt / (k * k)
    vtt = v0tt * vp + p * (
        2.0 * v0t * kt + ktt * v0 - 2.0 * kt * kt * v0 / k
    ) / (k * k)
    r0tt = (2.0 * vt * vt / v - vtt) / v2
    drv[2, 2] = r0tt
    drv[1, 0] = vt
    drv[2, 0] = vtt
    rhot = -vt / v2
    drv[1, 2] = rhot
    a = drv[0, 4]
    b = drv[0, 5]
    dkdp = 2.0 * b * p + a
    dvdp = -0.1 * v0 * (1.0 - p * dkdp / k) / k
    drv[0, 6] = -dvdp / v2
    drv[1, 6] = k
    drv[2, 6] = dvdp
    return eos8d, drv


@njit
def eosall(s, t, p0):
    pr = 0.0
    thet = theta(s, t, p0, pr)
    edum, drv = eos8d(s, t, p0)
    alfold = -drv[1, 2] / (drv[0, 2] + 1000.0)
    betold = drv[0, 7] / (drv[0, 2] + 1000.0)
    gamold = drv[0, 6] / (drv[0, 2] + 1000.0)
    sdum, sigthe = svan(s, thet, pr)
    dthedt, dtheds, dthedp = derthe(s, t, p0)
    alfnew = alfold / dthedt
    betnew = betold + alfnew * dtheds
    gamnew = gamold + alfnew * dthedp
    soundv = math.sqrt(abs(1.0e4 / (gamnew * (drv[0, 2] + 1000.0))))
    return thet, sigthe, alfnew, betnew, gamnew, soundv


@njit
def theta(s, t0, p0, pr):
    p = p0
    t = t0
    h = pr - p
    xk = h * atg(s, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * atg(s, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * atg(s, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * atg(s, t, p)
    theta = t + (xk - 2.0 * q) / 6.0
    return theta


@njit
def svan(s, t, p0):
    r3500, r4 = 1028.1063, 4.8314e-4
    dr350 = 28.106331
    p = p0 / 10.0
    sr = math.sqrt(abs(s))
    r1 = (
        (((6.536332e-9 * t - 1.120083e-6) * t + 1.001685e-4) * t - 9.095290e-3)
        * t
        + 6.793952e-2
    ) * t - 28.263737
    r2 = (
        ((5.3875e-9 * t - 8.2467e-7) * t + 7.6438e-5) * t - 4.0899e-3
    ) * t + 8.24493e-1
    r3 = (-1.6546e-6 * t + 1.0227e-4) * t - 5.72466e-3
    sig = (r4 * s + r3 * sr + r2) * s + r1
    v350p = 1.0 / r3500
    sva = -sig * v350p / (r3500 + sig)
    sigma = sig + dr350
    svan_val = sva * 1.0e8
    if p == 0.0:
        return svan_val, sigma
    e = (9.1697e-10 * t + 2.0816e-8) * t - 9.9348e-7
    bw = (5.2787e-8 * t - 6.12293e-6) * t + 3.47718e-5
    b = bw + e * s
    d = 1.91075e-4
    c = (-1.6078e-6 * t - 1.0981e-5) * t + 2.2838e-3
    aw = ((-5.77905e-7 * t + 1.16092e-4) * t + 1.43713e-3) * t - 0.1194975
    a = (d * sr + c) * s + aw
    b1 = (-5.3009e-4 * t + 1.6483e-2) * t + 7.944e-2
    a1 = ((-6.1670e-5 * t + 1.09987e-2) * t - 0.603459) * t + 54.6746
    kw = (
        ((-5.155288e-5 * t + 1.360477e-2) * t - 2.327105) * t + 148.4206
    ) * t - 1930.06
    k0 = (b1 * sr + a1) * s + kw
    dk = (b * p + a) * p + k0
    k35 = (5.03217e-5 * p + 3.359406) * p + 21582.27
    gam = p / k35
    pk = 1.0 - gam
    sva = sva * pk + (v350p + sva) * p * dk / (k35 * (k35 + dk))
    svan_val = sva * 1.0e8
    v350p = v350p * pk
    dr35p = gam / v350p
    dvan = sva / (v350p * (v350p + sva))
    sigma = dr350 + dr35p - dvan
    return svan_val, sigma


@njit
def e_solve(s, t, p, e, n, k, s0, t0, p0):
    n2 = 2
    pl = p[k - 1]
    el = e[k - 1]
    pu = p[k]
    eu = e[k]
    iter = 0
    isuccess = 0
    while isuccess == 0:
        iter += 1
        pm = (pl + pu) / 2
        sm, tm = stp_interp(
            s[k - 1 : k + 1], t[k - 1 : k + 1], p[k - 1 : k + 1], n2, pm
        )
        sigl, sigu = sig_vals(s0, t0, p0, sm, tm, pm)
        em = sigu - sigl
        if el * em < 0.0:
            pu = pm
            eu = em
        elif em * eu < 0.0:
            pl = pm
            el = em
        elif em == 0.0:
            sns = sm
            tns = tm
            pns = pm
            isuccess = 1
        if isuccess == 0:
            if abs(em) <= 5e-5 and abs(pu - pl) <= 5e-3:
                sns = sm
                tns = tm
                pns = pm
                isuccess = 1
            elif iter <= 20:
                isuccess = 0
            else:
                print("WARNING 1 in e-solve.f")
                sns = -99.0
                tns = -99.0
                pns = -99.0
                isuccess = 1
    return sns, tns, pns, iter


@njit
def sig_vals(s1, t1, p1, s2, t2, p2):
    pmid = (p1 + p2) / 2.0
    sd1, sig1 = svan(s1, theta(s1, t1, p1, pmid), pmid)
    sd2, sig2 = svan(s2, theta(s2, t2, p2, pmid), pmid)
    return sig1, sig2


@njit
def stp_interp(s, t, p, n, p0):
    k = indx(p, n, p0)
    r = (p0 - p[k]) / (p[k + 1] - p[k])
    s0 = s[k] + r * (s[k + 1] - s[k])
    thk = theta(s[k], t[k], p[k], 0.0)
    th0 = thk + r * (theta(s[k + 1], t[k + 1], p[k + 1], 0.0) - thk)
    t0 = theta(s0, th0, 0.0, p0)
    return s0, t0


@njit
def indx(x, n, z):
    if x[0] < z and z < x[n - 1]:
        kl = 0
        ku = n - 1
        while ku - kl > 1:
            km = (ku + kl) // 2
            if z > x[km]:
                kl = km
            else:
                ku = km
        k = kl
        if z == x[k + 1]:
            k = k + 1
    else:
        if z == x[0]:
            k = 0
        elif z == x[n - 1]:
            k = n - 2
        else:
            print("ERROR 1 in indx.f : out of range")
            raise ValueError("ERROR 1 in indx.f")
    return k


@njit
def depth_scv(s, t, p, n, s0, t0, p0):
    n_max = 2000
    nscv_max = 50
    e = np.zeros(n_max)
    sscv = np.zeros(nscv_max)
    tscv = np.zeros(nscv_max)
    pscv = np.zeros(nscv_max)
    n2 = 2
    if n > n_max:
        raise ValueError("n > n_max")
    ncr = 0
    nscv = 0
    for k in range(n):
        sdum, sigl = svan(s0, theta(s0, t0, p0, p[k]), p[k])
        sdum, sigu = svan(s[k], t[k], p[k])
        e[k] = sigu - sigl
        sscv_tmp, tscv_tmp, pscv_tmp = 0.0, 0.0, 0.0
        if k > 0:
            if e[k - 1] == 0.0:
                ncr += 1
                sscv_tmp = s[k - 1]
                tscv_tmp = t[k - 1]
                pscv_tmp = p[k - 1]
            elif e[k] * e[k - 1] < 0.0:
                ncr += 1
                pc0 = p[k - 1] - e[k - 1] * (p[k] - p[k - 1]) / (
                    e[k] - e[k - 1]
                )
                iter = 0
                isuccess = 0
                pc_0 = 0.0
                ec_0 = 0.0
                while isuccess == 0:
                    iter += 1
                    sc0, tc0 = stp_interp(
                        s[k - 1 : k + 1],
                        t[k - 1 : k + 1],
                        p[k - 1 : k + 1],
                        n2,
                        pc0,
                    )
                    sdum, sigl = svan(s0, theta(s0, t0, p0, pc0), pc0)
                    sdum, sigu = svan(sc0, tc0, pc0)
                    ec0 = sigu - sigl
                    p1 = (p[k - 1] + pc0) / 2
                    ez1 = (e[k - 1] - ec0) / (pc0 - p[k - 1])
                    p2 = (pc0 + p[k]) / 2
                    ez2 = (ec0 - e[k]) / (p[k] - pc0)
                    r = (pc0 - p1) / (p2 - p1)
                    ecz_0 = ez1 + r * (ez2 - ez1)
                    if iter == 1:
                        ecz0 = ecz_0
                    else:
                        ecz0 = -(ec0 - ec_0) / (pc0 - pc_0)
                        if ecz0 == 0:
                            ecz0 = ecz_0
                    pc1 = pc0 + ec0 / ecz0
                    if pc1 <= p[k - 1] or pc1 >= p[k]:
                        sscv_tmp, tscv_tmp, pscv_tmp, niter = scv_solve(
                            s, t, p, e, n, k, s0, t0, p0
                        )
                        if pscv_tmp < p[k - 1] or pscv_tmp > p[k]:
                            raise ValueError("ERROR 1 in depth-scv.f")
                        else:
                            isuccess = 1
                    else:
                        eps = abs(pc1 - pc0)
                        if abs(ec0) <= 5e-5 and eps <= 5e-3:
                            sscv_tmp = sc0
                            tscv_tmp = tc0
                            pscv_tmp = pc0
                            isuccess = 1
                            niter = iter
                        elif iter > 10:
                            sscv_tmp, tscv_tmp, pscv_tmp, niter = scv_solve(
                                s, t, p, e, n, k, s0, t0, p0
                            )
                            isuccess = 1
                        else:
                            pc_0 = pc0
                            ec_0 = ec0
                            pc0 = pc1
                            isuccess = 0
            if k == n - 1 and e[k] == 0.0:
                ncr += 1
                sscv_tmp = s[k]
                tscv_tmp = t[k]
                pscv_tmp = p[k]
            if ncr > nscv:
                nscv += 1
                if nscv > nscv_max:
                    raise ValueError("ERROR 2 in depth-scv.f")
                sscv[nscv - 1] = sscv_tmp
                tscv[nscv - 1] = tscv_tmp
                pscv[nscv - 1] = pscv_tmp
    if nscv == 0:
        sscv[0] = -99.0
        tscv[0] = -99.0
        pscv[0] = -99.0
    return sscv, tscv, pscv, nscv


@njit
def scv_solve(s, t, p, e, n, k, s0, t0, p0):
    n2 = 2
    pl = p[k - 1]
    el = e[k - 1]
    pu = p[k]
    eu = e[k]
    iter = 0
    isuccess = 0
    sscv = -99.0
    tscv = -99.0
    pscv = -99.0
    while isuccess == 0:
        iter += 1
        pm = (pl + pu) / 2
        sm, tm = stp_interp(
            s[k - 1 : k + 1], t[k - 1 : k + 1], p[k - 1 : k + 1], n2, pm
        )
        sdum, sigl = svan(s0, theta(s0, t0, p0, pm), pm)
        sdum, sigu = svan(sm, tm, pm)
        em = sigu - sigl
        if el * em < 0.0:
            pu = pm
            eu = em
        elif em * eu < 0.0:
            pl = pm
            el = em
        elif em == 0.0:
            sscv = sm
            tscv = tm
            pscv = pm
            isuccess = 1
        if isuccess == 0:
            if abs(em) <= 5e-5 and abs(pu - pl) <= 5e-3:
                sscv = sm
                tscv = tm
                pscv = pm
                isuccess = 1
            elif iter <= 20:
                isuccess = 0
            else:
                sscv = -99.0
                tscv = -99.0
                pscv = -99.0
                isuccess = 1
    return sscv, tscv, pscv, iter


@njit
def gamma_qdr(pl, gl, a, pu, gu, p):
    p1 = (p - pu) / (pu - pl)
    p2 = (p - pl) / (pu - pl)
    gamma = (a * p1 + (gu - gl)) * p2 + gl
    return gamma


@njit
def goor_solve(sl, tl, el, su, tu, eu, p, s0, t0, p0, sigb):
    rl = 0.0
    ru = 1.0
    pmid = (p + p0) / 2.0
    thl = theta(sl, tl, p, pmid)
    thu = theta(su, tu, p, pmid)
    iter = 0
    isuccess = 0
    sns = -99.0
    tns = -99.0
    while isuccess == 0:
        iter += 1
        rm = (rl + ru) / 2
        sm = sl + rm * (su - sl)
        thm = thl + rm * (thu - thl)
        tm = theta(sm, thm, pmid, p)
        sd, sigma = svan(sm, thm, pmid)
        em = sigma - sigb
        if el * em < 0.0:
            ru = rm
            eu = em
        elif em * eu < 0.0:
            rl = rm
            el = em
        elif em == 0.0:
            sns = sm
            tns = tm
            isuccess = 1
        if isuccess == 0:
            if abs(em) <= 5e-5 and abs(ru - rl) <= 5e-3:
                sns = sm
                tns = tm
                isuccess = 1
            elif iter <= 20:
                isuccess = 0
            else:
                sns = sm
                tns = tm
                isuccess = 1
    return sns, tns


@njit
def goor(s, t, p, gamma, n, sb, tb, pb):
    delt_b = -0.1
    delt_t = 0.1
    slope = -0.14
    pr0 = 0.0
    tbp = 2.7e-8
    pmid = (p[n - 1] + pb) / 2.0
    sd, sigma = svan(s[n - 1], theta(s[n - 1], t[n - 1], p[n - 1], pmid), pmid)
    sd, sigb = svan(sb, theta(sb, tb, pb, pmid), pmid)
    if sigb > sigma:
        n_sth = 0
        s_new = s[n - 1]
        t_new = t[n - 1]
        e_new = sigma - sigb
        s_old = 0.0
        t_old = 0.0
        e_old = 0.0
        while sigma < sigb:
            s_old = s_new
            t_old = t_new
            e_old = e_new
            n_sth += 1
            s_new = s[n - 1] + n_sth * delt_b * slope
            t_new = t[n - 1] + n_sth * delt_b
            sd, sigma = svan(s_new, theta(s_new, t_new, p[n - 1], pmid), pmid)
            e_new = sigma - sigb
        if sigma == sigb:
            sns = s_new
            tns = t_new
        else:
            sns, tns = goor_solve(
                s_old,
                t_old,
                e_old,
                s_new,
                t_new,
                e_new,
                p[n - 1],
                sb,
                tb,
                pb,
                sigb,
            )
        sigl, sigu = sig_vals(
            s[n - 2], t[n - 2], p[n - 2], s[n - 1], t[n - 1], p[n - 1]
        )
        bmid = (gamma[n - 1] - gamma[n - 2]) / (sigu - sigl)
        sd, sigl = svan(s[n - 1], t[n - 1], p[n - 1])
        sd, sigu = svan(sns, tns, p[n - 1])
        gammab = gamma[n - 1] + bmid * (sigu - sigl)
        pns = p[n - 1]
    else:
        pmid = (p[0] + pb) / 2.0
        sd, sigma = svan(s[0], theta(s[0], t[0], p[0], pmid), pmid)
        sd, sigb = svan(sb, theta(sb, tb, pb, pmid), pmid)
        if sigb < sigma:
            n_sth = 0
            s_new = s[0]
            t_new = t[0]
            e_new = sigma - sigb
            s_old = 0.0
            t_old = 0.0
            e_old = 0.0
            while sigma > sigb:
                s_old = s_new
                t_old = t_new
                e_old = e_new
                n_sth += 1
                s_new = s[0]
                t_new = t[0] + n_sth * delt_t
                sd, sigma = svan(s_new, theta(s_new, t_new, p[0], pmid), pmid)
                e_new = sigma - sigb
            if sigma == sigb:
                sns = s_new
                tns = t_new
            else:
                sns, tns = goor_solve(
                    s_new,
                    t_new,
                    e_new,
                    s_old,
                    t_old,
                    e_old,
                    p[0],
                    sb,
                    tb,
                    pb,
                    sigb,
                )
            sigl, sigu = sig_vals(s[0], t[0], p[0], s[1], t[1], p[1])
            bmid = (gamma[1] - gamma[0]) / (sigu - sigl)
            sd, sigl = svan(sns, tns, p[0])
            sd, sigu = svan(s[0], t[0], p[0])
            gammab = gamma[0] - bmid * (sigu - sigl)
            pns = p[0]
        else:
            raise ValueError("ERROR 1 in gamma-out-of-range.f")
    thb = theta(sb, tb, pb, pr0)
    thns = theta(sns, tns, pns, pr0)
    sdum, sig_ns = svan(sns, tns, pns)
    rho_ns = 1000 + sig_ns
    b = bmid
    dp = pns - pb
    dth = thns - thb
    g1_err = rho_ns * b * tbp * abs(dp * dth) / 6
    g2_err = rho_ns * b * tbp * dp * dth / 2
    if g2_err <= 0.0:
        g2_l_err = -g2_err
        g2_h_err = 0.0
    else:
        g2_l_err = 0.0
        g2_h_err = g2_err
    return gammab, g1_err, g2_l_err, g2_h_err


@njit
def ocean_test(x1, y1, io1, x2, y2, io2, z):
    x_js = np.array([129.87, 140.37, 142.83])
    y_js = np.array([32.75, 37.38, 53.58])
    y = (y1 + y2) / 2
    if io1 == io2:
        itest = 1
        return itest
    elif y <= -20.0:
        if y >= -48.0 and (io1 * io2) == 12:
            itest = 0
        else:
            itest = 1
    elif (io1 == 1 or io1 == 2) and (io2 == 1 or io2 == 2):
        itest = 1
    elif (io1 == 3 or io1 == 4) and (io2 == 3 or io2 == 4):
        itest = 1
    elif (io1 == 5 or io1 == 6) and (io2 == 5 or io2 == 6):
        itest = 1
    elif (
        io1 * io2 == 8
        and z <= 1200.0
        and 124.0 <= x1 <= 132.0
        and 124.0 <= x2 <= 132.0
    ):
        itest = 1
    else:
        itest = 0
    if (x_js[0] <= x1 <= x_js[2] and y_js[0] <= y1 <= y_js[2]) or (
        x_js[0] <= x2 <= x_js[2] and y_js[0] <= y2 <= y_js[2]
    ):
        em1 = (y_js[1] - y_js[0]) / (x_js[1] - x_js[0])
        c1 = y_js[0] - em1 * x_js[0]
        em2 = (y_js[2] - y_js[1]) / (x_js[2] - x_js[1])
        c2 = y_js[1] - em2 * x_js[1]
        if (y1 - em1 * x1 - c1) >= 0.0 and (y1 - em2 * x1 - c2) >= 0.0:
            isj1 = 1
        else:
            isj1 = 0
        if (y2 - em1 * x2 - c1) >= 0.0 and (y2 - em2 * x2 - c2) >= 0.0:
            isj2 = 1
        else:
            isj2 = 0
        if isj1 == isj2:
            itest = 1
        else:
            itest = 0
    if io1 * io2 == 12 and y < -60.0:
        itest = 0
    return itest


@njit
def gamma_errors(
    s, t, p, gamma, a, n, along, alat, s0, t0, p0, sns, tns, pns, kns, gamma_ns
):
    pr0 = 0.0
    tb = 2.7e-8
    gamma_limit = 26.845
    test_limit = 0.1
    th0 = theta(s0, t0, p0, pr0)
    thns = theta(sns, tns, pns, pr0)
    sdum, sig_ns = svan(sns, tns, pns)
    rho_ns = 1000 + sig_ns
    sig_l, sig_h = sig_vals(
        s[kns], t[kns], p[kns], s[kns + 1], t[kns + 1], p[kns + 1]
    )
    b = (gamma[kns + 1] - gamma[kns]) / (sig_h - sig_l)
    dp = pns - p0
    dth = thns - th0
    pth_error = rho_ns * b * tb * abs(dp * dth) / 6
    scv_l_error = 0.0
    scv_h_error = 0.0
    if alat <= -60.0 or gamma[0] >= gamma_limit:
        drldp = (sig_h - sig_l) / (rho_ns * (p[kns + 1] - p[kns]))
        test = tb * dth / drldp
        if abs(test) <= test_limit:
            if dp * dth >= 0.0:
                scv_h_error = (3 * pth_error) / (1.0 - test)
            else:
                scv_l_error = (3 * pth_error) / (1.0 - test)
        else:
            sscv_m, tscv_m, pscv_m, nscv = depth_scv(s, t, p, n, s0, t0, p0)
            if nscv > 0:
                if nscv == 1:
                    pscv = pscv_m[0]
                else:
                    pscv_mid = pscv_m[nscv // 2]
                    if p0 <= pscv_mid:
                        pscv = pscv_m[0]
                    else:
                        pscv = pscv_m[nscv - 1]
                kscv = indx(p, n, pscv)
                gamma_scv = gamma_qdr(
                    p[kscv],
                    gamma[kscv],
                    a[kscv],
                    p[kscv + 1],
                    gamma[kscv + 1],
                    pscv,
                )
                if pscv <= pns:
                    scv_l_error = gamma_ns - gamma_scv
                else:
                    scv_h_error = gamma_scv - gamma_ns
    if pth_error < 0.0 or scv_l_error < 0.0 or scv_h_error < 0.0:
        raise ValueError("ERROR 1 in gamma-errors: negative scv error")
    return pth_error, scv_l_error, scv_h_error


@njit
def read_nc(along, alat, s0, t0, p0, gamma0, a0, n0, along0, alat0, iocean0):
    nx, nz, ndx, ndy = 90, 33, 4, 4

    i0 = int(along / ndx) + 1
    j0 = int((88 + alat) / ndy) + 1

    if i0 == nx + 1:
        i0 = 1

    for k in range(nz):
        p0[k] = p0_s_global[k]

    along0[0] = along_d[i0 - 1]
    alat0[0] = alat_d[j0 - 1]
    alat0[1] = alat0[0] + ndy

    if i0 < nx:
        along0[1] = along0[0] + ndx
        krec1 = (i0 - 1) + (j0 - 1) * nx
        krec2 = i0 + (j0 - 1) * nx
        krec3 = (i0 - 1) + j0 * nx
        krec4 = i0 + j0 * nx
    else:
        along0[1] = 0.0
        krec1 = (i0 - 1) + (j0 - 1) * nx
        krec2 = 0 + (j0 - 1) * nx
        krec3 = (i0 - 1) + j0 * nx
        krec4 = 0 + j0 * nx

    for k in range(nz):
        s0[k, 0, 0] = stga_data[krec1, 0, k]
        t0[k, 0, 0] = stga_data[krec1, 1, k]
        gamma0[k, 0, 0] = stga_data[krec1, 2, k]
        a0[k, 0, 0] = stga_data[krec1, 3, k]

        s0[k, 1, 0] = stga_data[krec2, 0, k]
        t0[k, 1, 0] = stga_data[krec2, 1, k]
        gamma0[k, 1, 0] = stga_data[krec2, 2, k]
        a0[k, 1, 0] = stga_data[krec2, 3, k]

        s0[k, 0, 1] = stga_data[krec3, 0, k]
        t0[k, 0, 1] = stga_data[krec3, 1, k]
        gamma0[k, 0, 1] = stga_data[krec3, 2, k]
        a0[k, 0, 1] = stga_data[krec3, 3, k]

        s0[k, 1, 1] = stga_data[krec4, 0, k]
        t0[k, 1, 1] = stga_data[krec4, 1, k]
        gamma0[k, 1, 1] = stga_data[krec4, 2, k]
        a0[k, 1, 1] = stga_data[krec4, 3, k]

    i0_0 = int(along0[0] / ndx) + 1
    i0_1 = int(along0[1] / ndx) + 1
    j0_0 = int((88 + alat0[0]) / ndy) + 1
    j0_1 = int((88 + alat0[1]) / ndy) + 1

    n0[0, 0] = n_global[i0_0 - 1, j0_0 - 1]
    n0[1, 0] = n_global[i0_1 - 1, j0_0 - 1]
    n0[0, 1] = n_global[i0_0 - 1, j0_1 - 1]
    n0[1, 1] = n_global[i0_1 - 1, j0_1 - 1]

    iocean0[0, 0] = iocean_global[i0_0 - 1, j0_0 - 1]
    iocean0[1, 0] = iocean_global[i0_1 - 1, j0_0 - 1]
    iocean0[0, 1] = iocean_global[i0_0 - 1, j0_1 - 1]
    iocean0[1, 1] = iocean_global[i0_1 - 1, j0_1 - 1]


@njit
def gamma_n(s, t, p, n, along, alat, gamma, dg_lo, dg_hi):
    nz, ndx, ndy = 33, 4, 4
    iocean0 = np.zeros((2, 2), dtype=np.int32)
    n0 = np.zeros((2, 2), dtype=np.int32)
    along0 = np.zeros(2)
    alat0 = np.zeros(2)
    s0 = np.zeros((nz, 2, 2))
    t0 = np.zeros((nz, 2, 2))
    p0 = np.zeros(nz)
    gamma0 = np.zeros((nz, 2, 2))
    a0 = np.zeros((nz, 2, 2))
    gwij = np.zeros(4)
    wtij = np.zeros(4)
    pr0 = 0.0
    dgamma_0 = 0.0005
    dgw_max = 0.3

    if along < 0.0:
        along += 360.0
        ialtered = 1
    elif along == 360.0:
        along = 0.0
        ialtered = 2
    else:
        ialtered = 0

    if along < 0.0 or along > 360.0 or alat < -90.0 or alat > 90.0:
        raise ValueError("ERROR 1 in gamma-n.f : out of oceanographic range")

    for k in range(n):
        if (
            s[k] < 0.0
            or s[k] > 42.0
            or t[k] < -2.5
            or t[k] > 40.0
            or p[k] < 0.0
            or p[k] > 10000.0
        ):
            gamma[k] = -99.1
            dg_lo[k] = -99.1
            dg_hi[k] = -99.1
        else:
            gamma[k] = 0.0
            dg_lo[k] = 0.0
            dg_hi[k] = 0.0

    read_nc(along, alat, s0, t0, p0, gamma0, a0, n0, along0, alat0, iocean0)

    dist2_min = 1e10
    i_min, j_min = 0, 0
    for j0 in range(2):
        for i0 in range(2):
            if n0[i0, j0] != 0:
                dist2 = (along0[i0] - along) ** 2 + (alat0[j0] - alat) ** 2
                if dist2 < dist2_min:
                    i_min = i0
                    j_min = j0
                    dist2_min = dist2

    ioce = iocean0[i_min, j_min]
    dx = abs(along % ndx)
    dy = abs((alat + 80.0) % ndy)
    rx = dx / ndx
    ry = dy / ndy

    for k in range(n):
        if gamma[k] != -99.1:
            thk = theta(s[k], t[k], p[k], pr0)
            dgamma_1 = 0.0
            dgamma_2_l = 0.0
            dgamma_2_h = 0.0
            wsum = 0.0
            nij = 0

            for j0_idx in range(2):
                for i0_idx in range(2):
                    if n0[i0_idx, j0_idx] != 0:
                        if j0_idx == 0:
                            if i0_idx == 0:
                                wt = (1.0 - rx) * (1.0 - ry)
                            elif i0_idx == 1:
                                wt = rx * (1.0 - ry)
                        elif j0_idx == 1:
                            if i0_idx == 0:
                                wt = (1.0 - rx) * ry
                            elif i0_idx == 1:
                                wt = rx * ry

                        wt += 1e-6
                        itest = ocean_test(
                            along,
                            alat,
                            ioce,
                            along0[i0_idx],
                            alat0[j0_idx],
                            iocean0[i0_idx, j0_idx],
                            p[k],
                        )
                        if itest == 0:
                            wt = 0.0

                        sns, tns, pns = depth_ns(
                            s0[:, i0_idx, j0_idx],
                            t0[:, i0_idx, j0_idx],
                            p0,
                            n0[i0_idx, j0_idx],
                            s[k],
                            t[k],
                            p[k],
                        )

                        if pns > -99.0:
                            kns = indx(p0, n0[i0_idx, j0_idx], pns)
                            gw = gamma_qdr(
                                p0[kns],
                                gamma0[kns, i0_idx, j0_idx],
                                a0[kns, i0_idx, j0_idx],
                                p0[kns + 1],
                                gamma0[kns + 1, i0_idx, j0_idx],
                                pns,
                            )
                            g1_err, g2_l_err, g2_h_err = gamma_errors(
                                s0[:, i0_idx, j0_idx],
                                t0[:, i0_idx, j0_idx],
                                p0,
                                gamma0[:, i0_idx, j0_idx],
                                a0[:, i0_idx, j0_idx],
                                n0[i0_idx, j0_idx],
                                along0[i0_idx],
                                alat0[j0_idx],
                                s[k],
                                t[k],
                                p[k],
                                sns,
                                tns,
                                pns,
                                kns,
                                gw,
                            )
                        elif pns == -99.0:
                            gw, g1_err, g2_l_err, g2_h_err = goor(
                                s0[:, i0_idx, j0_idx],
                                t0[:, i0_idx, j0_idx],
                                p0,
                                gamma0[:, i0_idx, j0_idx],
                                n0[i0_idx, j0_idx],
                                s[k],
                                t[k],
                                p[k],
                            )
                            if (
                                gw
                                > gamma0[
                                    n0[i0_idx, j0_idx] - 1, i0_idx, j0_idx
                                ]
                            ):
                                rw = (
                                    min(
                                        dgw_max,
                                        gw
                                        - gamma0[
                                            n0[i0_idx, j0_idx] - 1,
                                            i0_idx,
                                            j0_idx,
                                        ],
                                    )
                                    / dgw_max
                                )
                                wt = (1.0 - rw) * wt
                        else:
                            gw = 0.0
                            g1_err = 0.0
                            g2_l_err = 0.0
                            g2_h_err = 0.0

                        if gw > 0.0:
                            gamma[k] += wt * gw
                            dgamma_1 += wt * g1_err
                            dgamma_2_l = max(dgamma_2_l, g2_l_err)
                            dgamma_2_h = max(dgamma_2_h, g2_h_err)
                            wsum += wt
                            wtij[nij] = wt
                            gwij[nij] = gw
                            nij += 1

            if wsum != 0.0:
                gamma[k] /= wsum
                dgamma_1 /= wsum
                dgamma_3 = 0.0
                for ij in range(nij):
                    dgamma_3 += wtij[ij] * abs(gwij[ij] - gamma[k])
                dgamma_3 /= wsum
                dg_lo[k] = max(dgamma_0, dgamma_1, dgamma_2_l, dgamma_3)
                dg_hi[k] = max(dgamma_0, dgamma_1, dgamma_2_h, dgamma_3)
            else:
                gamma[k] = -99.0
                dg_lo[k] = -99.0
                dg_hi[k] = -99.0

    if ialtered == 1:
        along -= 360.0
    elif ialtered == 2:
        along = 360.0


@njit
def neutral_surfaces(
    s, t, p, gamma, n, glevels, ng, sns, tns, pns, dsns, dtns, dpns
):
    nint_max = 50
    int_arr = np.zeros(nint_max, dtype=np.int32)
    n2 = 2
    pr0 = 0.0
    ptol = 1.0e-3
    in_error = 0

    for k in range(n):
        if gamma[k] <= 0.0:
            in_error = 1

    if in_error == 1:
        raise ValueError("ERROR 1 in neutral-surfaces.f : missing gamma value")

    for ig in range(ng):
        nint = 0
        for k in range(n - 1):
            gmin = min(gamma[k], gamma[k + 1])
            gmax = max(gamma[k], gamma[k + 1])
            if gmin <= glevels[ig] <= gmax:
                int_arr[nint] = k
                nint += 1
                if nint > nint_max:
                    raise ValueError("ERROR 2 in neutral-surfaces.f")

        if nint == 0:
            sns[ig] = -99.0
            tns[ig] = -99.0
            pns[ig] = -99.0
            dsns[ig] = 0.0
            dtns[ig] = 0.0
            dpns[ig] = 0.0
        else:
            if nint % 2 == 0 and int_arr[0] > (n - 1) // 2:
                int_middle = (nint + 2) // 2
            else:
                int_middle = (nint + 1) // 2

            for i_int in range(nint):
                k = int_arr[i_int]
                pmid = (p[k] + p[k + 1]) / 2.0
                thdum, sthdum, alfa_l, beta_l, gdum, sdum = eosall(
                    s[k], t[k], p[k]
                )
                thdum, sthdum, alfa_u, beta_u, gdum, sdum = eosall(
                    s[k + 1], t[k + 1], p[k + 1]
                )
                alfa_mid = (alfa_l + alfa_u) / 2.0
                beta_mid = (beta_l + beta_u) / 2.0
                smid, tmid = stp_interp(
                    s[k : k + 2], t[k : k + 2], p[k : k + 2], n2, pmid
                )
                sd, sigmid = svan(smid, tmid, pmid)
                rhomid = 1000.0 + sigmid
                thl = theta(s[k], t[k], p[k], pr0)
                thu = theta(s[k + 1], t[k + 1], p[k + 1], pr0)
                dels = s[k + 1] - s[k]
                delth = thu - thl
                pl = p[k]
                pu = p[k + 1]
                delp = pu - pl
                delp2 = delp * delp
                bden = rhomid * (beta_mid * dels - alfa_mid * delth)
                if abs(bden) <= 1e-6:
                    bden = 1e-6
                bmid = (gamma[k + 1] - gamma[k]) / bden

                a = dels * (beta_u - beta_l) - delth * (alfa_u - alfa_l)
                a = (a * bmid * rhomid) / (2.0 * delp2)
                b = dels * (pu * beta_l - pl * beta_u) - delth * (
                    pu * alfa_l - pl * alfa_u
                )
                b = (b * bmid * rhomid) / delp2
                c = dels * (beta_l * (pl - 2.0 * pu) + beta_u * pl) - delth * (
                    alfa_l * (pl - 2.0 * pu) + alfa_u * pl
                )
                c = gamma[k] + (bmid * rhomid * pl * c) / (2.0 * delp2)
                c = c - glevels[ig]

                if a != 0.0 and bden != 1e-6:
                    q = -(b + np.sign(b) * np.sqrt(b * b - 4 * a * c)) / 2.0
                    pns1 = q / a
                    pns2 = c / q
                    if p[k] - ptol <= pns1 <= p[k + 1] + ptol:
                        pns[ig] = min(p[k + 1], max(pns1, p[k]))
                    elif p[k] - ptol <= pns2 <= p[k + 1] + ptol:
                        pns[ig] = min(p[k + 1], max(pns2, p[k]))
                    else:
                        raise ValueError("ERROR 3 in neutral-surfaces.f")
                else:
                    rg = (glevels[ig] - gamma[k]) / (gamma[k + 1] - gamma[k])
                    pns[ig] = p[k] + rg * (p[k + 1] - p[k])

                sns[ig], tns[ig] = stp_interp(s, t, p, n, pns[ig])

                if nint > 1:
                    if i_int == 0:
                        sns_top = sns[ig]
                        tns_top = tns[ig]
                        pns_top = pns[ig]
                    elif i_int == int_middle - 1:
                        sns_middle = sns[ig]
                        tns_middle = tns[ig]
                        pns_middle = pns[ig]
                    elif i_int == nint - 1:
                        if (pns_middle - pns_top) > (pns[ig] - pns_middle):
                            dsns[ig] = sns_middle - sns_top
                            dtns[ig] = tns_middle - tns_top
                            dpns[ig] = pns_middle - pns_top
                        else:
                            dsns[ig] = sns[ig] - sns_middle
                            dtns[ig] = tns[ig] - tns_middle
                            dpns[ig] = pns[ig] - pns_middle
                        sns[ig] = sns_middle
                        tns[ig] = tns_middle
                        pns[ig] = pns_middle
                else:
                    dsns[ig] = 0.0
                    dtns[ig] = 0.0
                    dpns[ig] = 0.0


def init_fdt():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    llp_path = os.path.join(package_dir, "llp.fdt")
    stga_path = os.path.join(package_dir, "stga.fdt")

    f = FortranFile(llp_path, "r")
    record = f.read_reals(dtype=np.float32)
    f.close()

    along_d[:] = record[0:90]
    alat_d[:] = record[90:135]
    p0_s_global[:] = record[135:168]
    n_global[:] = record[168:4218].view(np.int32).reshape((90, 45), order="F")
    iocean_global[:] = (
        record[4218:8268].view(np.int32).reshape((90, 45), order="F")
    )

    raw_stga = np.fromfile(stga_path, dtype=np.float32)
    stga_data[:] = raw_stga.reshape((-1, 4, 33))


init_fdt()
