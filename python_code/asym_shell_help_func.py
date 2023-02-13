# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from cmath import phase, polar
from scipy.special import jv as besselj
from scipy.special import hankel2 as besselh
from scipy.optimize import fsolve
from math import cos, pi, e
from scipy.integrate import simps


def besselh_prime(m, z):
    f = m / z * besselh(m, z) - besselh(m + 1, z)
    return f


def besselj_prime(m, z):
    f = m / z * besselj(m, z) - besselj(m + 1, z)
    return f


def V(m, w0, n, h):
    res_1 = besselj(m, w0 * h) * besselj_prime(m, w0 * n * h)
    res_2 = besselj_prime(m, w0 * h) * besselj(m, w0 * n * h)
    return res_1 - n * res_2


def W(m, w0, n):
    res_1 = besselh(m, w0) * besselh_prime(m, n * w0)
    res_2 = besselh(m, n * w0) * besselh_prime(m, w0)
    return res_1 - n * res_2


def R(m, w0, n, h=1):
    res_1 = besselj(m, w0 * h) * besselh_prime(m, n * w0 * h)
    res_2 = besselj_prime(m, w0 * h) * besselh(m, n * w0 * h)
    return res_1 - n * res_2


def Q(m, w0, n):
    res_1 = besselh(m, w0) * besselj_prime(m, n * w0)
    res_2 = besselj(m, n * w0) * besselh_prime(m, w0)
    return res_1 - n * res_2


def U_1(m, s, w0, n, h, d):
    numer = W(s, w0, n) * besselj(s - m, n * w0 * d) * besselj(m, n * w0 * h)
    determ = Q(s, w0, n) * besselh(s, n * w0)
    res = -1 / h * numer / determ
    return res


def U_2(m, s, w0, n, h, d):
    numer = V(s, w0, n, h) * besselj(s - m, n * w0 * d) * besselh(m, n * w0) * (-1) ** (s - m)
    determ = R(s, w0, n, h) * besselj(s, n * w0 * h)
    res = -h * numer / determ
    return res


def rs_1(m, w0, n, h, N, alpha, d):
    rs = 0
    for s in range(-N, N + 1):
        rs = rs + (-1.j) ** s * R(s, w0, n) * besselj(s - m, n * w0 * d) * e ** (-1.j * s * alpha)
    return rs * besselj(m, n * w0 * h) / h


def rs_2(m, w0, n, h, N, alpha, d):
    term_1 = -2.j / (pi * n * w0) * besselj(m, w0)
    term_2 = -  besselj(m, n * w0) * R(m, w0, n)
    return (-1.j) ** m * (term_1 + term_2) * e ** (-1.j * m * alpha)


def M_12(w0, n, h, d, N):
    conv_check = True
    M_ = np.zeros((2 * N + 1, 2 * N + 1), dtype=complex)
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            M_[i + N, j + N] = U_1(i, j, w0, n, h, d)
    if np.abs(M_[0, 0]) > 1 or np.abs(M_[0, 2 * N]) > 1 or np.abs(M_[2 * N, 0]) > 1 or np.abs(M_[2 * N, 2 * N]) > 1:
        conv_check = False
    return M_, conv_check


def M_21(w0, n, h, d, N):
    conv_check = True
    M_ = np.zeros((2 * N + 1, 2 * N + 1), dtype=complex)
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            M_[i + N, j + N] = U_2(i, j, w0, n, h, d)
    if np.abs(M_[0, 0]) > 1 or np.abs(M_[0, 2 * N]) > 1 or np.abs(M_[2 * N, 0]) > 1 or np.abs(M_[2 * N, 2 * N]) > 1:
        conv_check = False
    return M_, conv_check


def right_side(w0, n, h, alpha, N, d):
    R_1 = np.zeros((2 * N + 1, 1), dtype=complex)
    R_2 = np.zeros((2 * N + 1, 1), dtype=complex)
    for i in range(-N, N + 1):
        R_1[i + N] = rs_1(i, w0, n, h, N, alpha, d)
    for i in range(-N, N + 1):
        R_2[i + N] = rs_2(i, w0, n, h, N, alpha, d)
    R_ = np.concatenate((R_1, R_2), axis=0)
    return R_


def get_coefficients(w0, n, h, alpha, N, d):
    M11 = np.identity(2 * N + 1)
    M12, conv_1 = M_12(w0, n, h, d, N)
    M21, conv_2 = M_21(w0, n, h, d, N)
    M22 = np.identity(2 * N + 1)
    M = np.block([[M11, M12], [M21, M22]])
    if conv_1 == False or conv_2 == False:
        return [], [], [], []

    Rs = right_side(w0, n, h, alpha, N, d)
    solution = np.linalg.solve(M, Rs)
    x_m = solution[:2 * N + 1]
    y_m = solution[2 * N + 1:]
    a_m, b_m, c_m, d_m = [], [], [], []
    for m in range(-N, N + 1):
        a_temp = x_m[m + N] / besselj(m, n * w0 * h) / R(m, w0, n, h)
        c_temp = pi * n * w0 * h / (2.j) * V(m, w0, n, h) * a_temp
        a_m.append(a_temp)
        c_m.append(c_temp)
    for m in range(-N, N + 1):
        d_temp = y_m[m + N] / besselh(m, n * w0) / Q(m, w0, n)
        b_temp = -pi * n * w0 / (2.j) * (d_temp * W(m, w0, n) + (-1.j) ** m * e ** (-1.j * m * alpha) * R(m, w0, n))
        d_m.append(d_temp)
        b_m.append(b_temp)
    return a_m, b_m, c_m, d_m


# def get_plasma_refr_index(eps_inf, wp, gamma, w0):
#     return (eps_inf*(w0 ** 2 - 1.j * gamma * w0) - wp ** 2) ** 0.5 / (w0 ** 2 - 1.j * gamma * w0) ** 0.5


def find_nearest(wavelen_arr, n_arr, k_arr, wavelen_current_value):
    wavelen_arr = np.asarray(wavelen_arr)
    idx = (np.abs(wavelen_arr - wavelen_current_value)).argmin()
    # next neighbor
    n1 = n2 = 0
    k1 = k2 = 0
    t1 = t2 = 0

    if wavelen_arr[idx] == wavelen_current_value:
        return n_arr[idx], k_arr[idx]

    if wavelen_arr[idx] > wavelen_current_value > wavelen_arr[idx + 1]:
        t1 = wavelen_arr[idx]
        t2 = wavelen_arr[idx + 1]
        n1 = n_arr[idx]
        n2 = n_arr[idx + 1]
        k1 = k_arr[idx]
        k2 = k_arr[idx + 1]

    if wavelen_arr[idx - 1] > wavelen_current_value > wavelen_arr[idx]:
        t1 = wavelen_arr[idx - 1]
        t2 = wavelen_arr[idx]
        n1 = n_arr[idx - 1]
        n2 = n_arr[idx]
        k1 = k_arr[idx - 1]
        k2 = k_arr[idx]

    scaling_n = (n2 - n1) / (t2 - t1)
    n_value_interp = n1 + scaling_n * (wavelen_current_value - t1)
    scaling_k = (k2 - k1) / (t2 - t1)
    k_value_interp = k1 + scaling_k * (wavelen_current_value - t1)

    return n_value_interp, k_value_interp


def get_refractive_index(wavelength, metal):
    planck_const = 4.135667516 * 10 ** (-15)  # eV
    light_vel = 299792458  # m/s

    energy_1 = [0.64, 0.77, 0.89, 1.02, 1.14, 1.26, 1.39, 1.51, 1.64, 1.76, 1.88, 2.01, 2.13, 2.26, 2.38, 2.50, 2.63,
                2.75, 2.88, 3.00]
    energy_2 = [3.12, 3.25, 3.37, 3.50, 3.62, 3.74, 3.87, 3.99, 4.12, 4.24, 4.36, 4.49, 4.61, 4.74, 4.86, 4.98, 5.11,
                5.23, 5.36, 5.48, 5.60, 5.73, 5.85, 5.98, 6.10, 6.22, 6.35, 6.47, 6.60]
    energy = energy_1 + energy_2

    ###silver
    n_silver_1 = [0.24, 0.15, 0.13, 0.09, 0.04, 0.04, 0.04, 0.04, 0.03, 0.04, 0.05, 0.06, 0.05, 0.06, 0.05, 0.05, 0.05,
                  0.04, 0.04, 0.05, 0.05, 0.05, 0.07]
    n_silver_2 = [0.1, 0.14, 0.17, 0.81, 1.13, 1.34, 1.39, 1.41, 1.41, 1.38, 1.35, 1.33, 1.31, 1.30, 1.28, 1.28, 1.26,
                  1.25, 1.22, 1.20, 1.18, 1.15, 1.14, 1.12, 1.1, 1.07]
    n_silver = n_silver_1 + n_silver_2

    k_silver_1 = [14.08, 11.85, 10.1, 8.825, 7.795, 6.992, 6.312, 5.727, 5.242, 4.838]
    k_silver_2 = [4.483, 4.152, 3.858, 3.586, 3.324, 3.093, 2.869, 2.657, 2.462, 2.275]
    k_silver_3 = [2.07, 1.864, 1.657, 1.419, 1.142, 0.829, 0.392, 0.616, 0.964, 1.161]
    k_silver_4 = [1.264, 1.331, 1.372, 1.387, 1.393, 1.389, 1.378, 1.367, 1.357, 1.344]
    k_silver_5 = [1.342, 1.336, 1.325, 1.312, 1.296, 1.277, 1.255, 1.232, 1.212]
    k_silver = k_silver_1 + k_silver_2 + k_silver_3 + k_silver_4 + k_silver_5

    #### gold
    n_gold_1 = [0.92, 0.56, 0.43, 0.35, 0.27, 0.22, 0.17, 0.16, 0.14, 0.13]
    n_gold_2 = [0.14, 0.21, 0.29, 0.43, 0.62, 1.04, 1.31, 1.38, 1.45, 1.46]
    n_gold_3 = [1.47, 1.46, 1.48, 1.50, 1.48, 1.48, 1.54, 1.53, 1.53, 1.49]
    n_gold_4 = [1.47, 1.43, 1.38, 1.35, 1.33, 1.33, 1.32, 1.32, 1.3, 1.31]
    n_gold_5 = [1.30, 1.30, 1.30, 1.30, 1.33, 1.33, 1.34, 1.32, 1.28]
    n_gold = n_gold_1 + n_gold_2 + n_gold_3 + n_gold_4 + n_gold_5

    k_gold_1 = [13.78, 11.21, 9.519, 8.145, 7.150, 6.350, 5.663, 5.083, 4.542, 4.103]
    k_gold_2 = [3.697, 3.272, 2.863, 2.455, 2.081, 1.833, 1.849, 1.914, 1.948, 1.958]
    k_gold_3 = [1.952, 1.933, 1.895, 1.866, 1.871, 1.883, 1.898, 1.893, 1.889, 1.878]
    k_gold_4 = [1.869, 1.847, 1.803, 1.749, 1.688, 1.631, 1.577, 1.536, 1.497, 1.460]
    k_gold_5 = [1.427, 1.387, 1.350, 1.304, 1.277, 1.251, 1.226, 1.203, 1.188]
    k_gold = k_gold_1 + k_gold_2 + k_gold_3 + k_gold_4 + k_gold_5

    ## cooper
    n_copper_1 = [1.09, 0.76, 0.6, 0.48, 0.36, 0.32, 0.3, 0.26, 0.24, 0.21]
    n_copper_2 = [0.22, 0.30, 0.7, 1.02, 1.18, 1.22, 1.25, 1.24, 1.25, 1.28]
    n_copper_3 = [1.32, 1.33, 1.36, 1.37, 1.36, 1.34, 1.38, 1.38, 1.40, 1.42]
    n_copper_4 = [1.45, 1.46, 1.45, 1.41, 1.41, 1.37, 1.34, 1.28, 1.23, 1.18]
    n_copper_5 = [1.13, 1.08, 1.04, 1.01, 0.99, 0.98, 0.97, 0.95, 0.94]
    n_copper = n_copper_1 + n_copper_2 + n_copper_3 + n_copper_4 + n_copper_5

    k_copper_1 = [13.43, 11.12, 9.439, 8.245, 7.217, 6.421, 5.768, 5.180, 4.665, 4.205]
    k_copper_2 = [3.747, 3.205, 2.704, 2.577, 2.608, 2.564, 2.483, 2.397, 2.305, 2.207]
    k_copper_3 = [2.116, 2.045, 1.975, 1.916, 1.864, 1.821, 1.783, 1.729, 1.679, 1.633]
    k_copper_4 = [1.633, 1.646, 1.668, 1.691, 1.741, 1.783, 1.799, 1.802, 1.792, 1.768]
    k_copper_5 = [1.737, 1.699, 1.651, 1.599, 1.550, 1.493, 1.440, 1.388, 1.337]
    k_copper = k_copper_1 + k_copper_2 + k_copper_3 + k_copper_4 + k_copper_5

    wavelength_range = planck_const * light_vel / np.array(energy) * 10 ** 9  # in nm
    if metal == 'silver':
        n_, k_ = n_silver, k_silver

    if metal == 'copper':
        n_, k_ = n_copper, k_copper

    if metal == 'gold':
        n_, k_ = n_gold, k_gold

    n_value_interp, k_value_interp = find_nearest(wavelength_range, n_, k_, wavelength)

    return n_value_interp, k_value_interp
