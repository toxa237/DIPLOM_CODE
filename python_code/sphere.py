import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, yve, yvp, sph_harm  # yve функції неймана
import tqdm
from sympy import Matrix, solve_linear_system
from sympy.abc import x, y
import plotly.graph_objects as go


def system_cords_sphere(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if z != 0:
        theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z) + np.pi/2
    else:
        theta = np.pi/2
    if x > 0:
        phi = np.arctan(y / x)
    elif x < 0 <= y:
        phi = np.arctan(y / x) + np.pi
    elif x < 0 and y < 0:
        phi = np.arctan(y / x) - np.pi
    elif x == 0 and y > 0:
        phi = np.pi/2
    elif x == 0 and y < 0:
        phi = -np.pi/2
    else:
        phi = 0
    return r, theta, phi


def coef_a_b(l):
    global Eps_1, Eps_2, K, R, n1, n2
    matrix = Matrix((
        ((Eps_2 / Eps_1) ** 0.25 * jv(l + 0.5, Eps_1 * K * R),
         (Eps_2 / Eps_1) ** 0.25 * yve(l + 0.5, Eps_1 * K * R),
         jv(l + 0.5, Eps_2 * K * R)),
        ((n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * jvp(l + 0.5, Eps_1 * K * R),
         (n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * yvp(l + 0.5, Eps_1 * K * R),
         jv(l + 0.5, Eps_2 * K * R))
    ))
    ab = solve_linear_system(matrix, x, y)
    return ab.values()


def coef_a_b2(l):
    a = (Eps_2 / Eps_1) ** 0.25 * jv(l + 0.5, Eps_1 * K * R)
    b = (Eps_2 / Eps_1) ** 0.25 * yve(l + 0.5, Eps_1 * K * R)
    c = jv(l + 0.5, Eps_2 * K * R)
    ap = (n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * jvp(l + 0.5, Eps_1 * K * R)
    bp = (n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * yvp(l + 0.5, Eps_1 * K * R)
    cp = jvp(l + 0.5, Eps_2 * K * R)
    A = (c * bp - b * cp) / (bp * a - ap * b)
    B = (a * cp - ap * c) / (bp * a - ap * b)
    return A, B


def val_field(x, y, z):
    global R, K, Eps_1, Eps_2, A_kml, B_kml
    L = range(2)
    val = 0
    r, theta, phi = system_cords_sphere(x, y, z)
    r = r * 10e-9
    if r == 0:
        return 0
    if r > R:
        for l in L:
            for m in range(-l, l + 1):
                val += (1 / ((K * r) ** 0.5 * Eps_2 ** 0.25)) * jv(l + 0.5, Eps_2 * K * r) * sph_harm(m, l, phi,
                                                                                                      theta)
    else:
        for l in L:
            for m in range(-l, l + 1):
                val += (1 / ((K * r) ** 0.5 * Eps_1 ** 0.25)) * (
                        A_kml * jv(l + 0.5, Eps_1 * K * r) + B_kml * yve(l + 0.5, Eps_1 * K * r)) * sph_harm(m, l, phi,
                                                                                                     theta)
    return np.complex128(val).real, np.complex128(val).imag


if __name__ == '__main__':
    num_points = 200
    X, Y, Z = np.mgrid[-50:50:num_points*1j, -50:50:num_points*1j, -50:50:num_points*1j]
    A_kml, B_kml = np.array([coef_a_b2(l) for l in range(3)]).T
    R = 10e-9
    Eps_2 = 1
    Eps_1 = 0.004
    n1 = np.sqrt(Eps_1)
    n2 = 1
    K = 2 * np.pi / 260*1e-9
    FIELD = np.zeros([num_points]*3)

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=FIELD,
        isomin=0.1,
        isomax=0.8,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=17,  # needs to be a large number for good volume rendering
    ))
    fig.show()

