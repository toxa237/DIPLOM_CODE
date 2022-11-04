import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, yve, yvp, sph_harm  # yve функції неймана
import tqdm
from sympy import Matrix, solve_linear_system
from sympy.abc import x as X, y as Y
from sqlalchemy import create_engine
import joblib


def system_cords_sphere(x, z):
    r = np.sqrt(x ** 2 + z ** 2)
    if z != 0:
        if x == 0 and z >= 0:
            theta = np.pi
        elif x == 0 and z < 0:
            theta = 0
        else:
            theta = np.arctan(x / z) + np.pi / 2
    else:
        theta = np.pi

    if x >= 0:
        phi = 0
    else:
        phi = np.pi
    return r, theta, phi


def coef_a_b(l):
    global Eps_1, Eps_2, K, R, n1, n2
    matrix = Matrix((
        ((Eps_2 / Eps_1) ** 0.25 * jv(l + 0.5, Eps_1 * K * R),
         (Eps_2 / Eps_1) ** 0.25 * yve(l + 0.5, Eps_1 * K * R),
         jv(l + 0.5, Eps_2 * K * R)),
        ((n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * jvp(l + 0.5, Eps_1 * K * R),
         (n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * yvp(l + 0.5, Eps_1 * K * R),
         jvp(l + 0.5, Eps_2 * K * R))
    ))
    a, b = solve_linear_system(matrix, X, Y).values()
    return float(a), float(b)


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


def val_field(x, z):
    global R, K, Eps_1, Eps_2, A_kml, B_kml, L
    val = 0
    r, theta, phi = system_cords_sphere(x, z)
    r = r * 1e-9
    if r == 0:
        return 0
    if r > R:
        for l in L:
            for m in range(-l, l + 1):
                val += (1 / (((K * r) ** 0.5) * (Eps_2 ** 0.25))) * jv(l + 0.5, Eps_2 * K * r) * \
                       sph_harm(m, l, phi, theta)
    else:
        for l in L:
            for m in range(-l, l + 1):
                val += (1 / ((K * r) ** 0.5 * Eps_1 ** 0.25)) * \
                       (A_kml[l] * jv(l + 0.5, Eps_1 * K * r) + B_kml[l] * yve(l + 0.5, Eps_1 * K * r)) * \
                       sph_harm(m, l, phi, theta)
    return np.complex128(val).real


if __name__ == '__main__':
    # coon = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)
    num_of_points = 400
    xy = np.linspace(-20, 20, num=num_of_points)
    R = 10e-6
    R2 = 10
    Eps_2 = 1
    Eps_1 = 0.004
    n2 = 1
    n1 = np.sqrt(Eps_1)
    K = 2 * np.pi / 800e-9
    L = range(2)

    A_kml, B_kml = np.array([coef_a_b2(l) for l in range(3)]).T
    # A_kml2, B_kml2 = np.array([coef_a_b(l) for l in range(3)]).T
    # print(A_kml, A_kml2, B_kml, B_kml2, sep='\n')
    # exit()

    FIELD = []
    for q, i in enumerate(tqdm.tqdm(xy)):
        FIELD.append([])
        for j in xy:
            FIELD[q].append(val_field(i, j))
    FIELD = np.array(FIELD)

    # x1, y1 = np.meshgrid(xy, xy)
    # plt.contourf(x1, y1, FIELD)
    # plt.show()

    fig = go.Figure(data=go.Contour(
            z=FIELD, x=xy, y=xy
        ))

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-R2, y0=-R2, x1=R2, y1=R2,
                  line_color="LightSeaGreen",
                  )
    fig.show()
