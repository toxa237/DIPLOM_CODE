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
    return r * STEP, theta, phi


def coef_a_b(l):
    global Eps_1, Eps_2, K, R, n1, n2
    r = R * STEP
    matrix = Matrix((
        ((Eps_2 / Eps_1) ** 0.25 * jv(l + 0.5, Eps_1 * K * r),
         (Eps_2 / Eps_1) ** 0.25 * yve(l + 0.5, Eps_1 * K * r),
         jv(l + 0.5, Eps_2 * K * r)),
        ((n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * jvp(l + 0.5, Eps_1 * K * r),
         (n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * yvp(l + 0.5, Eps_1 * K * r),
         jvp(l + 0.5, Eps_2 * K * r))
    ))
    a, b = solve_linear_system(matrix, X, Y).values()
    return a, b


def coef_a_b2(l):
    global Eps_1, Eps_2, K, R, n1, n2, A_kml1
    r = R*STEP
    a = (Eps_2 / Eps_1) ** 0.25 * jv(l + 0.5, Eps_1 * K * r)
    b = (Eps_2 / Eps_1) ** 0.25 * yve(l + 0.5, Eps_1 * K * r)
    c = jv(l + 0.5, Eps_2 * K * r)
    ap = (n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * jvp(l + 0.5, Eps_1 * K * r)
    bp = (n1 / n2) * (Eps_2 / Eps_1) ** 0.25 * yvp(l + 0.5, Eps_1 * K * r)
    cp = jvp(l + 0.5, Eps_2 * K * r)
    A = ((c * bp - b * cp) / (bp * a - ap * b))/A_kml1
    B = ((a * cp - ap * c) / (bp * a - ap * b))/A_kml1
    return A, B


def val_field(x, z):
    global R, K, Eps_1, Eps_2, A_kml1, A_kml2, B_kml2, L
    val = 0
    r, theta, phi = system_cords_sphere(x, z)
    if r == 0:
        return 0
    if r > R:
        for i, l in enumerate(L):
            for m in range(-l, l + 1):
                val += (1 / (((K * r) ** 0.5) * (Eps_2 ** 0.25))) * A_kml1 * jv(l + 0.5, Eps_2 * K * r) * \
                       sph_harm(m, l, phi, theta)
    else:
        for i, l in enumerate(L):
            for m in range(-l, l + 1):
                val += (1 / ((K * r) ** 0.5 * Eps_1 ** 0.25)) * \
                       (A_kml2[i] * jv(l + 0.5, Eps_1 * K * r) + B_kml2[i] * yve(l + 0.5, Eps_1 * K * r)) * \
                       sph_harm(m, l, phi, theta)
    return np.complex128(val).real


if __name__ == '__main__':
    # coon = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)
    num_of_points = 400
    xy = np.linspace(-100, 100, num=num_of_points)
    R = 10
    STEP = 10e-3
    Eps_2 = 1
    Eps_1 = 0.004
    n2 = 1
    n1 = np.sqrt(Eps_1)
    K = 2 * np.pi / (400 * 10e-9)
    L = range(3)
    A_kml1 = 1

    A_kml2, B_kml2 = np.array([coef_a_b2(l) for l in L]).T
    # A_kml, B_kml = np.array([coef_a_b(l) for l in range(3)]).T
    # print(A_kml, A_kml2, B_kml, B_kml2, sep='\n')
    # exit()

    FIELD = []
    for q, i in enumerate(tqdm.tqdm(xy)):
        FIELD.append([])
        for j in xy:
            FIELD[q].append(val_field(i, j))
    FIELD = np.array(FIELD)

    print(np.max(FIELD), np.min(FIELD))

    # x1, y1 = np.meshgrid(xy, xy)
    # plt.contourf(x1, y1, FIELD, cmap='hsv')
    # plt.show()
    # exit()

    fig = go.Figure(data=go.Contour(
        z=FIELD, x=xy, y=xy
    ))

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=-R, y0=-R, x1=R, y1=R,
                  line_color="LightSeaGreen",
                  )
    fig.show()
