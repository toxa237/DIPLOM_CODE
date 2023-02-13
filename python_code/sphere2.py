import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, lpmv, sph_harm
import sympy as sm
import tqdm
from sympy import Matrix, solve_linear_system
from sympy.abc import x as X, y as Y
import joblib


def cords2sphere_cords(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if z != 0:
        theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z) + np.pi / 2
    else:
        theta = np.pi / 2
    if x > 0:
        phi = np.arctan(y / x)
    elif x < 0 <= y:
        phi = np.arctan(y / x) + np.pi
    elif x < 0 and y < 0:
        phi = np.arctan(y / x) - np.pi
    elif x == 0 and y > 0:
        phi = np.pi / 2
    elif x == 0 and y < 0:
        phi = -np.pi / 2
    else:
        phi = 0
    return rho, theta, phi


def coef_a_b(l):
    global Eps_1, Eps_2, K, R, n1, n2
    r = R * STEP
    a = spherical_jn(l, n2 * K * r)
    b = spherical_jn(l, n1 * K * r)
    c = spherical_yn(l, n1 * K * r)
    ap = spherical_jn(l, n2 * K * r, derivative=True)
    bp = spherical_jn(l, n1 * K * r, derivative=True)
    cp = spherical_yn(l, n1 * K * r, derivative=True)
    matrix = Matrix((
        (b, c, a),
        (bp, cp, ap)
    ))
    a, b = solve_linear_system(matrix, X, Y).values()
    return a, b


def coef_a_b2(l):
    global K, R, n1, n2
    r = R * STEP
    a = spherical_jn(l, n2 * K * r)
    b = spherical_jn(l, n1 * K * r)
    c = spherical_yn(l, n1 * K * r)
    ap = spherical_jn(l, n2 * K * r, derivative=True)
    bp = spherical_jn(l, n1 * K * r, derivative=True)
    cp = spherical_yn(l, n1 * K * r, derivative=True)
    A = (a * cp + c * ap) / (b * cp - bp * c)
    B = (a * bp - ap * b) / (bp * c - b * cp)
    return A, B


def val_field(x, y, z):
    global R, K, Eps_1, Eps_2, A_kml, B_kml, L
    val = 0
    r, theta, phi = cords2sphere_cords(x, y, z)
    r = r * STEP
    if r == 0:
        return 0
    # lpmv(m, l, np.cos(theta)) * sph_harm(m, l, theta, phi) *
    if r < R * STEP:
        for i, l in enumerate(L):
            for m in range(-l, l + 1):
                val += lpmv(m, l, np.cos(theta)) * sph_harm(m, l, theta, phi) *\
                           (A_kml[i] * spherical_jn(l, K * r * n1) + B_kml[i] * spherical_yn(l, K * r * n1))
                # lpmv(m, l, np.cos(theta)) * sph_harm(m, l, theta, phi) *\
                # (A_kml[i] * spherical_jn(l, K * r * n1) + B_kml[i] * spherical_yn(l, K * r * n1))
    else:
        for l in L:
            for m in range(-l, l + 1):
                val += lpmv(m, l, np.cos(theta)) * sph_harm(m, l, theta, phi) *\
                           (spherical_jn(l, K * r * n2))  # + spherical_yn(l, K*r*n2)
    return np.complex128(val).real


if __name__ == '__main__':
    # coon = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)
    num_of_points = 200
    xy = np.linspace(-1000, 1000, num=num_of_points)
    R = 40
    STEP = 10e-9
    Eps_2 = 1
    Eps_1 = 1.6
    n2 = 1
    n1 = np.sqrt(Eps_1)
    K = 2 * np.pi / (400 * 10e-9)
    L = [1]
    
    A_kml, B_kml = np.array([coef_a_b2(l) for l in L]).T
    print(A_kml, B_kml, sep='\n')
    # A_kml, B_kml = np.array([coef_a_b(l) for l in L]).T
    # print(A_kml, B_kml, sep='\n')
    # exit()

    FIELD = []
    for q, i in enumerate(tqdm.tqdm(xy)):
        FIELD.append([])
        for j in xy:
            FIELD[q].append(val_field(i, 0, j))
    FIELD = np.array(FIELD)

    print(np.max(FIELD), np.min(FIELD))

    # x1, y1 = np.meshgrid(xy, xy)
    # plt.contourf(x1, y1, FIELD)
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
