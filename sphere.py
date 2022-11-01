import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, yve, yvp, sph_harm  # yve функції неймана
import tqdm
from sympy import Matrix, solve_linear_system
from sympy.abc import x, y
from sqlalchemy import create_engine
import joblib


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


def system_cords(l, r, s):
    global coon
    name = 'cd' + str(-l) + str(r) + str(s)
    try:
        c_d = pd.read_sql(f"SELECT * FROM {name}", coon)[['x', 'y', 'z']].values
    except:
        d = np.linspace(l, r, s)
        c_d = []
        for i in d:
            for j in d:
                for k in d:
                    c_d.append([i, j, k])
        pd.DataFrame(c_d, columns=['x', 'y', 'z']).to_sql(name, coon)
    return np.array(c_d)


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


def val_field(x, y, z):
    global R, K, Eps_1, Eps_2, coon
    L = range(2)
    val = 0
    r, theta, phi = system_cords_sphere(x, y, z)
    r = r * 1e-9
    if r == 0:
        return 0
    if r > R:
        for l in L:
            for m in range(-l, l + 1):
                val += (1 / ((K * r) ** 0.5 * Eps_2 ** 0.25)) * jv(l + 0.5, Eps_2 * K * r) * sph_harm(m, l, phi,
                                                                                                      theta)
    else:
        for l in L:
            a, b = coef_a_b(l)
            for m in range(-l, l + 1):
                val += (1 / ((K * r) ** 0.5 * Eps_1 ** 0.25)) * (
                        a * jv(l + 0.5, Eps_1 * K * r) + b * yve(l + 0.5, Eps_1 * K * r)) * sph_harm(m, l, phi,
                                                                                                     theta)
    return np.complex128(val).real, np.complex128(val).imag


if __name__ == '__main__':
    tqdm.tqdm.pandas()

    coon = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)
    cords_descartes = system_cords(-15, 15, 101)
    R = 10e-9
    Eps_2 = 1
    Eps_1 = 0.004
    n1 = np.sqrt(Eps_1)
    n2 = 1
    K = 2 * np.pi / 800e-9
    FIELD = pd.DataFrame(cords_descartes, columns=['x', 'y', 'z'])
    FIELD[['FIELDRe', 'FIELDIm']] = FIELD[['x', 'y', 'z']].progress_apply(lambda x: val_field(x[0], x[1], x[2]), axis=1,
                                                                          result_type='expand')
    joblib.dump(FIELD, 'data/FIELD.pkl')
    FIELD.to_sql('FIELDcd1515101', coon, if_exists='replace')
