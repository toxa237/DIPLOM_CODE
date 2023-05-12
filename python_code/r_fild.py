import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.special import spherical_jn, spherical_yn, lpmv, sph_harm
import joblib
import tqdm


def cords2sphere_cords(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if z > 0:
        theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
    elif z < 0:
        theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z) + np.pi
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


def val_field(x, y, z, a, k):
    global L, A0, A1, B1, C
    rho, theta, phi = cords2sphere_cords(x, y, z)
    # print(rho, theta, phi)
    var = 0
    if rho < a:
        for l in L:
            for m in range(-l, l+1):
                var += A0 * spherical_jn(l, k * rho * n0) * sph_harm(m, l, phi, theta)
    elif rho > a:
        for i, l in enumerate(L):
            for m in range(-l, l+1):
                var += (A1[i] * spherical_jn(l, k * rho * n1) + B1[i] * spherical_yn(l, k * rho * n1)) *\
                       sph_harm(m, l, phi, theta) + (1j**l) * (2*l + 1) * spherical_jn(l, k*rho) * \
                       lpmv(m, l, np.cos(theta)) * np.exp(1j * l * phi)
            # var += (1j**l) * (2*l + 1) * spherical_jn(l, k*rho) * lpmv(0, l, np.cos(theta)) * np.exp(1j * l * phi)
    else:
        return 0
    return var


def coef_a_b(a, k, l):
    global A0, n0, n1
    A = a * k * A0 * (a * k * n0 * n0 * spherical_jn(l + 1, a * k * n0) * spherical_yn(l, a * k * n1) -
                      spherical_jn(l, a * k * n0) * (l * (n0 - n1) * spherical_yn(l, a * k * n1) +
                                                     a * k * n1 * n1 * spherical_yn(l + 1, a * k * n1)))
    B = a * k * A0 * (-a * k * n0 * n0 * spherical_jn(l, a * k * n1) * spherical_jn(l + 1, a * k * n0) +
                      spherical_jn(l, a * k * n0) * (l * (n0 - n1) * spherical_jn(l, a * k * n1) +
                                                     a * k * n1 * n1 * spherical_jn(l + 1, a * k * n1)))
    return A, B


if __name__ == "__main__":
    num_of_points = 200
    eps0 = 1
    eps1 = 2.6
    n0 = np.sqrt(eps0)
    n1 = np.sqrt(eps1)
    L = np.arange(0, 5, 1)
    A0 = 1

    # C = 3*10e8
    X = np.linspace(-100*10e-8, 100*10e-8, num_of_points)
    #
    a = 50 * 10e-9
    lam = 400 * 10e-9
    k = 2 * np.pi / lam
    A1, B1 = np.array([coef_a_b(a, k, l) for l in L]).T
    print(A1, B1)
    exit()

    # field = [val_field(x, 0, 0, a, k) for x in X]
    # plt.plot(X, field)
    # plt.plot([a, a], [0, 0.5])
    # plt.grid()
    # plt.show()
    # exit()

    FIELD = np.zeros((num_of_points, num_of_points), dtype=np.complex128)
    for i in tqdm.tqdm(range(num_of_points)):
        for j in range(num_of_points):
            FIELD[i, j] = val_field(X[i], 0, X[j], a, k)

    plt.figure()
    x1, y1 = np.meshgrid(X, X)
    plt.contourf(x1, y1, np.real(FIELD))
    circle = plt.Circle((0, 0), a, color='white', fill=False)
    plt.gca().add_patch(circle)
    plt.colorbar()

    # plt.figure()
    # x1, y1 = np.meshgrid(X, X)
    # plt.contourf(x1, y1, np.imag(FIELD))
    # circle = plt.Circle((0, 0), a, color='white', fill=False)
    # plt.gca().add_patch(circle)
    # plt.colorbar()
    #
    # plt.figure()
    # x1, y1 = np.meshgrid(X, X)
    # plt.contourf(x1, y1, np.abs(FIELD))
    # circle = plt.Circle((0, 0), a, color='white', fill=False)
    # plt.gca   ().add_patch(circle)
    # plt.colorbar()
    #
    plt.show()

# graf = []

# for k in np.linspace(20000, 200000, 200):
#     A1, B1 = np.array([coef_a_b(a, k, l) for l in L]).T
#     graf.append([val_field(x, 0, 0, a, k) for x in X])
#
# graf = np.array(graf)
# fig = go.Figure(data=[go.Surface(z=graf)])
# fig.show()
# plt.plot(X, graf[100, :])
# plt.figure()
# plt.plot(np.linspace(20000, 200000, 200), graf[:, 100])
# plt.show()
