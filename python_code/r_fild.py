import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, lpmv, sph_harm


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


def val_field(rho):
    global a, L, A0, A1, B1
    var = 0
    if rho < a:
        for l in L:
            var += A0 * spherical_jn(l, k * rho * n0)
    elif rho > a:
        for l in L:
            var += A1[l-1] * spherical_jn(l, k * rho * n1) + B1[l-1] * spherical_yn(l, k * rho * n1)
    else:
        return 0
    return var


def coef_a_b(l):
    global a, k, A0, n0, n1
    A = a * k * A0 * (a * k * n0 * n0 * spherical_jn(l + 1, a * k * n0) * spherical_yn(l, a * k * n1) -
                      spherical_jn(l, a * k * n0) * (l * (n0 - n1) * spherical_yn(l, a * k * n1) +
                                                     a * k * n1 * n1 * spherical_yn(l + 1, a * k * n1)))
    B = a * k * A0 * (-a * k * n0 * n0 * spherical_jn(l, a * k * n1) * spherical_jn(l + 1, a * k * n0) +
                      spherical_jn(l, a * k * n0) * (l * (n0 - n1) * spherical_jn(l, a * k * n1) +
                                                     a * k * n1 * n1 * spherical_jn(l + 1, a * k * n1)))
    return A, B


if __name__ == "__main__":
    eps0 = 1
    eps1 = 1.6
    a = 60 * 10e-9
    lam = 400 * 10e-9
    n0 = np.sqrt(eps0)
    n1 = np.sqrt(eps1)
    k = 2 * np.pi / lam
    L = range(1, 11)
    A0 = 1
    A1, B1 = np.array([coef_a_b(l) for l in L]).T
    X = np.linspace(-5000*10e-9, 5000*10e-9, 600)
    fild = [val_field(x) for x in X]
    plt.plot(X, fild)
    plt.grid()
    plt.show()
