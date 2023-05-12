import numpy as np
from scipy.special import jn, hankel1
import matplotlib.pyplot as plt
from relative_permittivity import RelativePermittivity


class CrossSection:
    def __init__(self, r, eps, L=5):
        self.R = np.array(r)
        self.L = np.arange(0, L + 1, 1)
        self.Cross_section = None
        self.LAMDA = np.arange(200, 1200, 1)
        self.N = RelativePermittivity(eps + [1], self.LAMDA).Name
        self.N = self.N ** (1 / 2)

    def calc_coef_lamd_l(self, lamd_ind, l):
        k = 2 * np.pi * self.N[1, lamd_ind] / (self.LAMDA[lamd_ind] * 10e-9)
        k1 = 2 * np.pi * self.N[0, lamd_ind] / (self.LAMDA[lamd_ind] * 10e-9)

        rho = k * self.R[0]
        rho1 = k1 * self.R[0]

        n = self.N[1, lamd_ind]
        n1 = self.N[0, lamd_ind]

        a_n = (n1 ** 2 * (rho * self.besselj_prime(l, rho) + jn(l, rho)) * jn(l, rho1) - n ** 2 * (
                rho1 * self.besselj_prime(l, rho1) + jn(l, rho1)) * jn(l, rho)) / \
              (n1 ** 2 * (rho * self.besselh_prime(l, rho) + hankel1(l, rho)) * jn(l, rho1) - n ** 2 * (
                      rho1 * self.besselj_prime(l, rho1) + jn(l, rho1)) * hankel1(l, rho))

        b_n = ((rho * self.besselj_prime(l, rho) + jn(l, rho)) * jn(l, rho1) - (
                rho1 * self.besselj_prime(l, rho1) + jn(l, rho1)) * jn(l, rho)) / \
              ((rho * self.besselh_prime(l, rho) + hankel1(l, rho)) * jn(l, rho1) - (
                      rho1 * self.besselj_prime(l, rho1) + jn(l, rho1)) * hankel1(l, rho))

        return a_n, b_n

    @staticmethod
    def besselh_prime(m, z):
        f = m / z * hankel1(m, z) - hankel1(m + 1, z)
        return f

    @staticmethod
    def besselj_prime(m, z):
        f = m / z * jn(m, z) - jn(m + 1, z)
        return f

    def calc_cross_section(self):
        cs = []
        for i in range(self.LAMDA.shape[0]):
            a, b = np.array([self.calc_coef_lamd_l(i, l) for l in self.L]).T
            cs.append(sum([np.abs(a[n]) ** 2 + np.abs(b[n]) ** 2 for n in self.L[1:]]))
        self.Cross_section = np.array(cs)

    def plot_cross_section(self):
        plt.plot(self.LAMDA, self.Cross_section)
        # plt.show()


if __name__ == "__main__":
    R = np.array([20]) * 10e-9
    EPS = ["Au"]
    a = CrossSection(R, EPS, L=2)

    a.calc_cross_section()
    a.plot_cross_section()

    plt.show()


