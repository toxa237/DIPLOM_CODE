import numpy as np
from scipy.special import jv as besselj
from scipy.special import hankel2 as besselh
import matplotlib.pyplot as plt
from relative_permittivity import RelativePermittivity
from scipy.integrate import quad


class CrossSection:
    def __init__(self, r, eps, L=0, eps0=1):
        self.R = r
        self.L = L
        self.LAMDA = np.arange(200, 1200, 1)
        self.N = RelativePermittivity(eps, self.LAMDA).Name

    def calc_cross_section(self):
        SCS = []
        for i in range(self.LAMDA.shape[0]):
            ka = 2 * np.pi * self.R[1] / self.LAMDA[i]
            n = self.N[0, i]
            n1 = self.N[1, i]
            a_m, b_m, c_m, d_m = self.get_coefficients_1_layer(ka, n, n1, self.R[0]/self.R[1])
            res = 0
            for m in range(-self.L, self.L + 1):
                res = res + np.abs(d_m[m + self.L]) ** 2
            SCS.append(4 * res / ka)
        return SCS

    def get_coefficients_1_layer(self, ka, n, n1, r1):
        a_m = []
        b_m = []
        c_m = []
        d_m = []

        for m in range(-self.L, self.L + 1):
            M11 = besselj(m, n1 * ka * r1)
            M12 = -besselj(m, n * ka * r1)
            M13 = -besselh(m, n * ka * r1)
            M14 = 0

            M21 = n * self.besselj_prime(m, n1 * ka * r1)
            M22 = -n1 * self.besselj_prime(m, n * ka * r1)
            M23 = -n1 * self.besselh_prime(m, n * ka * r1)
            M24 = 0

            M31 = 0
            M32 = besselj(m, n * ka)
            M33 = besselh(m, n * ka)
            M34 = -besselh(m, ka)

            M41 = 0
            M42 = self.besselj_prime(m, n * ka)
            M43 = self.besselh_prime(m, n * ka)
            M44 = -n * self.besselh_prime(m, ka)

            M = np.array([[M11, M12, M13, M14], [M21, M22, M23, M24], [M31, M32, M33, M34], [M41, M42, M43, M44]])
            # right size
            rs_1 = 0
            rs_2 = 0
            rs_3 = (-1.j) ** m * besselj(m, ka)
            rs_4 = (-1.j) ** m * n * self.besselj_prime(m, ka)

            Rs = np.array([rs_1, rs_2, rs_3, rs_4])

            solution = np.linalg.solve(M, Rs.T)
            a_m.append(solution[0])
            b_m.append(solution[1])
            c_m.append(solution[2])
            d_m.append(solution[3])

        return a_m, b_m, c_m, d_m

    @staticmethod
    def besselh_prime(m, z):
        f = m / z * besselh(m, z) - besselh(m + 1, z)
        return f

    @staticmethod
    def besselj_prime(m, z):
        f = m / z * besselj(m, z) - besselj(m + 1, z)
        return f


if __name__ == "__main__":
    R = np.array([20, 30])
    EPS = ["Au", "Ag"]
    a = CrossSection(R, EPS, L=15)

    cs = a.calc_cross_section()

    plt.plot(a.LAMDA, cs)
    plt.show()

