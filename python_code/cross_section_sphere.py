import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
from relative_permittivity import RelativePermittivity
from scipy.integrate import quad


class CrossSection:
    def __init__(self, r, eps, L=0, eps0=1):
        self.R = np.array(r)
        self.L = np.arange(0, L+1, 1)
        self.Cross_section = None
        self.LAMDA = np.arange(300, 1202, 2)
        self.N = np.real(RelativePermittivity(eps + [eps0], self.LAMDA).Name)
        self.N = self.N ** (1/2)

    def calc_coef(self, lamb_ind, A0=1):
        A = np.zeros((self.R.shape[0] + 1, self.L.shape[0]), dtype=np.complex128)
        B = np.zeros((self.R.shape[0] + 1, self.L.shape[0]), dtype=np.complex128)
        A[0, :] = A0
        B[0, :] = 0
        for l in self.L:
            coef_x, coef_y = self.calc_matrix_for_coef(l, lamb_ind)
            # coef = np.linalg.solve(coef_x, coef_y)
            coef = self.cramer_rule(coef_x, coef_y)
            A[1:, l] = coef[::2]
            B[1:, l] = coef[1::2]
        return A, B

    def A0_coef(self, a, b, l, lamb_ind):
        k = 2 * np.pi / (self.LAMDA[lamb_ind] * 10e-9)

        E_r = quad(lambda x: spherical_jn(l, self.N[0, lamb_ind].real * k * x), 0, self.R[0])[0]
        if self.R.shape[0] > 1:
            for i in range(1, self.R.shape[0]):
                E_r += quad(lambda x: a[i].real * spherical_jn(l, self.N[i, lamb_ind].real * k * x) + b[i].real *
                                      spherical_yn(l, self.N[i, lamb_ind].real * k * x), self.R[i - 1], self.R[i])[0]

        E_r += quad(lambda x: a[-1].real * spherical_jn(l, self.N[0, lamb_ind].real * k * x) + b[-1].real *
                              spherical_yn(l, self.N[0, lamb_ind].real * k * x), self.R[-1], np.inf)[0]

        E_i = 0
        if np.imag(self.N).any():
            E_i = quad(lambda x: spherical_jn(l, self.N[0, lamb_ind].imag * k * x), 0, self.R[0])[0]

            if self.R.shape[0] > 1:
                for i in range(1, self.R.shape[0]):
                    E_i += quad(lambda x: a[i].imag * spherical_jn(l, self.N[i, lamb_ind].imag * k * x) + b[i].imag *
                                          spherical_yn(l, self.N[i, lamb_ind].imag * k * x), self.R[i - 1],
                                self.R[i])[0]

            E_i += quad(lambda x: a[-1].imag * spherical_jn(l, self.N[0, lamb_ind].imag * k * x) + b[-1].imag *
                                  spherical_yn(l, self.N[0, lamb_ind].imag * k * x), self.R[-1], np.inf)[0]

        E = E_r + 1j * E_i
        return a * E, b * E

    def calc_matrix_for_coef(self, l, lamb_ind):
        a = np.zeros((2 * (self.R.shape[0]), 2 * (self.R.shape[0])), dtype=np.complex128)
        b = np.zeros(2 * self.R.shape[0], dtype=np.complex128)
        k = 2 * np.pi / (self.LAMDA[lamb_ind] * 10e-9)

        b[0] = spherical_jn(l, self.N[0, lamb_ind] * k * self.R[0])
        b[1] = self.N[0, lamb_ind] * self.derivative_spherical_jn(l, self.N[0, lamb_ind] * k, self.R[0])

        b[-2] += (1j ** l) * (2*l + 1) * spherical_jn(l, k * self.R[-1])
        b[-1] += (1j ** l) * (2*l + 1) * self.derivative_spherical_jn(l, k, self.R[-1])

        a[0, 0] = spherical_jn(l, self.N[1, lamb_ind] * k * self.R[0])
        a[0, 1] = spherical_yn(l, self.N[1, lamb_ind] * k * self.R[0])

        a[1, 0] = self.N[1, lamb_ind] * self.derivative_spherical_jn(l, self.N[1, lamb_ind] * k, self.R[0])
        a[1, 1] = self.N[1, lamb_ind] * self.derivative_spherical_yn(l, self.N[1, lamb_ind] * k, self.R[0])

        if self.R.shape[0] >= 2:
            c = 0
            layer = 1
            for i in range(2, 2 * self.R.shape[0], 2):
                a[i, c] = spherical_jn(l, self.N[layer, lamb_ind] * k * self.R[layer])
                a[i, c + 1] = spherical_yn(l, self.N[layer, lamb_ind] * k * self.R[layer])
                a[i, c + 2] = -spherical_jn(l, self.N[layer + 1, lamb_ind] * k * self.R[layer])
                a[i, c + 3] = -spherical_yn(l, self.N[layer + 1, lamb_ind] * k * self.R[layer])

                a[i + 1, c] = self.N[layer, lamb_ind] * self.derivative_spherical_jn(l, self.N[layer, lamb_ind] * k, self.R[layer])
                a[i + 1, c + 1] = self.N[layer, lamb_ind] * self.derivative_spherical_yn(l, self.N[layer, lamb_ind] * k, self.R[layer])
                a[i + 1, c + 2] = -self.N[layer + 1, lamb_ind] * self.derivative_spherical_jn(l, self.N[layer + 1, lamb_ind] * k, self.R[layer])
                a[i + 1, c + 3] = -self.N[layer + 1, lamb_ind] * self.derivative_spherical_yn(l, self.N[layer + 1, lamb_ind] * k, self.R[layer])
                c += 2
                layer += 1

        return a, b

    @staticmethod
    def derivative_spherical_jn(l, c, r):
        # a = l * spherical_jn(l, c * r) / r - c * spherical_jn(l + 1, c * r)
        a = c * spherical_jn(l, c * r, derivative=True)
        return a

    @staticmethod
    def derivative_spherical_yn(l, c, r):
        # a = l * spherical_yn(l, c * r) / r - c * spherical_yn(l + 1, c * r)
        a = c * spherical_yn(l, c * r, derivative=True)
        return a

    @staticmethod
    def cramer_rule(matrix_a, matrix_b):
        matrix_a = np.array(matrix_a)
        matrix_b = np.array(matrix_b)
        det = np.linalg.det(matrix_a)
        x = np.zeros(matrix_b.shape[0], dtype=np.complex128)
        if det == 0:
            return np.zeros(matrix_b.shape[0])
        for i in range(matrix_a.shape[1]):
            m = matrix_a.copy()
            m[:, i] = matrix_b
            x[i] = np.linalg.det(m) / det
        return x

    @staticmethod
    def coef_a_b(r, k, l, A0, n0, n1):
        A = r * k * A0 * (r * k * n0 * n0 * spherical_jn(l + 1, r * k * n0) * spherical_yn(l, r * k * n1) -
                          spherical_jn(l, r * k * n0) * (l * (n0 - n1) * spherical_yn(l, r * k * n1) +
                                                         r * k * n1 * n1 * spherical_yn(l + 1, r * k * n1)))
        B = r * k * A0 * (-r * k * n0 * n0 * spherical_jn(l, r * k * n1) * spherical_jn(l + 1, r * k * n0) +
                          spherical_jn(l, r * k * n0) * (l * (n0 - n1) * spherical_jn(l, r * k * n1) +
                                                         r * k * n1 * n1 * spherical_jn(l + 1, r * k * n1)))
        return A, B

    def calc_cross_section(self):
        cs = []
        for i in range(self.LAMDA.shape[0]):
            a, b = self.calc_coef(i)
            a, b = a[-1], b[-1]
            cs.append(sum([np.abs(a[n]) ** 2 + np.abs(b[n]) ** 2 for n in self.L]))
        self.Cross_section = np.array(cs)
        return self.Cross_section

    def plot_cross_section(self, label=""):
        plt.plot(self.LAMDA, self.Cross_section/np.max(self.Cross_section), label=label)
        plt.xlabel("Довжина падаючої хвилі, нм", fontsize=20)  # Wavelength
        plt.ylabel("SCA", fontsize=20)
        if label:
            plt.legend()
        # plt.show()


if __name__ == "__main__":
    R = np.array([100, 150]) * 10e-9
    EPS = ["Au", "Cu"]
    a = CrossSection(R, EPS, L=0)
    a.calc_cross_section()
    a.plot_cross_section()
    plt.show()
