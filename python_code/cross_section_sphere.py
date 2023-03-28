import numpy as np
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
from relative_permittivity import RelativePermittivity
from scipy.integrate import quad


class CrossSection:
    def __init__(self, r, eps, L=5):
        self.R = np.array(r)
        self.L = np.arange(0, L + 1, 1)
        self.Cross_section = None
        self.LAMDA = np.arange(400, 1000, 1) * 10e-9
        self.N = np.append(RelativePermittivity(eps, self.LAMDA).Name, np.ones((1, self.LAMDA.shape[0])), axis=0)
        self.N = np.sqrt(self.N)

    def calc_coef(self, lamb, A0=1):
        A = np.zeros((self.R.shape[0] + 1, self.L.shape[0]), dtype=np.complex128)
        B = np.zeros((self.R.shape[0] + 1, self.L.shape[0]), dtype=np.complex128)
        A[0, :] = A0
        B[0, :] = 0
        for l in self.L:
            coef = self.calc_matrix_for_coef(l, lamb)
            # print(l, coef[0], coef[1], self.cramer_rule(coef[0], coef[1]), sep='\n', end='\n======\n')
            coef2 = np.linalg.solve(coef[0], coef[1])
            a, b, a0 = self.A0_coef(coef2[::2], coef2[1::2], l, lamb)
            A[0, l] = a0
            A[1:, l] = a  # coef2[::2]
            B[1:, l] = b  # coef2[1::2]
        return A, B
    
    def A0_coef(self, a, b, l, lamb):
        numb_k = np.where(self.LAMDA == lamb)[0][0]
        k = 2 * np.pi / lamb

        E_r = quad(lambda x: (spherical_jn(l, self.N[0, numb_k].real * k * x))**2, 0, self.R[0])[0]
        if self.R.shape[0] > 1:
            for i in range(1, self.R.shape[0]):
                E_r += quad(lambda x: (a[i].real * spherical_jn(l, self.N[i, numb_k].real * k * x) + b[i].real *
                            spherical_yn(l, self.N[i, numb_k].real * k * x))**2, self.R[i-1], self.R[i])[0]

        E_r += quad(lambda x: (a[-1].real * spherical_jn(l, self.N[0, numb_k].real * k * x) + b[-1].real *
                    spherical_yn(l, self.N[0, numb_k].real * k * x))**2, self.R[-1], np.inf)[0]

        E_i = 0
        if np.imag(self.N).any():
            E_i = quad(lambda x: (spherical_jn(l, self.N[0, numb_k].imag * k * x))**2, 0, self.R[0])[0]

            if self.R.shape[0] > 1:
                for i in range(1, self.R.shape[0]):
                    E_i += quad(lambda x: (a[i].imag * spherical_jn(self.N[i, numb_k].imag * k * x, l) + b[i].imag *
                                spherical_yn(l, self.N[i, numb_k].imag * k * x))**2, self.R[i-1], self.R[i])[0]

            E_i += quad(lambda x: (a[-1].imag * spherical_jn(l, self.N[0, numb_k].imag * k * x) + b[-1].imag *
                        spherical_yn(l, self.N[0, numb_k].imag * k * x))**2, self.R[-1], np.inf)[0]

        E = E_r + 1j*E_i
        return a*E, b*E, 1/E

    def calc_matrix_for_coef(self, l, lamb):
        a = np.zeros((2 * (self.R.shape[0]), 2 * (self.R.shape[0])), dtype=np.complex128)
        b = np.zeros(2 * self.R.shape[0], dtype=np.complex128)
        k = 2 * np.pi / lamb
        numb_k = np.where(self.LAMDA == lamb)[0][0]

        b[0] = spherical_jn(l, self.N[0, numb_k] * k * self.R[0])
        b[1] = (self.N[1, numb_k] / self.N[0, numb_k]) * self.derivative_spherical_jn(l, self.N[0, numb_k] * k,
                                                                                      self.R[0])

        a[0, 0] = spherical_jn(l, self.N[1, numb_k] * k * self.R[0])
        a[0, 1] = spherical_yn(l, self.N[1, numb_k] * k * self.R[0])

        a[1, 0] = self.derivative_spherical_jn(l, self.N[1, numb_k] * k, self.R[0])
        a[1, 1] = self.derivative_spherical_yn(l, self.N[1, numb_k] * k, self.R[0])

        if self.R.shape[0] >= 2:
            c = 0
            layer = 1
            for i in range(2, 2 * self.R.shape[0], 2):
                a[i, c] = spherical_jn(l, self.N[layer, numb_k] * k * self.R[layer])
                a[i, c + 1] = spherical_yn(l, self.N[layer, numb_k] * k * self.R[layer])
                a[i, c + 2] = -spherical_jn(l, self.N[layer + 1, numb_k] * k * self.R[layer])
                a[i, c + 3] = -spherical_yn(l, self.N[layer + 1, numb_k] * k * self.R[layer])

                a[i + 1, c] = (self.N[layer, numb_k] / self.N[layer + 1, numb_k]) * \
                              self.derivative_spherical_jn(l, self.N[layer, numb_k] * k, self.R[layer])
                a[i + 1, c + 1] = (self.N[layer, numb_k] / self.N[layer + 1, numb_k]) * \
                                  self.derivative_spherical_yn(l, self.N[layer, numb_k] * k, self.R[layer])
                a[i + 1, c + 2] = -self.derivative_spherical_jn(l, self.N[layer + 1, numb_k] * k, self.R[layer])
                a[i + 1, c + 3] = -self.derivative_spherical_yn(l, self.N[layer + 1, numb_k] * k, self.R[layer])
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
        x = np.zeros(matrix_b.shape[0])
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
            a, b = self.calc_coef(self.LAMDA[i])
            a, b = a[-1], b[-1]
            cs.append(sum([np.abs(a[n]) ** (2) + np.abs(b[n]) ** (2) for n in self.L]))
            # cs.append((2/(k**2 * self.R[-1]**2)) * sum([(2*n + 1) * (a[n]**2 + b[n]**2) for n in self.L]))
        self.Cross_section = cs

    def plot_cross_section(self):
        plt.plot(self.LAMDA, self.Cross_section)
        # plt.show()


if __name__ == "__main__":
    R = np.array([100]) * 10e-9  # , 200, 250
    EPS = [2.3]  # , 0.4, 0.8
    a = CrossSection(R, EPS, L=5)

    # lam = a.LAMDA[100]
    # print(lam, a.calc_coef(lam))

    a.calc_cross_section()
    a.plot_cross_section()

    # plt.plot(np.arange(400, 1000, 1), a.Cross_section)
    # plt.xlabel("Довжина падаючої хвилі, нм")  # Wavelength
    # plt.ylabel("Переріз екстинкції, у.о.")

    plt.show()
