import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, lpmv, sph_harm
from cross_section_sphere import CrossSection


class FildScattering(CrossSection):
    def __init__(self, r, eps, L, lamb_index, x_min=-600, x_max=600, size: int = 300):
        super().__init__(r, eps, L)
        self.Fild = np.zeros((size, size), dtype=np.complex128)
        self.xyz = np.linspace(x_min, x_max, size) * 10e-9
        self.A, self.B = self.calc_coef(lamb_index)
        # print(self.A, '\n', self.B)
        self.K = 2 * np.pi / (self.LAMDA[lamb_index] * 10e-9)
        self.LAM_IND = lamb_index

    def calc_field(self, plane: str = "xy", else_cords=0):
        else_cords = else_cords * 10e-9
        match plane:
            case "xy":
                for i, x in enumerate(self.xyz):
                    print(f'\r{i + 1}/{self.xyz.shape[0]}', sep=' ', end='')
                    for j, y in enumerate(self.xyz):
                        self.Fild[i, j] = self.calc_field_in_point(x, y, else_cords)
            case "xz":
                for i, x in enumerate(self.xyz):
                    print(f'\r{i + 1}/{self.xyz.shape[0]}', sep=' ', end='')
                    for j, z in enumerate(self.xyz):
                        self.Fild[i, j] = self.calc_field_in_point(x, else_cords, z)
            case "yz":
                for i, y in enumerate(self.xyz):
                    print(f'\r{i + 1}/{self.xyz.shape[0]}', sep=' ', end='')
                    for j, z in enumerate(self.xyz):
                        self.Fild[i, j] = self.calc_field_in_point(else_cords, y, z)

    def calc_field_in_point(self, x, y, z):
        rho, theta, phi = self.cords2sphere_cords(x, y, z)
        if rho < self.R[0]:
            layer_n = 0
        elif rho > self.R[-1]:
            layer_n = -1
        else:
            layer_n = np.searchsorted(self.R, rho, side='right')

        var = 0
        for i, l in enumerate(self.L):
            for m in range(-l, l + 1):
                var += (self.A[layer_n, i] * spherical_jn(l, self.K * rho * self.N[layer_n, self.LAM_IND]) +
                        self.B[layer_n, i] * spherical_yn(l, self.K * rho * self.N[layer_n, self.LAM_IND])) * \
                        sph_harm(m, l, theta, phi)
                if layer_n == -1:
                    var += (1j ** l) * (2 * l + 1) * spherical_jn(l, self.K * rho) * lpmv(0, l, np.cos(theta))
        return var

    @staticmethod
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

    def plot_field(self):
        x1, y1 = np.meshgrid(self.xyz, self.xyz)
        plt.contourf(x1, y1, np.abs(self.Fild))
        for i in self.R:
            circle = plt.Circle((0, 0), i, color='white', fill=False)
            plt.gca().add_patch(circle)
        plt.colorbar()


if __name__ == "__main__":
    R = np.array([100]) * 10e-9
    EPS = ["Ag"]
    a = FildScattering(R, EPS, L=10, lamb_index=300, x_min=-2000, x_max=2000, size=250)

    a.calc_field(plane="xz", else_cords=1)
    a.plot_field()

    plt.show()
