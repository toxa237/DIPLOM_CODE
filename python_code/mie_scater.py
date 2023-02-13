import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, jn, yn
import joblib
import plotly.graph_objects as go


class MieFild:
    def __init__(self, r=50, lam=364, eps_1=0.04, size=100, step=10e-9, calc=False):
        self.R = r * step
        self.K = 2 * np.pi / lam
        self.N1 = np.sqrt(eps_1)
        self.N2 = 1
        self.XYZ = np.linspace(-200 * step, 200 * step, size)
        self.count_harmonics = 3
        self.E0 = 100
        self.a_, self.b_, self.c_, self.d_ = self.calc_cof()
        self.STEP = step
        self.SIZE = size
        if calc:
            self.FIELD = np.zeros([size] * 2)
            self.calc_field()
        else:
            self.FIELD = joblib.load('field/field.pkl')

    def calc_field(self):
        for i, x in enumerate(self.XYZ):
            print(
                f"\r x = {i + 1}/{self.SIZE}[{'=' * int((i + 1) * 50 / self.SIZE)}{'-' * (50 - int((i + 1) * 50 / self.SIZE))}]",
                sep='', end='')
            for j, y in enumerate(self.XYZ):
                # for k, z in enumerate(self.XYZ):
                self.FIELD[i, j] = self.calc_field_in_point(x, y, 0)
        try:
            joblib.dump(self.FIELD, 'field/field.pkl')
        except:
            pass

    def calc_cof(self):
        c_one = 1 + 0j
        c_zero = 0.0 + 0.0j

        a = np.zeros((2, self.count_harmonics)) + c_zero
        b = np.zeros((2, self.count_harmonics)) + c_zero
        c = np.zeros((2, self.count_harmonics)) + c_zero
        d = np.zeros((2, self.count_harmonics)) + c_zero

        c[1, :] = c_one
        d[1, :] = c_one

        for n in range(self.count_harmonics):
            z = self.N1 * self.K * self.R
            z1 = self.N2 * self.K * self.R

            u = self.D1(n, z) - self.D3(n, z)
            t1 = a[1, n] * spherical_yn(n + 1, z1) - d[1, n] * spherical_jn(n + 1, z1)
            t2 = b[1, n] * spherical_yn(n + 1, z1) - c[1, n] * spherical_jn(n + 1, z1)
            t3 = d[1, n] * spherical_jn(n + 1, z1) * self.D1(n + 1, z1) - a[1, n] * spherical_yn(n + 1, z1) * \
                self.D3(n + 1, z1)
            t4 = c[1, n] * spherical_jn(n + 1, z1) * self.D1(n + 1, z1) - b[1, n] * spherical_yn(n + 1, z1) * \
                self.D3(n + 1, z1)

            a[1, n] = (self.D1(n + 1, z) * t1 + t3) / (spherical_yn(n + 1, z) * u)
            b[1, n] = (self.D1(n + 1, z) * t2 + t4) / (spherical_yn(n + 1, z) * u)
            c[0, n] = (self.D3(n + 1, z) * t2 + t4) / (spherical_jn(n + 1, z) * u)
            d[0, n] = (self.D3(n + 1, z) * t1 + t3) / (spherical_jn(n + 1, z) * u)
        print(a, b, c, d, sep='\n')
        return a, b, c, d

    def calc_field_in_point(self, x, y, z):
        rho, theta, phi = self.cords2sphere_cords(x, y, z)
        E_n = [self.E_n(n) for n in range(1, self.count_harmonics + 1)]
        if rho >= self.R:
            rho = rho
            M_oln = self.M_OLN(rho, theta, phi)
            M_oln = self.M_OLN(rho, theta, phi)
            N1_eln = self.N1_ELN(rho, theta, phi)
            N3_eln = self.N3_ELN(rho, theta, phi)
            Ei_in_point = np.sum([E_n[n] * (-1j * N1_eln[n] + M_oln[n]) for n in range(self.count_harmonics)], axis=0)
            Es_in_point = np.sum([E_n[n] * (1j * self.c_[0, n] * N3_eln[n] - self.d_[0, n] * M_oln[n])
                                  for n in range(self.count_harmonics)], axis=0)
            # print(1)
            M_eln = self.M_ELN(rho, theta, phi)
            N1_oln = self.N1_OLN(rho, theta, phi)
            N3_oln = self.N3_OLN(rho, theta, phi)
            Hi_in_point = np.sum([E_n[n] * (M_eln[n] + 1j * N1_oln[n]) for n in range(self.count_harmonics)], axis=0)
            Hs_in_point = np.sum([E_n[n] * (-1j * self.b_[0, n] * N3_oln[n] - self.a_[0, n] * M_eln[n])
                                  for n in range(self.count_harmonics)], axis=0)

            E_in_point = np.real(Es_in_point + Es_in_point)  # Ei_in_point + Es_in_point
            E_in_point[0] = E_in_point[0] * np.cos(phi) * np.sin(theta)
            E_in_point[1] = E_in_point[1] * np.sin(phi) * np.cos(theta)
            E_in_point[2] = E_in_point[2] * np.cos(phi)

            H_in_point = np.real(Hi_in_point + Hs_in_point)
            H_in_point[0] = H_in_point[0] * np.cos(phi) * np.sin(theta)
            H_in_point[1] = H_in_point[1] * np.sin(phi) * np.cos(theta)
            H_in_point[2] = H_in_point[2] * np.cos(phi)

            #F = np.cross(E_in_point, H_in_point)
            return np.sqrt(np.sum([i ** 2 for i in E_in_point]))
        else:
            return 0

    def E_n(self, n):
        return (1j ** n) * self.E0 * (2 * n + 1) / (n * (n + 1))

    def M_OLN(self, rho, theta, phi):
        M_oln = np.zeros((self.count_harmonics, 3))
        for n in range(1, self.count_harmonics + 1):
            M_oln[n - 1][0] = 0
            M_oln[n - 1][1] = np.cos(phi) * self.Pi_n(n, np.cos(theta)) * yn(n, rho) / rho
            M_oln[n - 1][2] = -np.sin(phi) * self.Tau_n(n, np.cos(theta)) * yn(n, rho) / rho
        return M_oln

    def M_ELN(self, rho, theta, phi):
        M_eln = np.zeros((self.count_harmonics, 3))
        for n in range(1, self.count_harmonics + 1):
            M_eln[n - 1][0] = 0
            M_eln[n - 1][1] = -np.sin(phi) * self.Pi_n(n, np.cos(theta)) * yn(n, rho) / rho
            M_eln[n - 1][2] = -np.cos(phi) * self.Tau_n(n, np.cos(theta)) * yn(n, rho) / rho
        return M_eln

    def N1_OLN(self, rho, theta, phi):
        N1_oln = np.zeros((self.count_harmonics, 3))
        for n in range(1, self.count_harmonics + 1):
            N1_oln[n - 1][0] = np.sin(phi) * n * (n + 1) * np.sin(theta) * self.Pi_n(n, np.cos(theta)) * yn(n, rho) / \
                               (rho ** 2)
            N1_oln[n - 1][1] = np.sin(phi) * self.Tau_n(n, np.cos(theta)) * self.D1(n, rho) * yn(n, rho) / rho
            N1_oln[n - 1][2] = np.cos(phi) * self.Pi_n(n, np.cos(theta)) * self.D1(n, rho) * yn(n, rho) / rho
        return N1_oln

    def N3_OLN(self, rho, theta, phi):
        N3_oln = np.zeros((self.count_harmonics, 3))
        for n in range(1, self.count_harmonics + 1):
            N3_oln[n - 1][0] = np.sin(phi) * n * (n + 1) * np.sin(theta) * self.Pi_n(n, np.cos(theta)) * yn(n, rho) / \
                               (rho ** 2)
            N3_oln[n - 1][1] = np.sin(phi) * self.Tau_n(n, np.cos(theta)) * self.D3(n, rho) * yn(n, rho) / rho
            N3_oln[n - 1][2] = np.cos(phi) * self.Pi_n(n, np.cos(theta)) * self.D3(n, rho) * yn(n, rho) / rho
        return N3_oln

    def N1_ELN(self, rho, theta, phi):
        N1_eln = np.zeros((self.count_harmonics, 3))
        for n in range(1, self.count_harmonics + 1):
            N1_eln[n - 1][0] = np.cos(phi) * n * (n + 1) * np.sin(theta) * self.Pi_n(n, np.cos(theta)) * yn(n, rho) / \
                               (rho ** 2)
            N1_eln[n - 1][1] = np.cos(phi) * self.Tau_n(n, np.cos(theta)) * self.D1(n, rho) * yn(n, self.K * rho) / rho
            N1_eln[n - 1][2] = -np.sin(phi) * self.Pi_n(n, np.cos(theta)) * self.D1(n, rho) * yn(n, self.K * rho) / rho
        return N1_eln

    def N3_ELN(self, rho, theta, phi):
        N3_eln = np.zeros((self.count_harmonics, 3))
        for n in range(1, self.count_harmonics + 1):
            N3_eln[n - 1][0] = np.cos(phi) * n * (n + 1) * np.sin(theta) * self.Pi_n(n, np.cos(theta)) * yn(n, rho) / \
                               (rho ** 2)
            N3_eln[n - 1][1] = np.cos(phi) * self.Tau_n(n, np.cos(theta)) * self.D3(n, rho) * yn(n, self.K * rho) / rho
            N3_eln[n - 1][2] = -np.sin(phi) * self.Pi_n(n, np.cos(theta)) * self.D3(n, rho) * yn(n, self.K * rho) / rho
        return N3_eln

    @staticmethod
    def D1(n, z):
        return spherical_yn(n, z, derivative=True) / spherical_yn(n, z)

    @staticmethod
    def D3(n, z):
        return spherical_jn(n, z, derivative=True) / spherical_jn(n, z)

    def Pi_n(self, n, theta):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return (2 * n - 1) * np.cos(theta) * self.Pi_n(n - 1, theta) / (n - 1) - n * self.Pi_n(n - 1, theta) / (
                        n - 1)

    def Tau_n(self, n, theta):
        if n == 0:
            return 1
        else:
            return n * np.cos(theta) * self.Pi_n(n, theta) - (n + 1) * self.Pi_n(n - 1, theta)

    @staticmethod
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

    def plot_field_volume(self):
        X, Y, Z = np.mgrid[-200:200:self.SIZE * 1j, -200:200:self.SIZE * 1j, -200:200:self.SIZE * 1j]

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=self.FIELD.flatten(),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,  # needs to be small to see through all surfaces
        ))
        fig.show()

    def plot_field_(self):
        # plt.contourf(self.XYZ, self.XYZ, self.FIELD, corner_mask=False)
        # circle = plt.Circle((0, 0), self.R, color='white', fill=False)
        # plt.gca().add_patch(circle)
        # plt.show()
        fig = go.Figure(data=go.Contour(
            z=self.FIELD, x=self.XYZ, y=self.XYZ
        ))
        fig.show()


if __name__ == '__main__':
    a = MieFild(calc=True)
    a.plot_field_()
