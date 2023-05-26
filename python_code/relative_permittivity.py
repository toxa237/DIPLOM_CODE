import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sqlalchemy import create_engine


class RelativePermittivity:
    def __init__(self, list_name: list, list_lambda: (list, np.array), func=lambda x: x):
        self.list_lambda = np.array(list_lambda)
        self.Name = []
        self.func = func
        for i in range(len(list_name)):
            if isinstance(list_name[i], str):
                self.Name.append(self.list_of_val(list_name[i]))
            elif isinstance(list_name[i], (int, float, complex)):
                self.Name.append(lambda _: list_name[i])

    def __getitem__(self, indices):
        i, j = indices
        if i < len(self.Name):
            return self.func(self.Name[i](j))
        raise IndexError("Invalid index or function")

    def list_of_val(self, name_of_material):
        if name_of_material in ['gold', 'Au']:
            df = pd.read_excel('data4materials/Au.xlsx')
            material = 'metal'
        elif name_of_material in ['silver', 'Ag']:
            df = pd.read_excel('data4materials/Ag.xlsx')
            material = 'metal'
        elif name_of_material in ['Al']:
            df = pd.read_excel('data4materials/Al.xlsx')
            material = 'metal'
        elif name_of_material in ['Cu']:
            df = pd.read_excel('data4materials/Cu.xlsx')
            material = 'metal'
        elif name_of_material in ['Ti']:
            df = pd.read_excel('data4materials/Ti.xlsx')
            material = 'metal'
        elif name_of_material in ['glass']:
            eps_ = 7
            material = 'dielectric'
        elif name_of_material in ['ebonite']:
            eps_ = 4.3
            material = 'dielectric'
        elif name_of_material in ['TiO2']:
            eps_ = 2.6142
            material = 'dielectric'     
        else:
            raise Exception("metal not in base")

        if material == 'metal':
            w = df['Wavelength, µm'].values * 1000
            n = df['n'].values
            k = df['k'].values
            fn = CubicSpline(w, n)
            fk = CubicSpline(w, k)
        else:
            fn = lambda _: eps_
            fk = lambda _: 0

        return lambda x: fn(x) + 1j * fk(x)


if __name__ == "__main__":
    # a = get_refractive_index(550, 'silver')
    # print(a)
    a = RelativePermittivity(['glass'], np.arange(200, 1000, 1))
    wave = np.arange(200, 1000, 1)
    rp = [a[0, i] for i in wave]
    plt.plot(wave, np.real(rp), label="real")
    plt.plot(wave, np.imag(rp), label="imag")
    plt.xlabel("Wavelength, nm", fontsize=20)
    plt.ylabel("Relative permittivity", fontsize=20)
    plt.legend(fontsize='large')
    plt.show()
