import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sqlalchemy import create_engine


class RelativePermittivity:
    def __init__(self, list_name: list, list_lambda: (list, np.array)):
        self.list_lambda = np.array(list_lambda)

        # self.SQL_CON = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)

        self.Name = []
        for i in range(len(list_name)):
            if isinstance(list_name[i], str):
                self.Name.append(self.list_of_val(list_name[i]))
            elif isinstance(list_name[i], (int, float, complex)):
                self.Name.append((np.ones(self.list_lambda.shape[0])*list_name[i]).tolist())

        self.Name = np.array(self.Name)

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
            fn = lambda x: eps_
            fk = lambda x: 0

        list_of_coef = []
        for i in self.list_lambda:
            list_of_coef.append(fn(i) + 1j * fk(i))

        return list_of_coef

# reqvest = f'''SELECT eps_inf, omega_p, gamma, const FROM Relative_Permittivity
# WHERE matirials_name="{name_of_metal}" '''
# met_param = pd.read_sql(reqvest, con=self.SQL_CON).to_dict('records')
#
# if not met_param:
#     print(name_of_metal, 'not in table')
#     raise Exception
#
# met_param = met_param[0]
#
# if met_param['const']:
#     return np.ones(self.list_lambda.shape[0]) * met_param['const']
#
# eps_inf, omega_p, gamma = met_param['eps_inf'], met_param['omega_p'], met_param['gamma']
# del met_param
#
# SofL = 299_792_458
# for lamb in self.list_lambda:
#     omega = 2 * np.pi * SofL / lamb
#     eps_k = eps_inf - omega_p**2 / (omega**2 + 1j * gamma * omega)
#     list_of_coef.append(eps_k)
# return np.array(list_of_coef, dtype=np.complex128)


if __name__ == "__main__":
    # a = get_refractive_index(550, 'silver')
    # print(a)
    a = RelativePermittivity(['Au'], np.arange(200, 1000, 1))
    print(a.Name)
    plt.plot(np.arange(200, 1000, 1), np.real(a.Name[0]))
    plt.plot(np.arange(200, 1000, 1), np.imag(a.Name[0]))
    plt.show()
