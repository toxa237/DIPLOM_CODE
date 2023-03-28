import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


class RelativePermittivity:
    def __init__(self, list_name: list, list_lambda: (list, np.array)):
        self.list_lambda = np.array(list_lambda)

        self.SQL_CON = create_engine('mysql+pymysql://toxa:password@localhost:3306/DIPLOM', echo=False)

        self.Name = []

        for i in range(len(list_name)):
            if isinstance(list_name[i], str):
                self.Name.append(self.list_of_val(list_name[i]))
            elif isinstance(list_name[i], (int, float)):
                self.Name.append((np.ones(self.list_lambda.shape[0])*list_name[i]).tolist())

        self.Name = np.array(self.Name)

    def list_of_val(self, name_of_metal):
        reqvest = f'''SELECT eps_inf, omega_p, gamma, const FROM Relative_Permittivity 
        WHERE matirials_name="{name_of_metal}" '''
        met_param = pd.read_sql(reqvest, con=self.SQL_CON).to_dict('records')

        if not met_param:
            print(name_of_metal, 'not in table')
            raise Exception

        met_param = met_param[0]

        if met_param['const']:
            return np.ones(self.list_lambda.shape[0]) * met_param['const']

        eps_inf, omega_p, gamma = met_param['eps_inf'], met_param['omega_p'], met_param['gamma']
        del met_param

        SofL = 299_792_458
        list_of_coef = []
        for lamb in self.list_lambda:
            omega = 2 * np.pi * SofL / lamb
            eps_k = eps_inf - omega_p**2 / (omega**2 + 1j * gamma * omega)
            list_of_coef.append(eps_k)
        return np.array(list_of_coef)


if __name__ == "__main__":
    a = RelativePermittivity(['Al'], np.arange(200, 1000, 1)*10e-9)
    print(a.Name)
    plt.plot(np.arange(200, 1000, 1), np.abs(a.Name[0]))
    plt.show()
