import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Calc:
    def __init__(self, h1, gamma_water, angle, d, temp_grad, roughness) -> None:
        """
        Создание объекта расчетника
        """
        self.h1 = h1
        self.gamma_water = gamma_water
        self.angle = angle
        self.d = d
        self.roughness = roughness
        self.temp_grad = temp_grad
    
    # функция расчета плотности воды в зависимости от давления и температуры
    def rho_w_kgm3(self, t, ws = 0):
        rho_w_sc_kgm3 = 1000*(1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
        return rho_w_sc_kgm3 /(1+(t-273)/10000*(0.269*(t-273)**(0.637)-0.8))
        
    # функция расчета солености через плотсноть
    def salinity_gg(self, rho_kgm3):
        sal = 1/rho_kgm3*(1.36545*rho_kgm3-(3838.77*rho_kgm3-2.009*rho_kgm3**2)**0.5)
        # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
        if sal>0 :
            return sal
        else:
            return 0
            
    # Расчет вязкости воды в зависимости от температуры и давления
    def visc_w_cP(self, p, t, ws = 0):
        a = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws ** 2 + 8.7221 * 1000 * ws ** 3
        b = 1.1217 - 2.6396 * ws + 6.7946 * ws ** 2 + 54.7119 * ws ** 3 - 155.586 * ws ** 4

        return a * (1.8 * t - 460) ** (-b) * (0.9994 + 0.0058 * p + 0.6534 * 1e-4 * p ** 2)

    def Re(self, q, d, mu_mPas = 0.2, rho_kgm3 = 1000):
        v_ms = q / (np.pi * d ** 2 / 4)
        return rho_kgm3 * v_ms * d / mu_mPas

    def friction_Churchill(self, q,d_m = 0.089, mu_mPas = 0.2,rho_kgm3 = 1000,roughness=0.000018):
        Re_val = self.Re(q,d_m,mu_mPas,rho_kgm3)
        A = (-2.457 * np.log((7/Re_val)**(0.9)+0.27*(roughness/d_m)))**16
        B = (37530/Re_val)**16
        return 8 * ((8/Re_val)**12+1/(A+B)**1.5)**(1/12)
        
    def dp_dh(self, h, pt, q_liq):
          
        rho = self.rho_w_kgm3(pt[1])
        mu = self.visc_w_cP(pt[0]/101325/10, pt[1], self.salinity_gg(rho))/ 1000
        print(q_liq, self.d, mu, rho, self.roughness)
        f = self.friction_Churchill(q_liq, self.d, mu, rho, self.roughness)
        dp_dl_grav = rho * 9.81 * m.sin(self.angle/180* m.pi)
        dp_dl_fric = -f * rho * q_liq** 2 / self.d ** 5
        return dp_dl_grav + 0.815 * dp_dl_fric, self.temp_grad/100
            
    def calc_pwf_pfl(self, p_wh, t_wh, q):
        result = solve_ivp(
            self.dp_dh,
            t_span=(0, self.h1),
            y0=[p_wh, t_wh],
            args=[q],
        )
        return result.y[0, :][-1]

# исходные данные
data = pd.read_json('8.json', typ='series').to_dict()
print(data)
# Исходные данные для проведения расчета
gamma_water = data['gamma_water']
p_wh = data['p_wh']         # давление на устье, бар
t_wh = data['t_wh']          # температура на устье скважины, С
temp_grad = data['temp_grad']      # температурный градиент град С на 100 м
md_vdp = data['md_vdp']          # измеренная глубина забоя скважины, м
roughness = data['roughness'] # шероховатость
angle = data['angle'] # угол искривления скважины
d_tub = data['d_tub']/10      # диаметр НКТ по которой ведется закачка, м


pwf_list = []
q_list = [_ for _ in range(1, 400, 10)]
well_calc = Calc(md_vdp, gamma_water, angle, d_tub, temp_grad, roughness)

for q in q_list:
    pwf = well_calc.calc_pwf_pfl(p_wh*101325, t_wh+273, q/86400) # рассчет забойного давления
    pwf_list.append(pwf/101325)

import json
with open('output.json', 'w', encoding='utf-8') as output_file:
        json.dump({"q_liq": q_list, "p_wf": pwf_list}, output_file, indent=4)
        
plt.plot(q_list, pwf_list)
plt.ylabel('забойного давления, атм')
plt.xlabel('дебита закачиваемой жидкости, м3/сут')
plt.show()
