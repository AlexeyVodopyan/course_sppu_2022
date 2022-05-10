# -*- coding: utf-8 -*-
"""
Created by 'kolya' on 30.03.2022 at 19:04
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# плотность от температуры, давления, солености ws
def rho_w_kgm3(T_K, ws=0):
    rho_w_sc_kgm3 = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)
    return rho_w_sc_kgm3 / (1 + (T_K - 273) / 10000 * (0.269 * (T_K - 273) ** (0.637) - 0.8))


# функция расчета солености через плотсноть
def salinity_gg(rho_kgm3):
    sal = 1 / rho_kgm3 * (1.36545 * rho_kgm3 - (3838.77 * rho_kgm3 - 2.009 * rho_kgm3 ** 2) ** 0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if sal > 0:
        return sal
    else:
        return 0

# Расчет вязкости воды в зависимости от температуры и давления Matthews and Russel
def visc_w_cP(P_Mpa, T_K, ws=0):
    A = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws * ws + 8.7221 * 1000 * ws * ws * ws
    B = 1.1217 - 2.6396 * ws + 6.7946 * ws * ws + 54.7119 * ws * ws * ws - 155.586 * ws * ws * ws * ws
    muw = A * (1.8 * T_K - 460) ** (-B) * (0.9994 + 0.0058 * P_Mpa + 0.6534 * (10) ** (0 - 4) * P_Mpa * P_Mpa)
    return muw

# Расчет числа Рейнольдса
def Re(q_m3day, d_m, mu_mPas=0.2, rho_kgm3=1000):
    # q_m3day - дебит жидкости, м3/сут
    # rho_kgm3 - плотность воды или жидкости, по умолчанию 1000 кг/м3, чистая вода
    # mu_mPas  - вязкость жидкости по умолчанию 0.2 мПас
    # d_m      - диаметр трубы, м
    v_ms = q_m3day / 86400 / 3.1415 * 4 / d_m ** 2
    return rho_kgm3 * v_ms * d_m / mu_mPas * 1000

def friction_Jain(q_m3day, d_m=0.089, mu_mPas=0.2, rho_kgm3=1000, roughness=0.000018):
    Re_val = Re(q_m3day, d_m, mu_mPas, rho_kgm3)
    if Re_val < 3000:
        return 64 / Re_val
    else:
        return 1 / (1.14 - 2 * np.log10(roughness / d_m + 21.25 / (Re_val ** 0.9))) ** 2

def friction_Churchill(q_m3day, d_m=0.089, mu_mPas=0.2, rho_kgm3=1000, roughness=0.000018):
    Re_val = Re(q_m3day, d_m, mu_mPas, rho_kgm3)
    A = (-2.457 * np.log((7 / Re_val) ** (0.9) + 0.27 * (roughness / d_m))) ** 16
    B = (37530 / Re_val) ** 16
    return 8 * ((8 / Re_val) ** 12 + 1 / (A + B) ** 1.5) ** (1 / 12)

def dp_dh(h, initial, q_liq, rho, d_m, angle,temp_grad,ws, roughness,well_md):
    dt=temp_grad
#     rho=rho_w_kgm3(initial[1], ws = ws)
    mu = visc_w_cP(initial[0], initial[1])
    f = friction_Churchill(q_liq, d_m, mu, rho, roughness)
    dp_dl_grav = rho * 9.81 * mt.sin(angle*mt.pi/180)
    dp_dl_fric = f * rho * (q_liq/86400)** 2 / d_m ** 5
    dp=(dp_dl_grav - 0.810 * dp_dl_fric)*10**(-6)

    return np.array([dp,dt])


def get_BHP(q_liq,p_init,t_init,rho ,d_m,angle,temp_grad,ws,roughness,well_md):
    t_span=[0, well_md]
    times=np.linspace(t_span[0],t_span[1],10)
    solution = solve_ivp(dp_dh, t_span=t_span, y0=np.array([p_init,t_init]), t_eval=times,args=(q_liq, d_m, rho, angle,temp_grad,ws,roughness,well_md))
    result_MPA=solution.y[0][-1]
    result_atm=result_MPA*10**6/101325
    return result_atm

def plotVLP(result:dict):
    plt.title('VLP нагнетательной скважины')
    plt.plot(result['q_liq'], result['p_wf'])
    plt.show()

# def write_result(result:dict):
#     result_json=json.dumps(result,indent=4)
#     print(result_json)
#     with open('output.json', 'w') as output_json:
#         output_json.write(result_json)

def getVLP(input_data:dict):
    """MAIN FUNCTION!"""
    p_init = input_data['p_wh'] * 101325 * 10 ** (-6)  # в МПА
    t_init = input_data['t_wh'] + 273  # в К
    temp_grad = input_data["temp_grad"] / 100  # градус/метр
    rho_init = input_data['gamma_water'] * 1000  # кг/м3
    d_m = input_data["d_tub"]  # в м
    angle = input_data['angle']  # градусов к горизонтали!
    well_md = input_data["md_vdp"]  # md скважины в м
    ws = salinity_gg(rho_init)
    roughness = input_data['roughness']

    q_list=np.linspace(1,400,20)
    pressure_list=[]
    for q in q_list:
        pressure_list.append(get_BHP(q,p_init,t_init ,d_m, rho_init, angle,temp_grad,ws, roughness,well_md))

    result=dict({'q_liq':list(q_list),'p_wf':pressure_list})
    return result

if __name__ == '__main__':
    input_data={"gamma_water": 1.051335195601004, "md_vdp": 3039.187950560042, "d_tub": 0.08238636257424214, "angle": 56.8531526670004, "roughness": 0.0005239809355760531, "p_wh": 162.25483127203776, "t_wh": 31.242276359683466, "temp_grad": 2.6101485472727397}
    result=getVLP(input_data)

    print('Дебиты:', result['q_liq'])
    print('Давления:', result['p_wf'])

    # plotVLP(result)
    with open('output.json','w') as output:
        json.dump(result, output, indent=4)

