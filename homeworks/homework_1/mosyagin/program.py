#!/usr/bin/env python
# coding: utf-8

# # Импорт библиотек

# In[1]:


import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import plotly.express as px


# # Считывание данных

# In[2]:


base_path = os.getcwd()
input_data_path = 'input_data'
json_number = 15
this_path = os.path.join(base_path, input_data_path, f'{json_number}.json')

with open(this_path, "r") as read_file:
    data = json.load(read_file)
data


# In[3]:


gamma_water = data['gamma_water']
md_vdp = data['md_vdp']
d_tub = data['d_tub']/10
angle = data['angle']
roughness = data['roughness']
p_wf = data['p_wh'] * 0.101325
t_wh = data['t_wh'] + 273.15
temp_grad = data['temp_grad']
# массив дебитов
array_q_liq = np.arange(0, 400 + 10, 10)
array_q_liq[0] = 1


# # Функции

# In[4]:


# соленость
def salinity_gg(gamma_water):
    rho_kgm3 = gamma_water*1000
    sal = 1/rho_kgm3*(1.36545*rho_kgm3 -
                      (3838.77*rho_kgm3-2.009*rho_kgm3**2)**0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if sal > 0:
        return sal
    else:
        return 0

def rho_w_kgm3(P_Mpa, T_K, ws):
    rho_w_sc_kgm3 = 1000*(1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
    return rho_w_sc_kgm3 / (1+(T_K-273)/10000*(0.269*(T_K-273)**(0.637)-0.8))


# Расчет вязкости воды в зависимости от температуры и давления
def visc_w_cP(P_Mpa, T_K, ws=0):

    A = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws * ws + 8.7221 * 1000 * ws * ws * ws
    B = 1.1217 - 2.6396 * ws + 6.7946 * ws * ws + 54.7119 * ws * ws * ws - 155.586 * ws * ws * ws * ws
    muw = A * (1.8 * T_K - 460) ** (-B) * (0.9994 + 0.0058 * P_Mpa + 0.6534 * (10) ** (-4) * P_Mpa * P_Mpa)
    return muw


def Re(rho_m, d_in,  mu_m, q_m3day):
    # rho_m , kg/m3
    # v_m , m/s
    # d_in , m
    # mu_m , MPa s
    v_m = q_m3day / 86400 / 3.1415  * 4 / d_in ** 2
    return rho_m*v_m*(d_in) / (mu_m) * 1000


def friction_Churchill(q_m3day, d_m, mu_mPas, rho_kgm3, roughness, Re_val):
    A = (-2.457 * np.log((7/Re_val)**(0.9)+0.27*(roughness/d_m)))**16
    B = (37530/Re_val)**16
    return 8 * ((8/Re_val)**12+1/(A+B)**1.5)**(1/12)


def dp_dl(depth, pressure, tempetature_init,
          q_liq, d_m, angle, roughness, func_calc_temp, gamma_water):

    this_temperature = func_calc_temp(depth)

    ws = salinity_gg(gamma_water)
    rho = rho_w_kgm3(pressure, this_temperature, ws)
    mu = visc_w_cP(pressure, this_temperature)
    re = Re(rho, d_m, mu, q_liq)
    f = friction_Churchill(q_liq, d_m, mu, rho, roughness, re)

    dp_dx = rho * 9.81 * np.sin(angle*np.pi/180) - (8 / np.pi**2) *         f * rho * (q_liq / 86400) ** 2 / (d_m ** 5)
    return dp_dx / 10**6


# # Инициализация температурного градиента

# In[5]:


# глубина
array_h_m = np.linspace(0, md_vdp)
# темпераутра
array_temp_k = t_wh + array_h_m * temp_grad / 100
# линейная интерполяция
restore_temp_k = interp1d(array_h_m, array_temp_k)


# # Расчет

# In[6]:


array_p_wf = []

for this_q_liq in array_q_liq:
    result = solve_ivp(dp_dl,
                       t_span=[0, md_vdp],
                       y0=np.array([p_wf]),
                       args= (t_wh, this_q_liq,
                              d_tub, angle,
                              roughness, restore_temp_k, gamma_water),
                       max_step = 1)
    array_p_wf.append(result.y[0][-1])

array_p_wf = np.array(array_p_wf) * 9.8692327


# # График

# In[7]:


plt.plot(array_q_liq, array_p_wf, marker='o')
plt.grid()
plt.xlabel('Q_LIQ_M3DAY')
plt.ylabel('P_wf')
plt.title('Зависимость Q от P_wf')


# # Экспорт в json

# In[9]:


dict_to_json = {"q_liq": array_q_liq.tolist(), "p_wf": array_p_wf.tolist()}
filename = 'output.json'
with open(filename, 'w') as f:
    json.dump(dict_to_json, f, indent=4)

