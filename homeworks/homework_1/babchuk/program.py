# Загрузка библиотек
import json
import pandas as pd
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint

with open('2.json') as f:
    input_data = json.load(f)
input_data

# функция расчета плотности воды в зависимости от давления и температуры
def rho_w_kgm3(P_Mpa,T_K, ws = 0, rho_w_sc_kgm3=None):
    if rho_w_sc_kgm3 is None:
        rho_w_sc_kgm3 = 1000*(1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
    return rho_w_sc_kgm3 /(1+(T_K-273)/10000*(0.269*(T_K-273)**(0.637)-0.8))

# функция расчета солености через плотсноть
def salinity_gg(rho_kgm3):
    sal = 1/rho_kgm3*(1.36545*rho_kgm3-(3838.77*rho_kgm3-2.009*rho_kgm3**2)**0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if sal>0 :
        return sal
    else:
        return 0

# Расчет вязкости воды в зависимости от температуры и давления
def visc_w_cP(P_Mpa,T_K, ws = 0):
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

def pressure_gradient(q_m3day, P_Mpa, T_C, d_m, rho_water_sc_kgm3, cos_alpha, roughness):
    rho_kgm3 = rho_w_kgm3(P_Mpa, T_C + 273, rho_w_sc_kgm3 = rho_water_sc_kgm3)
    mu_cP = visc_w_cP(P_Mpa, T_C + 273, salinity_gg(rho_water_sc_kgm3))
    f = friction_Jain(q_m3day, d_m, mu_cP, rho_kgm3, roughness)
    g = 9.81
    q_m3sec = q_m3day / 86400
    return (rho_kgm3 * g * cos_alpha - 0.815 * f * rho_kgm3 /( d_m ** 5) * (q_m3sec )**2) / 1000000

Pwh = input_data['p_wh'] * 0.1013
Twh = input_data['t_wh']
H = input_data['md_vdp']
Q = 100
dtub_m = input_data['d_tub'] / 10
yw = input_data['gamma_water']
ang_fl = 90 - input_data['angle']
roughness_m = input_data['roughness']
TempGrad = input_data['temp_grad'] / 100

def Cos_Ang_fl(h):
    return np.cos(2 * np.pi * ang_fl / 360)

def dPTdL(PT, h):
    dPdL = pressure_gradient(Q, PT[0], PT[1], dtub_m , yw * 1000, Cos_Ang_fl(h), roughness_m)
    dTdL = TempGrad * Cos_Ang_fl(h)
    return [dPdL, dTdL]

# граничные условия
PTwh = [Pwh,Twh]
# диапазон и интервалы интегрирования
hs = np.linspace(0, H, 100)
# решаем систему уравнений
PTs = odeint(dPTdL, PTwh, hs)
# результаты
P = PTs[:,0]
T = PTs[:,1]

plt.plot(P,hs, label ="давление")
plt.plot(T,hs, label = "температура")
plt.xlabel("P, Т")
plt.ylabel("h, длина скважины, м")
ax = plt.gca()
ax.invert_yaxis()
plt.legend()
plt.title("Распределение давления")


def dPTdL(PT, h, Q=Q):
    dPdL = pressure_gradient(Q, PT[0], PT[1], dtub_m, yw * 1000, Cos_Ang_fl(h), roughness_m)
    dTdL = TempGrad * Cos_Ang_fl(h)
    return [dPdL, dTdL]


def calc_p_wf_mpa(Q):
    # граничные условия
    PTwh = [Pwh, Twh]
    # диапазон и интервалы интегрирования
    hs = np.linspace(0, H, 10)
    # решаем систему уравнений
    PTs = odeint(dPTdL, PTwh, hs, args=(Q,))
    # результаты
    P = PTs[:, 0]
    T = PTs[:, 1]
    return P[-1]


calc_p_wf_mpa(Q)

res_list = []
q_range_m3day = np.arange(1, 400, dtype=int)
for thiq_q_liq_m3day in q_range_m3day:
    this_p_wf_mpa = calc_p_wf_mpa(thiq_q_liq_m3day)
    this_p_wf_atm = this_p_wf_mpa * 9.86
    res_list.append(round(this_p_wf_atm, 4))

plt.plot(q_range_m3day, res_list)
plt.xlabel('Q, дебит жидкости, m3/сут')
plt.ylabel('P, забойное давление, атм')
plt.title("Зависимость градиента давления от дебита");

res_list[0], res_list[-1]

res_json = {"q_liq": [int(x) for x in q_range_m3day], 'p_wf' : res_list}
with open('output.json', 'w') as oj:
    json.dump(res_json, oj)

res_json