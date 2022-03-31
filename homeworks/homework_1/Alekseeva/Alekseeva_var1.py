# Загрузка библиотек необходимых для отрисовки графиков
import matplotlib
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import json
import plotly.express as px
import plotly.graph_objects as go
# %matplotlib inline

with open('1.json', 'r') as file:
    data = json.load(file)

gamma_water = data['gamma_water']
md_vdp = data['md_vdp'] # м
d_tub = data['d_tub'] # м
angle = data['angle'] # град
roughness = data['roughness'] # м
p_wh = data['p_wh']*101325*1e-6 # МПа
t_wh = data['t_wh'] + 273 # K
temp_grad = data['temp_grad']/100 # 1/м

rho = gamma_water * 1000


# функция расчета плотности воды в зависимости от давления и температуры
def rho_w_kgm3(P_Mpa, T_K, ws):
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


ws = salinity_gg(rho)


# Расчет вязкости воды в зависимости от температуры и давления
def visc_w_cP(P_Mpa, T_K, ws):
    #     ws = salinity_gg(rho)
    A = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws * ws + 8.7221 * 1000 * ws * ws * ws
    B = 1.1217 - 2.6396 * ws + 6.7946 * ws * ws + 54.7119 * ws * ws * ws - 155.586 * ws * ws * ws * ws
    muw = A * (1.8 * T_K - 460) ** (-B) * (0.9994 + 0.0058 * P_Mpa + 0.6534 * (10) ** (0 - 4) * P_Mpa * P_Mpa)
    return muw


# Расчет числа Рейнольдса
def Re(q_m3day, d_m, mu_mPas, rho_kgm3):
    # q_m3day - дебит жидкости, м3/сут
    # rho_kgm3 - плотность воды или жидкости, по умолчанию 1000 кг/м3, чистая вода
    # mu_mPas  - вязкость жидкости по умолчанию 0.2 мПас
    # d_m      - диаметр трубы, м
    v_ms = q_m3day / 86400 / 3.1415 * 4 / d_m ** 2
    return rho_kgm3 * v_ms * d_m / mu_mPas * 1000


def friction_Jain(q_m3day, d_m, mu_mPas, rho_kgm3, roughness):
    Re_val = Re(q_m3day, d_tub, mu_mPas, rho_kgm3)
    if Re_val < 3000:
        return 64 / Re_val
    else:
        return 1 / (1.14 - 2 * np.log10(roughness / d_m + 21.25 / (Re_val ** 0.9))) ** 2


def friction_Churchill(q_m3day, d_m, mu_mPas, rho_kgm3, roughness):
    Re_val = Re(q_m3day, d_tub, mu_mPas, rho_kgm3)
    A = (-2.457 * np.log((7 / Re_val) ** (0.9) + 0.27 * (roughness / d_m))) ** 16
    B = (37530 / Re_val) ** 16
    return 8 * ((8 / Re_val) ** 12 + 1 / (A + B) ** 1.5) ** (1 / 12)

def dp_dh(h, p, q_liq, d_m, angle, roughness, rho, ws):

    t = t_wh + temp_grad * mt.sin(angle * mt.pi / 180) * h
    rho_new = rho_w_kgm3(p, t, ws)
    mu = visc_w_cP(p, t, ws)
    q_liq_s = q_liq / 86400
    f = friction_Churchill(q_liq, d_m, mu, rho_new, roughness)
    dp_dl_grav = rho_new * 9.81 * mt.sin(angle * mt.pi / 180)
    dp_dl_fric = f * rho_new * q_liq_s ** 2 / d_m ** 5
    return (dp_dl_grav - 0.815 * dp_dl_fric) / 1e6


q = np.linspace(1e-6, 400, 40)
h = np.linspace(0, md_vdp, 40)
p_d = []
for j in q:
    p_q = [p_wh]

    for i in range(1, len(h)):
        p = p_q[-1] + dp_dh(h[i - 1], p_q[-1], j, d_tub, angle, roughness, rho, ws) * (h[i] - (h[i - 1]))
        p_q.append(p)
    p_d.append(p_q[-1] / 0.101325)

fig = go.Figure()
fig.add_trace(go.Scatter(x=q, y=p_d, mode='lines+markers', line_width=3, line_color="turquoise"))
fig.update_layout(xaxis_title="$Дебит, м^3/сут$", yaxis_title='Забойное давление, атм')
fig.show()

output = {"q_liq": q.tolist(), "p_wf": p_d}
with open('output.json', 'w') as outfile:
    json.dump(output, outfile, indent='\t')