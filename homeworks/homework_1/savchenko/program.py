# import mkl
import json
import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt

# mkl.domain_set_num_threads(1, domain='fft')

# Исходные данные
df = pd \
    .read_json(
    'https://raw.githubusercontent.com/AlexeyVodopyan/course_sppu_2022/main/homeworks/homework_1/input_data/19.json',
    orient='index', typ='series')
q_liqs, p_wf = [], []
[q_liqs.append(i) for i in range(10, 400, 10)]


# Расчет солености через плотность
def salinity_gg(rho_kgm3=df.gamma_water * 1000):
    sal = 1 / rho_kgm3 * (1.36545 * rho_kgm3 - (3838.77 * rho_kgm3 - 2.009 * rho_kgm3 ** 2) ** 0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    return sal if sal > 0 else 0


# Расчет плотности воды в зависимости от температуры
def rho_w_kgm3(T_K, ws):
    rho_w_sc_kgm3 = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)
    return rho_w_sc_kgm3 / (1 + (T_K - 273) / 10000 * (0.269 * (T_K - 273) ** 0.637 - 0.8))


# Расчет вязкости воды в зависимости от температуры и давления
def visc_w_cP(P_Mpa, T_K, ws):
    A = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws * ws + 8.7221 * 1000 * ws * ws * ws
    B = 1.1217 - 2.6396 * ws + 6.7946 * ws * ws + 54.7119 * ws * ws * ws - 155.586 * ws * ws * ws * ws
    muw = A * (1.8 * T_K - 460) ** (-B) * (0.9994 + 0.0058 * P_Mpa + 0.6534 * (10) ** (0 - 4) * P_Mpa * P_Mpa)
    return muw


# Расчет числа Рейнольдса
def Re(q_m3day, mu_mPas, rho_kgm3, d_m=df.d_tub / 10):
    v_ms = (q_m3day / 86400) / mt.pi * 4 / d_m ** 2
    return rho_kgm3 * v_ms * d_m / mu_mPas * 1000


# Расчет коэффициента трения с помощью корреляции Черчилля
def friction_Churchill(q_m3day, mu_mPas, rho_kgm3, d_m=df.d_tub / 10, roughness=df.roughness):
    Re_val = Re(q_m3day, mu_mPas, rho_kgm3)
    A = (-2.457 * np.log((7 / Re_val) ** (0.9) + 0.27 * (roughness / d_m))) ** 16
    B = (37530 / Re_val) ** 16
    return 8 * ((8 / Re_val) ** 12 + 1 / (A + B) ** 1.5) ** (1 / 12)


# Расчет градиента давления
def dp_dh(p, t, q_liq, ws, d_m=df.d_tub / 10, angle=df.angle):
    rho = rho_w_kgm3(t, ws)
    mu = visc_w_cP(p, t, ws)

    f = friction_Churchill(q_liq, mu, rho)
    dp_dl_grav = rho * 9.81 * mt.sin(angle / 180 * mt.pi)
    dp_dl_fric = f * rho * (q_liq / 86400) ** 2 / d_m ** 5
    return (dp_dl_grav - 0.810 * dp_dl_fric) * 10 ** (-5)


# Расчет VLP-кривой
ws = salinity_gg()
for q in q_liqs:
    P, T = df.p_wh / 10, df.t_wh + 273
    i, deapth = 0, 0
    while deapth < df.md_vdp:
        dp_dh_step = dp_dh(P, T, q, ws)
        T += df.temp_grad / 100
        P += dp_dh_step
        deapth += 1
    p_wf.append(P)

# Сохранение результатов в файл
plt.title("VLP-кривая для нагнетательной скважины")
plt.plot(q_liqs, p_wf)
plt.xlabel('Дебит нагнетаемой воды, м3/сут')
plt.ylabel('Забойное давление, атм')
plt.show()
result = {'q_liq': q_liqs, 'p_wf': p_wf}
with open('C:/Users/Vladislav/Desktop/output.json', 'w') as outfile:
    outfile.write(json.dumps(result, indent=4))
