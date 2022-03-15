import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import json
import os


def rho_w_kgm3(P_MPa, T_K, ws=0):
    """
    рассчитывает плотность воды в зависимости от давления, температуры и солёности
    :param P_MPa: давление в МПа (в данной функции пренебрегаем сжимаемостью воды, поэтому давление не используется)
    :param T_K: температура в К
    :param ws: солёность воды (по умолчанию равна нулю)
    :return: плотность воды в кг на кубический метр
    """
    rho_w_sc_kgm3 = 1000*(1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
    return rho_w_sc_kgm3 /(1+(T_K-273)/10000*(0.269*(T_K-273)**(0.637)-0.8))


def salinity_gg(rho_kgm3):
    """
    рассчитывает солёность воды через плотность (в стандартных условиях p и T)
    :param rho_kgm3: плотность воды в кг на кубический метр
    :return: солёность воды
    """
    sal = 1/rho_kgm3*(1.36545*rho_kgm3-(3838.77*rho_kgm3-2.009*rho_kgm3**2)**0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if sal > 0:
        return sal
    else:
        return 0


def visc_w_cP(P_MPa, T_K, ws=0):
    """
    рассчитывает вязкость воды в зависимости от давления, температуры и солёности
    :param P_MPa: давление в МПа
    :param T_K: температура в К
    :param ws: солёность (по умолчанию равна нулю)
    :return: вязкость воды в сПз
    """
    A = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws**2 + 8.7221 * 1000 * ws**3
    B = 1.1217 - 2.6396 * ws + 6.7946 * ws**2 + 54.7119 * ws**3 - 155.586 * ws**4
    muw = A * (1.8 * T_K - 460) ** (-B) * (0.9994 + 0.0058 * P_MPa + 0.6534 * (10) ** (0 - 4) * P_MPa**2)
    return muw


def Re(q_m3day, d_m, mu_mPas=0.2, rho_kgm3=1000):
    """
    рассчитывает число Рейнольдса
    :param q_m3day: дебит жидкости в кубометрах в сутки
    :param d_m: диаметр трубы в метрах
    :param mu_mPas: вязкость жидкости в миллипуазах (по умолчанию 0.2 мПз)
    :param rho_kgm3: плотность жидкости в кг на кубический метр (по умолчанию 1000 кг/м3, чистая вода)
    :return: число Рейнольдса
    """
    v_ms = q_m3day / 86400 / 3.1415 * 4 / d_m ** 2
    return rho_kgm3 * v_ms * d_m / mu_mPas * 1000


def friction_Jain(q_m3day, d_m=0.089, mu_mPas=0.2, rho_kgm3=1000, roughness=0.000018):
    """
    рассчитывает коэффициент трения Муди по Джейн
    :param q_m3day: дебит жидкости в кубометрах в сутки
    :param d_m: диаметр трубы в метрах (по умолчанию 0.089 м)
    :param mu_mPas: вязкость жидкости в миллипуазах (по умолчанию 0.2 мПз)
    :param rho_kgm3: плотность жидкости в кг на кубический метр (по умолчанию 1000 кг/м3, чистая вода)
    :param roughness: шероховатость (по умолчанию 0.000018)
    :return: коэффициент трения Муди
    """
    Re_val = Re(q_m3day, d_m, mu_mPas, rho_kgm3)
    if Re_val < 3000:
        return 64/Re_val
    else:
        return 1/(1.14-2 * np.log10(roughness/d_m + 21.25 / (Re_val ** 0.9)))**2


def friction_Churchill(q_m3day, d_m=0.089, mu_mPas=0.2, rho_kgm3=1000, roughness=0.000018):
    """
    рассчитывает коэффициент трения Муди по Черчиллю
    :param q_m3day: дебит жидкости в кубометрах в сутки
    :param d_m: диаметр трубы в метрах (по умолчанию 0.089 м)
    :param mu_mPas: вязкость жидкости в миллипуазах (по умолчанию 0.2 мПз)
    :param rho_kgm3: плотность жидкости в кг на кубический метр (по умолчанию 1000 кг/м3, чистая вода)
    :param roughness: шероховатость (по умолчанию 0.000018)
    :return: коэффициент трения Муди
    """
    Re_val = Re(q_m3day, d_m, mu_mPas, rho_kgm3)
    A = (-2.457 * np.log((7/Re_val)**(0.9)+0.27*(roughness/d_m)))**16
    B = (37530/Re_val)**16
    return 8 * ((8/Re_val)**12+1/(A+B)**1.5)**(1/12)


def dp_dh(p_MPa, T_K, q_liq_per_day, d_m, angle_deg, roughness, rho_water):

    salinity = salinity_gg(rho_water)
    rho = rho_w_kgm3(p_MPa, T_K, salinity)
    mu = visc_w_cP(p_MPa, T_K, salinity)

    dp_dl_grav = rho * 9.81 * np.sin(angle_deg * np.pi / 180)

    q_liq_per_second = q_liq_per_day / 86400
    f = friction_Churchill(q_liq_per_second, d_m, mu, rho, roughness)
    dp_dl_fric = f * rho * q_liq_per_second**2 / d_m**5

    return (dp_dl_grav - 0.815 * dp_dl_fric) / 1e6


def dpdT_dh(pT, h, q_liq_per_day, d_m, angle_deg, roughness, T_grad, rho_water):
    dpdh = dp_dh(pT[0], pT[1], q_liq_per_day, d_m, angle_deg, roughness, rho_water)
    dTdh = T_grad*np.sin(angle_deg * np.pi / 180)
    return [dpdh, dTdh]


if __name__ == '__main__':

    f = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input_data', '16.json'))
    input_parameters = json.load(f)
    f.close()

    rho_water = input_parameters['gamma_water'] * 1000  # плотность воды, кг/м3
    md_vdp = input_parameters['md_vdp']  # глубина верхних дыр перфорации, м
    d_tub = input_parameters['d_tub']  # диаметр НКТ, м
    angle = input_parameters['angle']  # угол наклона скважины к горизонтали, градус
    roughness = input_parameters['roughness']  # шероховатость
    p_wh = input_parameters['p_wh'] * 0.101325  # буферное давление, атм
    T_wh = input_parameters['t_wh'] + 273  # температура жидкости у буферной задвижки, градус Кельвина
    temp_grad = input_parameters['temp_grad'] / 100  # геотермический градиент, градус Кельвина / 1 м

    q_liqs = np.arange(10, 400, 10)
    pressure_result = np.array([])
    temperature_result = np.array([])
    for q_liq in q_liqs:
        #result = solve_ivp(dpdT_dh, t_span=[0, md_vdp], y0=np.array([p_wh, T_wh]),
        #                   args=(q_liq, d_tub, angle, roughness, temp_grad, rho_water))
        result = odeint(dpdT_dh, [p_wh, T_wh], np.linspace(0, md_vdp, 100),
                        args=(q_liq, d_tub, angle, roughness, temp_grad, rho_water))
        pressure_result = np.append(pressure_result, result[:, 0][-1])
        temperature_result = np.append(temperature_result, result[:, 1][-1])

    plt.plot(q_liqs, pressure_result*9.86923)
    plt.show()
    #print(result.y[0])
