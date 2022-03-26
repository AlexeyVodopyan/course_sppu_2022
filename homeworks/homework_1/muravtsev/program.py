import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import json
import os


def rho_w_kgm3(P_MPa, T_K, ws=0.0):
    """
    рассчитывает плотность воды в зависимости от давления, температуры и солёности
    :param P_MPa: давление в МПа (в данной функции пренебрегаем сжимаемостью воды, поэтому давление не используется)
    :param T_K: температура в К
    :param ws: солёность воды (по умолчанию равна нулю)
    :return: плотность воды в кг на кубический метр
    """
    rho_w_sc_kgm3 = 1000*(1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
    return rho_w_sc_kgm3 / (1+(T_K-273)/10000*(0.269*(T_K-273)**(0.637)-0.8))


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


def visc_w_cP(P_MPa, T_K, ws=0.0):
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
    v_ms = q_m3day / 86400 / np.pi * 4 / d_m ** 2
    Re_num = rho_kgm3 * v_ms * d_m / mu_mPas * 1000
    return Re_num


def friction_Jain(q_m3day, d_m=0.089, mu_mPas=0.2, rho_kgm3=1000, roughness=0.000018):
    """
    рассчитывает коэффициент трения Муди по Джейн
    :param q_m3day: дебит жидкости в кубометрах в сутки
    :param d_m: диаметр трубы в метрах (по умолчанию 0.089 м)
    :param mu_mPas: вязкость жидкости в миллипуазах (по умолчанию 0.2 мПз)
    :param rho_kgm3: плотность жидкости в кг на кубический метр (по умолчанию 1000 кг/м3, чистая вода)
    :param roughness: шероховатость в метрах (по умолчанию 0.000018 м)
    :return: коэффициент трения Муди по Джейн
    """
    Re_val = Re(q_m3day, d_m, mu_mPas, rho_kgm3)
    if Re_val < 3000:
        fr = 64/Re_val
    else:
        fr = 1/(1.14-2 * np.log10(roughness/d_m + 21.25 / (Re_val ** 0.9)))**2
    return fr


def friction_Churchill(q_m3day, d_m=0.089, mu_mPas=0.2, rho_kgm3=1000, roughness=0.000018):
    """
    рассчитывает коэффициент трения Муди по Черчиллю
    :param q_m3day: дебит жидкости в кубометрах в сутки
    :param d_m: диаметр трубы в метрах (по умолчанию 0.089 м)
    :param mu_mPas: вязкость жидкости в миллипуазах (по умолчанию 0.2 мПз)
    :param rho_kgm3: плотность жидкости в кг на кубический метр (по умолчанию 1000 кг/м3, чистая вода)
    :param roughness: шероховатость в метрах (по умолчанию 0.000018 м)
    :return: коэффициент трения Муди по Черчиллю
    """
    Re_val = Re(q_m3day, d_m, mu_mPas, rho_kgm3)
    A = (-2.457 * np.log((7/Re_val)**(0.9)+0.27*(roughness/d_m)))**16
    B = (37530/Re_val)**16
    fr = 8 * ((8/Re_val)**12+1/(A+B)**1.5)**(1/12)
    return fr


def Dp_Dx(x):
    """
    расчёт градиента давления на текущем шаге интегрирования
    :param x: шаг интегрирования
    :return: градиент давления на данном шаге
    """
    cos_alpha = (TVD[1] - TVD[0]) / (MD[1] - MD[0])
    T = T_wh + temp_grad * x * cos_alpha

    salinity = salinity_gg(rho_water)
    rho = rho_w_kgm3(p[-1], T, salinity)
    mu = visc_w_cP(p[-1], T, salinity)
    f = friction_Churchill(q_liq_per_day, d_tub, mu, rho, roughness)
    dp_dx = (rho * 9.81 * cos_alpha - f * rho * q_liq_per_second**2 / d_tub**5) / 1e6
    return dp_dx


if __name__ == '__main__':

    # графики основных зависимостей между параметрами
    temperatures = np.linspace(0, 400, 50)
    density = np.linspace(992, 1300, 50)
    rate = np.linspace(0, 5, 50)
    axs = [None] * 4
    fig, ((axs[0], axs[1]), (axs[2], axs[3])) = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))
    fig.tight_layout(pad=5)
    axs[0].plot(temperatures, [rho_w_kgm3(0.101325, temperature + 273, 0.001) for temperature in temperatures])
    axs[0].set_title('Зависимость плотности воды от\n температуры при 1 атм')
    axs[1].plot(density, [salinity_gg(dens) for dens in density])
    axs[1].set_title('Зависимость солёности от плотности\n воды в стандартных условиях')
    for i in [0.1, 0.01, 0.001, 0.0001]:
        axs[2].plot(temperatures, [visc_w_cP(0.101325, temperature + 273, i) for temperature in temperatures],
                 label=f'солёность {i}')
    axs[2].set_title('Зависимость вязкости воды от\n температуры при 1 атм')
    axs[2].legend()
    axs[3].plot(rate, [Re(r, 0.089) for r in rate])
    axs[3].set_title('Зависимость числа Рейнольдса от дебита\n нагнетательной скважины')
    for ax in axs:
        ax.grid(linewidth=0.3)
    plt.show()

    # входные параметры моего варианта задания
    f = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input_data', '16.json'))
    input_parameters = json.load(f)
    f.close()

    # перевод в единицы измерения, необходимые для расчёта
    rho_water = input_parameters['gamma_water'] * 1000  # плотность воды в ст.усл., кг/м3
    md_vdp = input_parameters['md_vdp']  # глубина верхних дыр перфорации, м
    d_tub = input_parameters['d_tub']  # диаметр НКТ, м
    angle = input_parameters['angle']  # угол наклона скважины к горизонтали, градус
    roughness = input_parameters['roughness']  # шероховатость
    p_wh = input_parameters['p_wh'] * 0.101325  # буферное давление, МПа
    T_wh = input_parameters['t_wh'] + 273  # температура жидкости у буферной задвижки, градус Кельвина
    temp_grad = input_parameters['temp_grad'] / 100  # геотермический градиент, градус Кельвина / 1 м

    # задание геометрии (инклинометрии)
    MD = [0, md_vdp]
    TVD = [0, np.sin(angle * np.pi / 180) * md_vdp]

    # интегрирование вдоль скважины
    p_wf_q = np.empty(0)  # лист с забойными давлениями при разных значениях дебита

    q_liqs = np.arange(0, 400, 10)

    for q_liq_per_day in q_liqs:
        q_liq_per_second = q_liq_per_day / 86400
        h_md = np.linspace(0, md_vdp, 200)
        p = [p_wh]
        for i in range(0, 199):
            p_next = p[-1] + quad(Dp_Dx, h_md[i], h_md[i+1])[0]
            p.append(p_next)
        p_wf_q = np.append(p_wf_q, p[-1])

    p_wf_q = p_wf_q / 0.101325  # из МПа в атм

    plt.plot(q_liqs, p_wf_q)
    plt.title('Зависимость забойного давления от дебита')
    plt.xlabel("Дебит, м^3/сут")
    plt.ylabel("Давление, атм")
    plt.grid()
    plt.show()

    # запись в json файл
    to_json_file = {
        'q_liq': q_liqs.tolist(),
        'p_wf': p_wf_q.tolist()
    }

    with open('output.json', mode='w', encoding='UTF-8') as f:
        json.dump(to_json_file, f, indent=4)
