"""
Расчёт забойного давления нагнетательной скважины
"""
import json
from os import PathLike
from typing import Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def calc_ws(
        dens_water: float,
) -> float:
    """
    Функция для расчета солености воды.

    :param dens_water: плотность воды, кг/м3;

    :return: значение солености, г/г
    """
    ws = 1 / dens_water * (1.36545 * dens_water - (3838.77 * dens_water - 2.009 * dens_water ** 2) ** 0.5)

    if ws > 0:
        return ws
    else:
        return 0


def calc_dens_water(
        temp: float,
        ws: float,
) -> float:
    """
    Функция для расчета плотности воды при известной температуре и солености.

    :param temp: температура, К;
    :param ws: соленость, г/г;

    :return: значение плотности воды в кг/м3
    """

    dens_ = (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)

    return dens_ / (1 + (temp - 273) / 10000 * (0.269 * (temp - 273) ** 0.637 - 0.8))


def calc_mu_water(
        p: float,
        temp: float,
        ws: float,
) -> float:
    """
    Функция расчета вязкости воды при известных давлении, температуре и солености.

    :param p: давление, МПа;
    :param temp: температура, К;
    :param ws: соленость, г/г;

    :return: значение вязкости в мПа*с
    """
    a = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws ** 2 + 8.7221 * 1000 * ws ** 3
    b = 1.1217 - 2.6396 * ws + 6.7946 * ws ** 2 + 54.7119 * ws ** 3 - 155.586 * ws ** 4

    return a * (1.8 * temp - 460) ** (-b) * (0.9994 + 0.0058 * p + 0.6534 * 1e-4 * p ** 2)


def calc_re_number(
        dens: float,
        velocity: float,
        d: float,
        mu: float,
) -> float:
    """
    Функция расчета числа Рейнольдса.

    :param dens: плотность флюида, кг/м3;
    :param velocity: скорость флюида, м/с;
    :param d: диаметр трубы, м;
    :param mu: вязкость флюида, Па*с;

    :return: значение числа Рейнольдса, безразм
    """
    return dens * velocity * d / mu


def fric_jain(
        d: float,
        roughness: float,
        re_num: float,
) -> float:
    """
    Функция расчета коэффициента трения по корреляции Jain.

    :param d: диаметр трубы, м;
    :param roughness: шероховатость трубы, м;
    :param re_num: число Рейнольдса;

    :return: значение коэффициента трения по корреляции Jain
    """
    if re_num < 3000:
        return 64 / re_num
    else:
        return 1 / (1.14 - 2 * np.log10(roughness / d + 21.25 / re_num ** 0.9)) ** 2


def fric_church(
        d: float,
        roughness: float,
        re_num: float,
) -> float:
    """
    Функция расчета коэффициента трения по корреляции Churchill.

    :param d: диаметр трубы, м;
    :param roughness: шероховатость трубы, м;
    :param re_num: число Рейнольдса;

    :return: значение коэффициента трения по корреляции Churchill
    """
    a = (-2.457 * np.log((7 / re_num) ** 0.9 + 0.27 * roughness / d)) ** 16
    b = (37530 / re_num) ** 16

    return 8 * ((8 / re_num) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)


def calc_fric_factor(
        d: float,
        roughness: float,
        re_num: float,
        method: Optional[Literal['jain', 'church']] = 'church',
) -> float:
    """
    Общая функция для расчета коэффициента трения по выбранной корреляции.

    :param d: диаметр трубы, м;
    :param roughness: шероховатость трубы, м;
    :param re_num: число Рейнольдса;
    :param method: корреляция, варианты: 'jain', 'church';

    :return: значение коэффициента трения по выбранной корреляции
    """

    if method == 'jain':
        return fric_jain(d=d, roughness=roughness, re_num=re_num)
    elif method == 'church':
        return fric_church(d=d, roughness=roughness, re_num=re_num)
    else:
        raise KeyError(f"Wrong correlation type for friction factor calculation, {method} not implemented yet")


def dp_dl(
        q: float,
        d: float,
        dens: float,
        angle: float,
        fric_factor: float,
        g: Optional[float] = 9.81,
) -> float:
    """
    Функция расчета градиента давления для произвольного участка скважины.

    :param q: расход жидкости в трубе, м3/с;
    :param d: диаметр трубы, м;
    :param dens: плотность флюида в трубе, кг/м3;
    :param angle: угол искривления трубы, град;
    :param fric_factor: коэффициент трения жидкости в трубе;
    :param g: гравитационная постоянная;

    :return: значение градиента давления для участка трубы
    """
    dp_dl_grav = dens * g * np.cos(angle * np.pi / 180)
    dp_dl_fric = fric_factor * dens / d ** 5 * q ** 2

    return dp_dl_grav - 0.815 * dp_dl_fric


def calc_p(
        h: float,
        p: float,
        t_grad: Callable,
        q: float,
        d: float,
        angle: float,
        roughness: float,
        ws: float,
        p_units: Optional[Literal['MPa', 'bar']] = 'MPa',
):
    """
    Функция для расчета градиента давления в трубе.

    :param h: глубина, м;
    :param p: давление в начале трубы, МПа;
    :param t_grad: градиент температуры по трубе;
    :param q: расход флюида в трубе, м3/с;
    :param d: диаметр трубы, м;
    :param angle: угол искривления трубы, град;
    :param roughness: шероховатость трубы, м;
    :param ws: соленость воды, г/г;
    :param p_units: единицы измерения давления

    :return: градиент давления в трубе
    """
    if p_units == 'MPa':
        unit_coeff = 1e-6
    elif p_units == 'bar':
        unit_coeff = 1e-5
    else:
        raise KeyError(f"Wrong p_units, {p_units} not implemented yet")

    t = t_grad(h)

    dens_water = calc_dens_water(temp=t, ws=ws) * 1000
    mu_water = calc_mu_water(p=p, temp=t, ws=ws) / 1000

    velocity = q / (np.pi * d ** 2 / 4)

    re_num = calc_re_number(dens=dens_water, velocity=velocity, d=d, mu=mu_water)

    fric_factor = calc_fric_factor(d=d, roughness=roughness, re_num=re_num, method='church')

    return unit_coeff * dp_dl(q=q, d=d, dens=dens_water, angle=angle, fric_factor=fric_factor)


def calc_p_distribution(
        h: ArrayLike,
        p: float,
        t_grad: Callable,
        q: float,
        d: float,
        angle: float,
        roughness: float,
        ws: float,
        p_units: Optional[Literal['MPa', 'bar']] = 'MPa',
) -> tuple[NDArray, NDArray]:
    """
    Функция для расчета распределения давления по трубе.

    :param h: глубина, м;
    :param p: давление в начале трубы, МПа;
    :param t_grad: градиент температуры по трубе;
    :param q: расход флюида в трубе, м3/с;
    :param d: диаметр трубы, м;
    :param angle: угол искривления трубы, град;
    :param roughness: шероховатость трубы, м;
    :param ws: соленость воды, г/г;
    :param p_units: единицы измерения давления

    :return: распределение давления по трубе
    """

    p_integrated = solve_ivp(
        calc_p,
        t_span=h,
        y0=np.array([p]),
        args=(
            t_grad,
            q,
            d,
            angle,
            roughness,
            ws,
            p_units,
        )
    )

    return p_integrated.y[0], p_integrated.t


def calc_pwf(
        h: ArrayLike,
        p: float,
        t_grad: Callable,
        q: float,
        d: float,
        angle: float,
        roughness: float,
        ws: float,
        p_units: Optional[Literal['MPa', 'bar']] = 'MPa',
) -> float:
    """
    Функция для расчета забойного давления.

    :param h: глубина, м;
    :param p: давление в начале трубы, МПа;
    :param t_grad: градиент температуры по трубе;
    :param q: расход флюида в трубе, м3/с;
    :param d: диаметр трубы, м;
    :param angle: угол искривления трубы, град;
    :param roughness: шероховатость трубы, м;
    :param ws: соленость воды, г/г;
    :param p_units: единицы измерения давления

    :return: значение забойного давления
    """

    return calc_p_distribution(
        h=h,
        p=p,
        t_grad=t_grad,
        q=q,
        d=d,
        angle=angle,
        roughness=roughness,
        ws=ws,
        p_units=p_units
    )[0][-1]


def calc_vlp(
        q: NDArray,
        h: ArrayLike,
        p: float,
        t_grad: Callable,
        d: float,
        angle: float,
        roughness: float,
        ws: float,
        p_units: Optional[Literal['MPa', 'bar']] = 'MPa',
) -> tuple[ArrayLike, ArrayLike]:
    """
    Функция для построения VLP нагнетательной скважины.

    :param q: расход флюида в трубе, м3/с;
    :param h: глубина, м;
    :param p: давление в начале трубы, МПа;
    :param t_grad: градиент температуры по трубе;
    :param d: диаметр трубы, м;
    :param angle: угол искривления трубы, град;
    :param roughness: шероховатость трубы, м;
    :param ws: соленость воды, г/г;
    :param p_units: единицы измерения давления

    :return: VLP нагнетательной скважины
    """

    p_wf_array = np.vectorize(calc_pwf, excluded=['h'])

    return p_wf_array(h=h, p=p, t_grad=t_grad, q=q, d=d, angle=angle, roughness=roughness, ws=ws, p_units=p_units), q


def main(
        input_path: Union[str, PathLike],
) -> None:
    """
    Функция выполнения домашнего задания.

    :param input_path: путь к .json файлу с исходными данными, должен содержать в себе ключи:
            gamma_water - Относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
            md_vdp - Измеренная глубина верхних дыр перфорации, м
            d_tub - Диаметр НКТ, м
            angle - Угол наклона скважины к горизонтали, м
            roughness - Шероховатость, м
            p_wh - Буферное давление, атм
            t_wh - Температура жидкости у буферной задвижки, м
            temp_grad - Геотермический градиент, градусы цельсия/100 м

    :return: None
    """

    with open(input_path, 'r', encoding='utf-8') as input_file:
        input_data = json.load(input_file)

    water_dens = input_data.get('gamma_water')
    md_vdp = input_data.get('md_vdp')
    d_tub = input_data.get('d_tub') / 9.8692327  # перевод из атм в МПа
    angle = input_data.get('angle')
    roughness = input_data.get('roughness')
    p_wh = input_data.get('p_wh') / 10  # ошибка в исх данных
    t_wh = input_data.get('t_wh') + 273.15  # перевод из градусов цельсия в кельвины
    temp_grad = input_data.get('temp_grad')

    h_md = np.linspace(0, md_vdp, 10_000)
    t = interp1d(h_md, t_wh + h_md / 100 * temp_grad, fill_value='extrapolate')

    p_wf, q_liq = calc_vlp(
        q=np.linspace(1 / 86400, 400 / 86400, 100),  # перевод из м3/сут в м3/с
        h=[0, md_vdp],
        p=p_wh,
        t_grad=t,
        d=d_tub,
        angle=angle,
        roughness=roughness,
        ws=calc_ws(water_dens * 1000),
    )

    p_wf = p_wf * 9.8692327  # перевод из МПа обратно в атм
    q_liq = q_liq * 86400  # перевод из м3/с в м3/сут

    with open('output.json', 'w', encoding='utf-8') as output_file:
        json.dump({"q_liq": q_liq.tolist(), "p_wf": p_wf.tolist()}, output_file, indent=4)


if __name__ == '__main__':
    main('17.json')
