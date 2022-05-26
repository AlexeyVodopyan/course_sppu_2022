#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pandas as pd


# ### Домашнее Задание №1
# 
# Дедлайн: 01.04.2022
# 
# #### Моделирование распределения давления в нагнетательной скважине.
# 
# Реализовать программу по расчёту забойного давления нагнетательной скважины и построить зависимость 
# забойного давления, атм от дебита закачиваемой жидкости, м3/сут (VLP).
# 
# При расчёте учитывайте, что температура меняется согласно геотермическому градиенту (нет теплопотерь).
# В скважину спущена НКТ до верхних дыр перфорации, угол искривления скважины постоянный.
# Солёность зависит от входной плотности.
# Диапазон дебитов жидкости для генерации VLP `1 - 400 м3/сут`.
# 
# Результатом выполнения задания является *pull-request*,
# содержащий в себе:
# - `program.py`- файл программы
# - `output.json` - результат расчёта 
# Формат выходного файла доступен [по ссылке](../../homeworks/homework_1/output_example.json)
# - Файлы нужно положить в папку с фамилией в директорию `homeworks/homework_1/ваша_фамилия`. 
# Например, `homeworks/homework_1/vodopyan`
# 
# Исходные данные нужно взять [по ссылке](https://github.com/AlexeyVodopyan/course_sppu_2022/tree/main/homeworks/homework_1/input_data). 
# Нужный файл = вашему номеру в списке группы.
# 
# Если непонятны сокращения, то расшифровку можно посмотреть [здесь](glossary.md).
# 
# Для реализации можно использовать [заготовку расчётного файла](../../homeworks/homework_1/demo.ipynb). 

# In[ ]:


fp = open(r'..\..\input_data\13.json')


# In[ ]:


input_data = json.load(fp)
input_data


# ## Коэффициент трения 
# 
# Коэффициент трения Муди $f$ расчитывается для ламинарного потока по формуле 
# 
# $$ f= \frac{64}{Re}, Re < 3000 $$
# 
# При закачке воды поток не бывает ламинарным (для НКТ с внутренним диаметром 89 мм дебит воды при котором нарушается ламинарность потока составляет около 3 м3/сут)
# Для турбулентного режима течения $ Re > 3000 $ коэффициент трения Муди может быть рассчитан по Джейн (3. Swamee, P.K.; Jain, A.K. (1976). "Explicit equations for pipe-flow problems". Journal of the Hydraulics Division. 102 (5): 657–664)
# 
# $$ f = \frac{1} {\left [  1.14 - 2  \log \left ( \frac{ \epsilon} {d } + \frac{ 21.25}  { Re ^ {0.9} } \right ) \right ]  ^ 2} $$
# 
# или расчет может быть произведен для любых значений числа Рейнольдса $Re$ с использованием корреляции Черчилля (1974)
# 
# $$ f =  8  \left[ \left( \frac{8}{Re} \right ) ^{12} + \frac{1}{(A+B)^{1.5}} \right ] ^ {\frac{1}{12}} $$
# 
# где
# 
# $$ A = \left [- 2.457 \ln \left ( { \left(\frac{7}{Re} \right) ^{0.9} + 0.27 \frac{\epsilon} {d} } \right) \right ] ^{16} $$ 
# 
# $$ B = \left( \frac{37530}{Re} \right) ^{16}  $$
# 
# $\epsilon$ - шероховатость, м. 
# 
# Для НКТ часто берут $\epsilon = 0.000018 м$. Вообще, диапазон изменения значений шероховатости $\epsilon = [0.000015 - 0.000045] м$
# 
# Решение уравнения на распределение давления и температуры в стволе скважины. Решается система двух уравнений вида
# 
# $$ \frac{dP}{dL} = \frac{1}{10^{-5}} \left [  \rho g  \cos \alpha  - 0.815 \frac{f \rho}{d^5} q ^ 2  \right ]  $$
# 
# $$ \frac{dT}{dL} = geograd $$
# 
# Граничные условия задаются на устье скважины

# In[ ]:


# Загрузка библиотек необходимых для отрисовки графиков
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp


# In[ ]:


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
def Re(q_m3day, d_m, mu_mPas = 0.2, rho_kgm3 = 1000):
    # q_m3day - дебит жидкости, м3/сут
    # rho_kgm3 - плотность воды или жидкости, по умолчанию 1000 кг/м3, чистая вода
    # mu_mPas  - вязкость жидкости по умолчанию 0.2 мПас
    # d_m      - диаметр трубы, м
    v_ms = q_m3day/ 86400 / 3.1415 * 4 / d_m ** 2
    return rho_kgm3 * v_ms * d_m / mu_mPas * 1000

def friction_Jain(q_m3day,d_m = 0.089, mu_mPas = 0.2,rho_kgm3 = 1000,roughness=0.000018):
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    if Re_val < 3000:
        return 64/Re_val
    else:
        return 1/(1.14-2 * np.log10(roughness/d_m + 21.25 / (Re_val ** 0.9)))**2
    
def friction_Churchill(q_m3day,d_m = 0.089, mu_mPas = 0.2,rho_kgm3 = 1000,roughness=0.000018):
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    A = (-2.457 * np.log((7/Re_val)**(0.9)+0.27*(roughness/d_m)))**16
    B = (37530/Re_val)**16
    return 8 * ((8/Re_val)**12+1/(A+B)**1.5)**(1/12)

def pressure_gradient_MPam(q_m3day, P_Mpa, T_C, d_m, rho_water_sc_kgm3, cos_alpha, roughness):
    # q_m3day - дебит жидкости, м3/сут
    # P_Mpa - давление, МПа
    # T_C - температура, С
    # d_m - диаметр 
    # rho_water_sc_kgm3 -  
    # cos_alpha - косинус угол отклонения от вертикали 
    # roughness_m - шероховатость
    rho_kgm3 = rho_w_kgm3(P_Mpa, T_C + 273,
                          #salinity_gg(rho_water_sc_kgm3)
                          rho_w_sc_kgm3 = rho_water_sc_kgm3
                         ) # соленость не нужна, т.к. уже в sc
    mu_cP = visc_w_cP(P_Mpa, T_C + 273,
                      salinity_gg(rho_water_sc_kgm3)
                     )
    f = friction_Jain(q_m3day, d_m, mu_cP, rho_kgm3, roughness)
    g = 9.81
    q_m3sec = q_m3day /86400
    return (rho_kgm3 * g * cos_alpha - 0.815 * f * rho_kgm3 /( d_m ** 5) * (q_m3sec )**2) / 1000000


# In[ ]:


input_data


# In[ ]:


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


# In[ ]:


def dPTdL(PT, h):
    dPdL = pressure_gradient_MPam(Q, PT[0], PT[1], dtub_m , yw * 1000, Cos_Ang_fl(h), roughness_m)
    dTdL = TempGrad * Cos_Ang_fl(h)
    return [dPdL, dTdL]

# задаем граничные условия
PTwh = [Pwh,Twh]
# определяем диапазон и интервалы интегрирования
hs = np.linspace(0, H, 100)
# решаем систему уравнений численно
PTs = odeint(dPTdL, PTwh, hs)
# созраняем результаты расчета
P = PTs[:,0]
T = PTs[:,1]


# In[ ]:


plt.plot(P,hs, label ="давление")
plt.plot(T,hs, label = "температура")
plt.xlabel("P, Т")
plt.ylabel("h, длина скважины, м")
ax = plt.gca()
ax.invert_yaxis()
plt.legend()
plt.title("Распределение давления");


# In[ ]:


def dPTdL(PT, h, Q=Q):
        dPdL = pressure_gradient_MPam(Q, PT[0], PT[1], dtub_m , yw*1000, Cos_Ang_fl(h), roughness_m)
        dTdL = TempGrad * Cos_Ang_fl(h)
        return [dPdL, dTdL]

def calc_p_wf_mpa(Q):
    
    # задаем граничные условия
    PTwh = [Pwh,Twh]
    # определяем диапазон и интервалы интегрирования
    hs = np.linspace(0, H, 10)
    # решаем систему уравнений численно
    PTs = odeint(dPTdL, PTwh, hs, args = (Q, ))
    # созраняем результаты расчета
    P = PTs[:,0]
    T = PTs[:,1]
    return P[-1]

calc_p_wf_mpa(Q)


# In[ ]:


res_list = []
q_range_m3day = np.arange(1, 400, dtype=int)
for thiq_q_liq_m3day in q_range_m3day:
    this_p_wf_mpa = calc_p_wf_mpa(thiq_q_liq_m3day)
    this_p_wf_atm = this_p_wf_mpa * 9.86
    res_list.append(round(this_p_wf_atm, 4))


# In[ ]:


plt.plot(q_range_m3day, res_list)
plt.xlabel('Q, m3/day')
plt.ylabel('P, atm')
plt.show()


# In[ ]:


res_list[0], res_list[-1]


# In[ ]:


res_json = {"q_liq": [int(x) for x in q_range_m3day], 'p_wf' : res_list}
with open('output.json', 'w') as oj:
    json.dump(res_json, oj)


# In[ ]:


res_json


# In[ ]:





# In[ ]:




