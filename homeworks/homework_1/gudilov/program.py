#!/usr/bin/env python
# coding: utf-8

# # Импорт библиотек

# In[1]:


import json
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.integrate import quad
from scipy.interpolate import interp1d
import plotly.express as px


# # Входные данные

# In[2]:


filename = '6.json'

with open(filename) as f:
    input_data = json.load(f)


# In[3]:


input_data


# In[4]:


h = input_data['md_vdp'] # Глубина верхних дыр перфорации, м
d_tub = input_data['d_tub'] / 10 # Диаметр НКТ, м

gamma_water = input_data['gamma_water'] # Относительная плотность воды, безразмерная

P_wh = input_data['p_wh'] * 0.101325 # Буферное давление, МПа
T_wh = input_data['t_wh'] + 273.15 # Температура жидкости у буферной линии, К

angle = input_data['angle'] # Угол наклона скважины к горизонтали
roughness = input_data['roughness'] # Шероховатость, м
t_grad = input_data['temp_grad'] # Температурный градиент


# # Функции

# In[5]:


# Функция расчета температуры от глубины
h_md = np.linspace(0, h, 100) # h_md - глубина от 0 до глубины верхних дыр перофрации, м
temp = T_wh + h_md*t_grad / 100 # Температура на каждой глубине h_md
temp = interp1d(h_md, temp) 


# In[6]:


def rho_w_kgm3(T_K, ws = 0):
    '''
    Функция расчета плотности воды. -> rho_w_kgm3 [кг/м^3]
    
    T_K - температура жидкости [K]
    ws - соленость жидкости [г/г]
    
    '''
    # Зависимость плотности от солености
    rho_w_sc_kgm3 = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
    
    # Зависимость плотности от температуры
    rho_w_kgm3 = rho_w_sc_kgm3 / (1 + (T_K - 273) / 10000 * (0.269 * (T_K - 273)**(0.637) - 0.8))
    
    return rho_w_kgm3


# In[7]:


def salinity_gg(rho_kgm3):
    """
    Фунцкия расчета солености. -> ws [г/г]
    
    rho_kgm3 - плотность жидкости [кг/м^3]
    
    """
    
    ws = 1/rho_kgm3*(1.36545*rho_kgm3-(3838.77*rho_kgm3-2.009*rho_kgm3**2)**0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws>0 :
        return ws
    else:
        return 0


# In[8]:


def visc_w_cP(P_Mpa, T_K, ws = 0):
    '''
    Функция расчета вязкости от давления и температуры. -> muw [сПз]
    
    P_Mpa - давление жидкости [МПа]
    T_K - температура жидкости [К]
    ws - соленость жидкости [г/г]
    
    '''
    
    A = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws * ws + 8.7221 * 1000 * ws * ws * ws
    B = 1.1217 - 2.6396 * ws + 6.7946 * ws * ws + 54.7119 * ws * ws * ws - 155.586 * ws * ws * ws * ws
    muw = A * (1.8 * T_K - 460) ** (-B) * (0.9994 + 0.0058 * P_Mpa + 0.6534 * (10) ** (-4) * P_Mpa * P_Mpa)
    return muw


# In[9]:


def Re(q_m3day, d_m, mu_mPas = 0.2, rho_kgm3 = 1000):
    '''
    Функция расчета числа Рейнольдса. -> Re
    q_m3day - дебит жидкости [м3/сут]
    rho_kgm3 - плотность воды или жидкости, по умолчанию 1000 кг/м3, чистая вода
    mu_mPas- вязкость жидкости по умолчанию 0.2 [мПа*с = сПз]
    d_m - диаметр трубы [м]
    
    '''
    # Расчет скорости потока [м/с]
    v_ms = q_m3day / 86400 / 3.1415 * 4 / d_m ** 2
    # Расчет числа Рейнольдса
    Re = rho_kgm3 * v_ms * d_m / mu_mPas * 1000
    return Re


# In[10]:


def friction_Jain(q_m3day, d_m = 0.089, mu_mPas = 0.2, rho_kgm3 = 1000, roughness=0.000018):
    '''
    Функция расчета коэффицента трения Муди по Джейн. -> f
        
    q_m3day - расход [м3/сут]
    d_m - диаметр трубы [м]
    mu_mPas - вязкость [мПа*с = сПз]
    rho_kgm3 = плотность жидкости [кг/м3]
    roughness - шероховатость [м]
    
    '''
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    
    # Ламинарный поток
    if Re_val < 3000:
        f = 64 / Re_val
        return f
    # Турбулентный поток
    else:
        f = 1 / (1.14 - 2 * np.log10(roughness / d_m + 21.25 / (Re_val**0.9)))**2
        return f


# In[11]:


def friction_Churchill(q_m3day, d_m = 0.089, mu_mPas = 0.2, rho_kgm3 = 1000, roughness=0.000018):
    """
    Функция расчета коэффициента трения по корреляции Черчилля. -> f
    
    q_m3day - расход [м3/сут]
    d_m - диаметр трубы [м]
    mu_mPas - вязкость [мПа*с = сПз]
    rho_kgm3 = плотность жидкости [кг/м3]
    roughness - шероховатость [м]
    
    """
    
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    A = (-2.457 * np.log((7 / Re_val)**(0.9) + 0.27*(roughness / d_m)))**16
    B = (37530 / Re_val)**16
    f = 8 * ((8 / Re_val)**12 + 1 / (A + B)**1.5)**(1 / 12)
    return f


# In[12]:


def dp_dh(P_Mpa, h_m, q_m3day, d_m, angle, roughness, gamm_water):
    """
    Функция расчета градиента давления. -> dp_dl [МПа/м]
    
    
    P_Mpa - давление [МПа]
    q_m3day - расход [м3/сут]
    h_m - глубина [м]
    d_m - диаметр трубы [м]
    angle - угол отклонения скважины от вертикали [град]
    roughness - шероховатость [м]
    gamm_water - относительная плотность воды, безразмерная [-]
    """
    sal_w = salinity_gg(gamm_water*1000) # Соленость воды от входной плотности
    T_K = temp(h_m) # Температура на каждой глубине temp = T_wh + h_md*t_grad/100
    rho = rho_w_kgm3(T_K, sal_w) # Плотность
    mu = visc_w_cP(P_Mpa, T_K) # Вязкость
    f = friction_Churchill(q_m3day, d_m, mu, rho, roughness) # Коэффициент трения
    
    dp_dl_grav = rho * 9.81 * mt.sin(angle * mt.pi / 180) # Потери давления гравитационные
    dp_dl_fric = f * rho * (q_m3day / 86400)** 2 / (d_m ** 5) # Потери давления из-за трения
    dp_dl = dp_dl_grav - (8 / mt.pi**2) * dp_dl_fric # Потери давления суммарное (градиент давления)
    return dp_dl/1000000


# # Расчет

# In[13]:


# Зависимость забойного давления от расхода нагнетальной скважины (от 1 до 400 м3/сут)
q_list = [1] + [i for i in range(0,410,10)][1:] # Дебит от 1 до 400

h_md = np.linspace(1, h, 2000)
delta_h = h_md[1] - h_md[0]
P_final = []
for q in q_list:
    P = [P_wh]
    for h_h in h_md:
        dp_dl = dp_dh(P[-1], h_h, q, d_tub, angle, roughness, gamma_water)
        delta_p = dp_dl * delta_h
        P.append(P[-1] + delta_p)
    P_final.append(P[-1])


# In[14]:


p_in_atm = list(np.array(P_final)*9.8692327) # Перевод забойного давления в атмосферы


# In[15]:


# График заивимости забойного давления [атм] от расхода воды в наг.скважину [м3/сут]
px.line(x = q_list, y = p_in_atm, markers=True).show() 


# In[16]:


# Забойное давлние в атмосферах
# p_in_atm


# In[17]:


# Расход м3/сут
# q_list


# # Загрузка в json

# In[18]:


# q_liq - Расход м3/сут [1,10,20,...,400]
# p_wf - Забойное давление атм.
dict_to_json = {"q_liq": q_list, "p_wf": p_in_atm}


# In[19]:


filename = 'output.json'
with open(filename, 'w') as f:
    json.dump(dict_to_json, f)

