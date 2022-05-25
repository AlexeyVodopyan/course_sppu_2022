#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import matplotlib.pyplot as plt
import math as mt
from scipy.integrate import quad
from scipy.interpolate import interp1d


# In[2]:


name = '12.json'
with open(name) as f:
    input_data = json.load(f)


# In[3]:


input_data


# In[4]:


gamma_water = input_data['gamma_water']
h = input_data['md_vdp'] 
d_tub = input_data['d_tub'] / 10 
angle = input_data['angle']  
roughness = input_data['roughness'] 
P_wh = input_data['p_wh'] * 0.101325
T_wh = input_data['t_wh'] + 273.15
t_grad = input_data['temp_grad']


# In[5]:


def rho_w_kgm3(T_K, ws = 0):
    rho_w_sc_kgm3 = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
    rho_w_kgm3 = rho_w_sc_kgm3 / (1 + (T_K - 273) / 10000 * (0.269 * (T_K - 273)**(0.637) - 0.8))
    return rho_w_kgm3


# In[6]:


def salinity_gg(rho_kgm3):
    ws = 1/rho_kgm3*(1.36545*rho_kgm3-(3838.77*rho_kgm3-2.009*rho_kgm3**2)**0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws>0 :
        return ws
    else:
        return 0


# In[7]:


def visc_w_cP(P_Mpa, T_K, ws = 0):
    A = 109.574 - 0.8406 * 1000 * ws + 3.1331 * 1000 * ws * ws + 8.7221 * 1000 * ws * ws * ws
    B = 1.1217 - 2.6396 * ws + 6.7946 * ws * ws + 54.7119 * ws * ws * ws - 155.586 * ws * ws * ws * ws
    muw = A * (1.8 * T_K - 460) ** (-B) * (0.9994 + 0.0058 * P_Mpa + 0.6534 * (10) ** (-4) * P_Mpa * P_Mpa)
    return muw


# In[8]:


def Re(q_m3day, d_m, mu_mPas = 0.2, rho_kgm3 = 1000):
    v_ms = q_m3day / 86400 / 3.1415 * 4 / d_m ** 2
    Re = rho_kgm3 * v_ms * d_m / mu_mPas * 1000
    return Re


# In[9]:


def friction_Jain(q_m3day, d_m = 0.089, mu_mPas = 0.2, rho_kgm3 = 1000, roughness=0.000018):
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    if Re_val < 3000:
        f = 64 / Re_val
        return f
    else:
        f = 1 / (1.14 - 2 * np.log10(roughness / d_m + 21.25 / (Re_val**0.9)))**2
        return f


# In[10]:


def friction_Churchill(q_m3day, d_m = 0.089, mu_mPas = 0.2, rho_kgm3 = 1000, roughness=0.000018):
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    A = (-2.457 * np.log((7 / Re_val)**(0.9) + 0.27*(roughness / d_m)))**16
    B = (37530 / Re_val)**16
    f = 8 * ((8 / Re_val)**12 + 1 / (A + B)**1.5)**(1 / 12)
    return f


# In[11]:


h_md = np.linspace(0, h, 100) #набор значений глубины 100 точек
temp = T_wh + h_md*t_grad / 100 # Температура на каждой глубине h_md
temp = interp1d(h_md, temp) 


# In[12]:


def dp_dh(P_Mpa, h_m, q_m3day, d_m, angle, roughness, gamm_water):
    
    sal_w = salinity_gg(gamm_water*1000) #так как в условии указанл, что плотность относительная, домнажаем на 1000
    T_K = temp(h_m) #функция температуры от глубины
    rho = rho_w_kgm3(T_K, sal_w)
    mu = visc_w_cP(P_Mpa, T_K)
    
    f = friction_Churchill(q_m3day, d_m, mu, rho, roughness)
    dp_dl_grav = rho * 9.81 * mt.sin(angle * mt.pi / 180) 
    dp_dl_fric = f * rho * (q_m3day / 86400)** 2 / (d_m ** 5) 
    dp_dl = dp_dl_grav - (8 / mt.pi**2) * dp_dl_fric
    return dp_dl/1000000


# In[13]:


q_inj = [1]+[i for i in range(0,410,10)][1:]#вместо 0 ставим 1 по условию задачи диапазон 1-400
h_md = np.linspace(0, h, 41)
delta_h = h_md[1] - h_md[0]
P_out = []
for q in q_inj:
    P = [P_wh]
    for h_h in h_md:
        dp_dl = dp_dh(P[-1], h_h, q, d_tub, angle, roughness, gamma_water)
        delta_p = dp_dl * delta_h
        P.append(P[-1] + delta_p)
    P_out.append(P[-1])


# In[14]:


p_in_atm = list(np.array(P_out)*9.8692327)


# In[15]:


out_file = {"q": q_inj, "p_wf": p_in_atm}


# In[16]:


name = 'output.json'
with open(name,'w') as f:
    json.dump(out_file, f)


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


plt.plot(q_inj, p_in_atm)

