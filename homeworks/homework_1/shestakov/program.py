#!/usr/bin/env python
# coding: utf-8

# # Расчетная часть

# ![jupyter](Глоссарий1ДЗ.PNG)

# In[1]:


# Загрузка библиотек необходимых для отрисовки графиков
import matplotlib
import math
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import math as mt
from tqdm import tqdm 
import plotly.express as px

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Загрузка данных
df = pd.read_json('input.json', lines=True)
gamma_water = float(df.gamma_water[0])
# gamma_water = 0.9670655405969301
md_vdp = float(df.md_vdp[0])
d_tub = float(df.d_tub[0])/10
angle = float(df.angle[0])
roughness = float(df.roughness[0])
p_wh = float(df.p_wh[0])*0.101325
t_wh = float(df.t_wh[0]) + 273.15
temp_grad = float(df.temp_grad[0])
# check data
df.to_dict()


# In[3]:


# функция расчета плотности воды в зависимости от давления и температуры
def rho_w_kgm3(T_K, gamma_water):

    ws = salinity_gg(1000*gamma_water)
    ws = 0
    rho_w_sc_kgm33 = 1000*(1.0009 - 0.7114 * ws + 0.2605 * ws**2)**(-1)
#     rho_w_sc_kgm3 = 1000*gamma_water
#     print(rho_w_sc_kgm33 /(1+(T_K-273)/10000*(0.269*(T_K-273)**(0.637)-0.8)))
    return rho_w_sc_kgm33 /(1+(T_K-273)/10000*(0.269*(T_K-273)**(0.637)-0.8))

# функция расчета солености через плотсноть
def salinity_gg(rho_kgm3):
    sal = 1/rho_kgm3*(1.36545*rho_kgm3-(3838.77*rho_kgm3-2.009*rho_kgm3**2)**0.5)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if sal>0 :
        return sal
    else:
        return 0
    return sal
    
# Расчет вязкости воды в зависимости от температуры и давления
def visc_w_cP(P_Mpa,T_K, gamma_water):
    ws = salinity_gg(1000*gamma_water)
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
    v_ms = q_m3day/ 86400 / 3.1415 * 4 / d_m ** 2
    return rho_kgm3 * v_ms * d_m / mu_mPas * 1000

def friction_Jain(q_m3day,d_m, mu_mPas,rho_kgm3,roughness):
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    if Re_val < 3000:
        return 64/Re_val
    else:
        return 1/(1.14-2 * np.log10(roughness/d_m + 21.25 / (Re_val ** 0.9)))**2
    
def friction_Churchill(q_m3day,d_m, mu_mPas,rho_kgm3,roughness=0.000018):
    Re_val = Re(q_m3day,d_m,mu_mPas,rho_kgm3)
    A = (-2.457 * np.log((7/Re_val)**(0.9)+0.27*(roughness/d_m)))**16
    B = (37530/Re_val)**16
    return 8 * ((8/Re_val)**12+1/(A+B)**1.5)**(1/12)

h_md = np.linspace(0, md_vdp, 100) # h_md - глубина от 0 до глубины верхних дыр перофрации, м
temp = t_wh + h_md*temp_grad/100 # Температура на каждой глубине h_md
temp = interp1d(h_md, temp) 
# print(temp)

def dp_dh(h, p, q_liq, d_m, angle, roughness, gamma_water):
    
    t = temp(h)
    rho = rho_w_kgm3(t, gamma_water)
    mu = visc_w_cP(p, t, gamma_water)
                   
    f = friction_Churchill(q_liq, d_m, mu, rho, roughness)
    
    dp_dL_grav = rho*9.81*mt.sin(mt.pi*angle/180)
    
    dp_dL_fric = f*rho*(q_liq/86400)**2 / d_m**5
    
    # в МегаПаскали
    return (dp_dL_grav - (8/mt.pi**2)*dp_dL_fric)/1000000


# In[9]:


list_res = []
Q = list(range(1,401))
# Q = np.linspace(1,400,100)
for i in tqdm(Q):
    result = solve_ivp(dp_dh, t_span = [0, md_vdp], y0 = np.array([p_wh]), args = (i, d_tub, angle, roughness, gamma_water))
    list_res.append( (result.y[0][-1])*9.8692327 )
fig = px.scatter(x=Q, y=list_res).update_traces(mode='lines+markers') 
fig.update_layout(
    title="Зависимость забойного давления от дебита",
    yaxis_title="$$P, атм$$",
    xaxis_title="Q, $$м^3/cут$$",
#     legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="Black"
    )
)
fig.show()


# In[10]:


dict_to_json = {"q_liq": Q, "p_wf": list_res}


# In[11]:


filename = 'output.json'
with open(filename, 'w') as f:
    json.dump(dict_to_json, f)


# In[ ]:





# In[ ]:





# In[7]:


v_rho_w_kgm3 = np.vectorize(rho_w_kgm3)
v_temp = np.vectorize(temp)


# In[8]:


# температура
fig = px.scatter(x=np.linspace(0,md_vdp, 100), y=v_temp(np.linspace(0,md_vdp, 100))).update_traces(mode='lines+markers') 
fig.show()


# In[ ]:





# In[ ]:




