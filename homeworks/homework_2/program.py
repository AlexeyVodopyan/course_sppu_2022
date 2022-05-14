#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from unifloc.pipe._beggsbrill import BeggsBrill
from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory
from unifloc.well.gaslift_well import GasLiftWell
import matplotlib.pyplot as plt
import seaborn as sns
import json
from shapely.geometry import LineString


sns.set_theme()
sns.set(rc={'figure.figsize':(15,10)})


# # import data

# In[2]:


filename = 'input_data\\15-1.json'
filename_gl = 'input_data\\15-2.json'

with open(filename) as f:
    data = json.load(f)
with open(filename_gl) as f:
    data_gl = json.load(f)

data


# # IPR

# In[3]:


def calc_ipr(p_res, p_wf, pi):
    return pi* (p_res-p_wf)

pi = data['reservoir']['pi']
p_res = data['reservoir']['p_res']
p_wf = np.linspace(0.1, p_res, 100)
q_ipr = calc_ipr(p_res, p_wf, pi)


# # Фонтан

# In[4]:


def calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
    return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15


# In[5]:


# температура
t = []
t_res = data["temperature"]["t_res"]
tvd_vdp = data["inclinometry"]["tvd"][-1]
grad_t = data["temperature"]["temp_grad"]

for depth in data["inclinometry"]["tvd"]:
    t.append(calc_amb_temp(t_res, tvd_vdp, grad_t, depth))


# In[6]:


fluid =  FluidFlow(q_fluid=100/86400, wct=data["fluid"]["wct"],
                   pvt_model_data={"black_oil": 
                                   {"gamma_gas": data["fluid"]["gamma_gas"], 
                                    "gamma_wat": data["fluid"]["gamma_water"], 
                                    "gamma_oil": data["fluid"]["gamma_oil"],
                                    "rp": data["fluid"]["rp"]}})
inclinometry = {"MD": data["inclinometry"]["md"],
               "TVD": data["inclinometry"]["tvd"]}
traj = Trajectory(inclinometry=inclinometry)

amb_dist = {"MD": data["inclinometry"]["md"],
            "T": t}

amb = AmbientTemperatureDistribution(ambient_temperature_distribution=amb_dist)

casing = Pipeline(
    top_depth = data["pipe"]["tubing"]["md"],
    bottom_depth = data["reservoir"]["md_vdp"],
    d = data["pipe"]["casing"]["d"],
    roughness=data["pipe"]["casing"]["roughness"],
    fluid=fluid,
    trajectory=traj,
    ambient_temperature_distribution=amb,
    )

tubing = Pipeline(
    top_depth = 0,
    bottom_depth =  data["pipe"]["tubing"]["md"],
    d = data["pipe"]["tubing"]["d"],
    roughness=data["pipe"]["tubing"]["roughness"],
    fluid=fluid,
    trajectory=traj,
    ambient_temperature_distribution=amb,
)


# In[7]:


def interpolated_intercepts(x, y1, y2):

    def intercept(point1, point2, point3, point4):
  
        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)


# In[8]:


pt_wh = np.zeros(len(p_wf))
for i in range(len(p_wf)):
    pt = casing.calc_pt(
             h_start='bottom',
             p_mes=p_wf[i]*101325,
             flow_direction=-1,
             q_liq=q_ipr[i]/86400,
             extra_output=True
         )
    pt_wh[i] = (tubing.calc_pt(
             h_start='bottom',
             p_mes=pt[0],
             flow_direction=-1,
             q_liq=q_ipr[i]/86400,
             extra_output=True
         )[0])

pt_wh = pt_wh / 101325
p_wh_regime = pt_wh*0 + data['regime']['p_wh']

min_q = interpolated_intercepts(q_ipr, pt_wh, p_wh_regime)[0]

min_q_array = pt_wh*0 + min_q[0]
min_p_wf = interpolated_intercepts(p_wf, q_ipr, min_q_array)[0]


# # Газлифт

# In[9]:


def find_intersection(x, y1, y2):
    first_line = LineString(np.column_stack((x, y1)))
    second_line = LineString(np.column_stack((x, y2)))
    intersection = first_line.intersection(second_line)
    x_intersect, y_intersect = intersection.xy
    return x_intersect[0], y_intersect[0]


gaslift_well = GasLiftWell(
         fluid_data={
             'q_fluid': 100 / 86000,
             'wct': data['fluid']['wct'],
             'pvt_model_data': {
                 'black_oil': {
                     'gamma_gas': data['fluid']['gamma_gas'],
                     'gamma_wat': data['fluid']['gamma_water'],
                     'gamma_oil': data['fluid']['gamma_oil'],
                     'rp': data['fluid']['rp']
                 }

             }
         },
         pipe_data={
             'casing': {'bottom_depth': data['reservoir']['md_vdp'],
                        'd': data['pipe']['casing']['d'],
                        'roughness': data['pipe']['casing']['roughness']},
             'tubing': {'bottom_depth': data['pipe']['tubing']['md'],
                        'd': data['pipe']['tubing']['d'],
                        'roughness': data['pipe']['tubing']['roughness']}
         },
         well_trajectory_data={'inclinometry': {'MD': data['inclinometry']['md'],
                                                'TVD': data['inclinometry']['tvd']}},
         ambient_temperature_data={'MD': data['inclinometry']['md'],
                                   'T': t},
         equipment_data={'gl_system': {'valve1': {'h_mes': data_gl['md_valve'],
                                                  'd': 0.006}}}
     )

q_liqs = q_ipr
q_gas_injs = range(5000, 150000, 10000)
gaslift_curve = []

for q_gas_inj in q_gas_injs:
    gaslift_vlp = []
    for q_liq in q_liqs:
         gaslift_vlp.append(
             gaslift_well.calc_pwf_pfl(
                 p_fl=data['regime']['p_wh'] * 101325,
                 q_liq=q_liq / 86400,
                 wct=data['fluid']['wct'],
                 q_gas_inj=q_gas_inj / 86400
             ) / 101325)
    gaslift_curve.append(
         find_intersection(q_liqs, p_wf, gaslift_vlp)[0]
     )

q_liq_optimal = max(gaslift_curve)
q_gas_optimal = q_gas_injs[gaslift_curve.index(max(gaslift_curve))]


# # Output

# In[10]:


dict_to_json = {   
                "t1": {
                    "pwf": min_p_wf[0][0]
                },
                "t2": {
                "q_inj": q_gas_optimal,
                "q_liq": q_liq_optimal
                }
            }

filename = 'output.json'

with open(filename, 'w') as f:
    json.dump(dict_to_json, f, indent=4)


