#!/usr/bin/env python
# coding: utf-8

# In[46]:


import seaborn as sns
import pandas as pd
import json
import numpy as np
from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory
import matplotlib.pyplot as plt
from __future__ import division 
from unifloc.well.gaslift_well import GasLiftWell
from shapely.geometry import LineString
import scipy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
sns.set_theme()

print('FIRST TASK:')
# функция поиска пересечений
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

filename_1 = '/Users/alexander/Desktop/Study/DZ/22-1.txt'
falename_2 = '/Users/alexander/Desktop/Study/DZ/22-2.txt'
with open(filename_1) as f:
    data = json.load(f)
with open(falename_2) as f:
    data1 = json.load(f)
    
def calc_ipr(p_res, p_wf, pi):
    return pi * (p_res - p_wf)
    
p_res = data['reservoir']['p_res']
pi = data['reservoir']['pi']
p_wf = np.linspace(0, p_res, 100)
q_liq = [calc_ipr(p_res, p_wf_i, pi) for p_wf_i in p_wf]
t_res = data['temperature']['t_res']
tvd_vdp = data['inclinometry']['tvd'][-1]
grad_t = data['temperature']['temp_grad']
p_wh = data['regime']['p_wh']

#ipr
q_liq = calc_ipr(p_res, p_wf, pi)
plt.figure(figsize=(10, 8))
plt.plot(q_liq, p_wf, color='blue')
plt.ylabel("Забойное давление, атм")
plt.xlabel("Расход, м3/сут")
plt.show()

fluid = FluidFlow(
         q_fluid=100/86400,
         wct=data['fluid']['wct'],
         pvt_model_data={"black_oil": {"gamma_gas": data['fluid']['gamma_gas'],
                                       "gamma_wat": data['fluid']['gamma_water'],
                                       "gamma_oil":  data['fluid']['gamma_oil'],
                                       "rp": data['fluid']['rp']}}
     )

inclinometry = {'MD': data['inclinometry']['md'], 'TVD': data['inclinometry']['tvd']}
traj = Trajectory(inclinometry=inclinometry)

def calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
     return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15
    
T = []
for depth in data['inclinometry']['tvd']:
         T.append(calc_amb_temp(t_res, tvd_vdp, grad_t, depth))
    
amb_dist = {'MD': data['inclinometry']['md'], 'T': T}
amb = AmbientTemperatureDistribution(ambient_temperature_distribution=amb_dist)

casing = Pipeline(
         top_depth=data['pipe']['tubing']['md'],
         bottom_depth=data['reservoir']['md_vdp'],
         d=data['pipe']['casing']['d'],
         roughness=data['pipe']['casing']['roughness'],
         fluid=fluid,
         trajectory=traj,
         ambient_temperature_distribution=amb
     )
tubing = Pipeline(
         top_depth=0,
         bottom_depth=data['pipe']['tubing']['md'],
         d=data['pipe']['tubing']['d'],
         roughness=data['pipe']['tubing']['roughness'],
         fluid=fluid,
         trajectory=traj,
         ambient_temperature_distribution=amb
     )

q_liq = np.zeros(len(p_wf))
pt_wh = np.zeros(len(p_wf))
for i in range(len(p_wf)):
    q_liq[i] = calc_ipr(p_res, p_wf[i], pi)
    pt = casing.calc_pt(
             h_start='bottom',
             p_mes=p_wf[i]*101325,
             flow_direction=-1,
             q_liq=q_liq[i]/86400,
             extra_output=True
         )
    pt_wh[i] = (tubing.calc_pt(
             h_start='bottom',
             p_mes=pt[0],
             flow_direction=-1,
             q_liq=q_liq[i]/86400,
             extra_output=True
         )[0])

pt_wh_arr = (np.array(pt_wh))/101325
limit_p = np.ones(len(pt_wh_arr))*[data['regime']['p_wh']]

X, Y = interpolated_intercepts(q_liq, pt_wh_arr, limit_p)

x = q_liq
y1 = pt_wh_arr
y2 = limit_p
plt.figure(figsize=(10, 8))
plt.plot(x, y1, marker='o', mec='none', ms=4, lw=1, label='y1', color='blue')
plt.plot(x, y2, marker='o', mec='none', ms=4, lw=1, label='y2', color='black')
plt.plot(X, Y, 'co', ms=5, label='Nearest data-point, with linear interpolation', color='red')
plt.xlabel("Расход, м3/сут")
plt.ylabel("Устьевое давление, атм")

#расход при минимально возможном устьевом давлении
min_q = X

min_q_arr = np.ones(len(pt_wh_arr))*min_q[0]
X1, Y1 = interpolated_intercepts(p_wf, q_liq, min_q_arr)

x = q_liq
y1 = p_wf
y2 = min_q_arr
plt.figure(figsize=(10, 8))
plt.plot(y1, x, marker='o', mec='none', ms=4, lw=1, label='y1', color='blue')
plt.plot(y1, y2, marker='o', mec='none', ms=4, lw=1, label='y2', color='black')
plt.plot(X1, Y1, 'co', ms=5, label='Nearest data-point, with linear interpolation', color='red')
plt.xlabel("Забойное давление, атм")
plt.ylabel("Расход, м3/сут")
plt.show()
print('Минимальное устьевое давление:', X1[0][0], 'атм')

print('SECOND TASK:')

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
                                   'T': T},
         equipment_data={'gl_system': {'valve1': {'h_mes': data1['md_valve'],
                                                  'd': 0.006}}}
     )

q_liqs = q_liq
q_gas_injs = range(5000, 150000, 10000)
gaslift_curve = []

for q_gas_inj in tqdm(q_gas_injs):
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
print(q_gas_optimal, q_liq_optimal)

delta1 = abs(3 - q_gas_optimal*3/100)
q_gas_injs_new = range( int(q_gas_optimal - delta1), int(q_gas_optimal + delta1), 400)
gaslift_curve_new = []

for q_gas_inj_new in tqdm(q_gas_injs_new):
    gaslift_vlp_new = []
    for q_liq in q_liqs:
         gaslift_vlp_new.append(
             gaslift_well.calc_pwf_pfl(
                 p_fl=data['regime']['p_wh'] * 101325,
                 q_liq=q_liq / 86400,
                 wct=data['fluid']['wct'],
                 q_gas_inj=q_gas_inj_new / 86400
             ) / 101325)
    gaslift_curve_new.append(
         find_intersection(q_liqs, p_wf, gaslift_vlp_new)[0]
     )
q_liq_optimal_new = max(gaslift_curve_new)
q_gas_optimal_new = q_gas_injs_new[gaslift_curve_new.index(max(gaslift_curve_new))]
print(q_gas_optimal_new, q_liq_optimal_new)

delta2 = abs(1 - q_gas_optimal_new*1/100)
q_gas_injs_new2 = range( int(q_gas_optimal_new - delta2), int(q_gas_optimal_new + delta2), 25)
gaslift_curve_new2 = []
q_gas_injs_new2

for q_gas_inj_new2 in tqdm(q_gas_injs_new2):
    gaslift_vlp_new2 = []
    for q_liq in q_liqs:
         gaslift_vlp_new2.append(
             gaslift_well.calc_pwf_pfl(
                 p_fl=data['regime']['p_wh'] * 101325,
                 q_liq=q_liq / 86400,
                 wct=data['fluid']['wct'],
                 q_gas_inj=q_gas_inj_new2 / 86400
             ) / 101325)
    gaslift_curve_new2.append(
         find_intersection(q_liqs, p_wf, gaslift_vlp_new2)[0]
     )
q_liq_optimal_new2 = max(gaslift_curve_new2)
q_gas_optimal_new2 = q_gas_injs_new2[gaslift_curve_new2.index(max(gaslift_curve_new2))]
plt.figure(figsize=(10, 8))
plt.plot(q_gas_injs_new, gaslift_curve_new, color='blue')
plt.plot(q_gas_optimal_new2, q_liq_optimal_new2, 'co', ms=5, color='green')
plt.plot(q_gas_optimal_new, q_liq_optimal_new, 'co', ms=5, color='black')
plt.plot(q_gas_optimal, q_liq_optimal, 'co', ms=5, color='red')
plt.show()
print(q_gas_optimal_new2, q_liq_optimal_new2)

plt.figure(figsize=(10, 8))
plt.plot(q_gas_injs, gaslift_curve, color='blue')
plt.plot(q_gas_optimal_new2, q_liq_optimal_new2, 'co', ms=5, color='green')
plt.show()

dict_to_json = {   
                "t1": {
                    "pwf": X1[0][0]
                },
                "t2": {
                "q_inj": q_gas_optimal_new2,
                "q_liq": q_liq_optimal_new2
                }
            }

filename = 'output.json'

with open(filename, 'w') as f:
    json.dump(dict_to_json, f)


# In[ ]:




