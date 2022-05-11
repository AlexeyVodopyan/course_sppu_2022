# Загрузка библиотек необходимых для отрисовки графиков
import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory
from unifloc.well.gaslift_well import GasLiftWell
import time
from tqdm import tqdm

with open('23-1.json', 'r')  as file1:
    data1 = json.load(file1)
with open('23-2.json', 'r')  as file2:
    data2 = json.load(file2)

gamma_oil = data1['fluid']['gamma_oil']
gamma_water = data1['fluid']['gamma_water']
gamma_gas = data1['fluid']['gamma_gas']
wct = data1['fluid']['wct']
rp = data1['fluid']['rp']
d_tubing = data1['pipe']['tubing']['d']
md_tubing = data1['pipe']['tubing']['md']
roughness_tubing = data1['pipe']['tubing']['roughness']
d_casing = data1['pipe']['casing']['d']
roughness_casing = data1['pipe']['casing']['roughness']
md_inclinometry = data1['inclinometry']['md']
tvd_inclinometry = data1['inclinometry']['tvd']
t_res = data1['temperature']['t_res'] #Температура пласта, градусы цельсия
temp_grad = data1['temperature']['temp_grad'] #Геотермический градиент, градусы цельсия/100 м
md_vdp = data1['reservoir']['md_vdp'] #Измеренная глубина верхних дыр перфорации, м
p_res = data1['reservoir']['p_res'] #Пластовое давление, атм
pi = data1['reservoir']['pi'] #Коэффициент продуктивности скважины, м3/сут/атм
p_wh = data1['regime']['p_wh'] #Буферное давление, атм
md_valve = data2['md_valve'] #Глубина клапана (точки ввода газа), м

def calc_ipr(p_res, p_wf, pi):
    return pi * (p_res - p_wf)

p_wf = np.linspace(0, p_res, 150) #забойное давление

q_liq = [calc_ipr(p_res, p_wf_i, pi) for p_wf_i in p_wf]

fig, ax = plt.subplots()
plt.plot(q_liq, p_wf)
ax.grid()
plt.show()

tvd_vdp = tvd_inclinometry[-1]
def calc_amb_temp(t_res, tvd_vdp, temp_grad, tvd_h):
    return t_res - temp_grad * (tvd_vdp - tvd_h) / 100 + 273.15

t = []
for depth in tvd_inclinometry:
    t.append(calc_amb_temp(t_res, tvd_vdp, temp_grad, depth))

fluid = FluidFlow(q_fluid = 100/86400, wct = wct,
                   pvt_model_data = {"black_oil":
                                   {"gamma_gas": gamma_gas,
                                    "gamma_wat": gamma_water,
                                    "gamma_oil": gamma_oil,
                                    "rp": rp}})

inclinometry = {"MD": md_inclinometry,
               "TVD": tvd_inclinometry}

traj = Trajectory(inclinometry = inclinometry)

amb_dist = {"MD": md_inclinometry,
            "T": t}

amb = AmbientTemperatureDistribution(ambient_temperature_distribution=amb_dist)

casing = Pipeline(
top_depth = md_tubing,
bottom_depth = md_vdp,
d = d_casing,
roughness = roughness_casing,
fluid = fluid,
trajectory = traj,
ambient_temperature_distribution = amb)


tubing = Pipeline(
top_depth = 0,
bottom_depth = md_tubing,
d = d_tubing,
roughness = roughness_tubing,
fluid = fluid,
trajectory = traj,
ambient_temperature_distribution = amb)

gaslift = GasLiftWell(
        fluid_data = {'q_fluid': 100/86400,
                    'wct': wct,
                    'pvt_model_data': {'black_oil': {'gamma_gas': gamma_gas,
                                                    'gamma_wat': gamma_water,
                                                    'gamma_oil': gamma_oil,
                                                    'rp': rp}}},
        pipe_data = {'casing': {'bottom_depth': md_vdp,
                                'd': d_casing,
                                'roughness': roughness_casing},
                    'tubing': {'bottom_depth': md_tubing,
                               'd': d_tubing,
                               'roughness': roughness_tubing}},
        well_trajectory_data = {'inclinometry': {'MD': md_inclinometry,
                                               'TVD': tvd_inclinometry}},
        ambient_temperature_data = {'MD': md_inclinometry,
                                  'T': t},
        equipment_data = {'gl_system': {'valve1': {'h_mes': md_valve,
                                                 'd': 0.006}}})

p_wf = np.linspace(0.1, p_res, 150)  # забойное давление
pt_wh = np.zeros(len(p_wf))
q_liq = np.zeros(len(p_wf))
for i in range(len(p_wf)):
    q_liq[i] = calc_ipr(p_res, p_wf[i], pi)
    pt = casing.calc_pt(  # эксплуатационная
        h_start='bottom',
        p_mes=p_wf[i] * 101325,
        flow_direction=-1,
        q_liq=q_liq[i] / 86400,
        extra_output=True)

    pt_wh[i] = tubing.calc_pt(  # НКТ
        h_start='bottom',
        p_mes=pt[0],
        flow_direction=-1,
        q_liq=q_liq[i] / 86400,
        extra_output=True)[0]

fig, ax = plt.subplots()
plt.plot(casing.distributions['p']/101325, casing.distributions['depth'] * (-1))
plt.plot(tubing.distributions['p']/101325, tubing.distributions['depth'] * (-1))
ax.grid()

fig, ax = plt.subplots()

ax.plot(q_liq, np.array(pt_wh)/101325, label='Давление на устье')
ax.plot(q_liq, [data1["regime"]["p_wh"]]*len(q_liq), label='Заданное буферное давление')

idx = np.argwhere(np.diff(np.sign(pt_wh/101325 - [data1["regime"]["p_wh"]]*len(q_liq)))).flatten()
ax.plot(q_liq[idx], (pt_wh/101325)[idx], 'ro')
ax.grid()
ax.set_xlabel('Дебит жидкости')
ax.set_ylabel('Давление')
ax.legend()
plt.show()

q_liq_intersection = q_liq[idx]
p_wh_intersection = (pt_wh/101325)[idx]

pt = casing.calc_pt(
        h_start = 'top',
        p_mes = p_wh_intersection*101325,
        flow_direction = 1,
        q_liq = q_liq_intersection/86400,
        extra_output = True)

p_wf_result = tubing.calc_pt(
        h_start = 'top',
        p_mes = pt[0],
        flow_direction = 1,
        q_liq = q_liq_intersection/86400,
        extra_output = True)[0]

p_wf_result =  p_wf_result/101325
print('p_wf_result =', p_wf_result) # минимальное забойное давление фонтанирования

def intersection(x, y1, y2): 
    first_line = LineString(np.column_stack((x, y1)))
    second_line = LineString(np.column_stack((x, y2)))
    intersection = first_line.intersection(second_line)
    x_intersect, y_intersect = intersection.xy
    return x_intersect[0], y_intersect[0]

gaslift_value = []
q_liq_ = [calc_ipr(p_res, p_wf_i, pi) for p_wf_i in p_wf]

for q_inj in tqdm(range(5000, 150000, 2000)):
    vlp = []
    for q_liq_i in q_liq_:
        vlp.append(gaslift.calc_pwf_pfl(p_fl = p_wh * 101325,
                                        q_liq = q_liq_i / 86400,
                                        wct = wct,
                                        q_gas_inj = q_inj / 86400) / 101325)
    gaslift_value.append(intersection(q_liq_, p_wf, vlp)[0])

q_liq_opt = max(gaslift_value)
q_inj_opt = range(5000, 150000, 2000)[gaslift_value.index(max(gaslift_value))]

fig, ax = plt.subplots()
plt.plot(range(5000, 150000, 2000), gaslift_value)
plt.plot(q_inj_opt, q_liq_opt, 'ro')
ax.grid()
plt.show()

with open('output.json', 'w', encoding='utf-8') as file:
    json.dump({"t1": {"pwf": p_wf_result}, "t2": {"q_inj": q_inj_opt, "q_liq": q_liq_opt}}, file, indent='\t')