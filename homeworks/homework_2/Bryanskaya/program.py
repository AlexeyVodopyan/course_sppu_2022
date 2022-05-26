from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory
from unifloc.well.gaslift_well import GasLiftWell
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString
import json


# Задание 1
with open("4-1.json") as f:
    data = json.load(f)

t_res = data["temperature"]["t_res"] #Температура пласта, градусы
tvd_vdp = data["inclinometry"]["tvd"][-1]
md_vdp = data["inclinometry"]["md"][-1]
grad_t = data["temperature"]["temp_grad"]
gamma_gas = data["fluid"]["gamma_gas"]
gamma_wat = data["fluid"]["gamma_water"]
gamma_oil = data["fluid"]["gamma_oil"]
rp = data["fluid"]["rp"] #Газовый фактор
md = data["inclinometry"]["md"] #Измеренная глубина по стволу
tvd = data["inclinometry"]["tvd"] #Вертикальная глубина
d = data["pipe"]["casing"]["d"]
roughness = data["pipe"]["casing"]["roughness"]
wct = data["fluid"]["wct"]
p_res = data["reservoir"]["p_res"]
pi = data["reservoir"]["pi"]
p_wh = data["regime"]["p_wh"] #буферное давление
d_tubing = data['pipe']['tubing']['d']
d_casing = data['pipe']['casing']['d']
roughness_tubing = data['pipe']['tubing']['roughness']
roughness_casing = data['pipe']['casing']['roughness']
md_tubing = data['pipe']['tubing']['md']

n = 100
p_wf = np.linspace(0, p_res, n)

# Расчет дебита от забойного давление
def calc_ipr(p_res, p_wf, pi):
    return pi * (p_res - p_wf)

# Расчет температуры на устье
def calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
    return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15

# Распределение температуры с глубиной
t = []
for depth in tvd:
    t.append(calc_amb_temp(t_res, tvd_vdp, grad_t, depth))
amb_dist = {"MD": md,
            "T": t}

# Расчет свойств флюида
fluid = FluidFlow(
    q_fluid=100 / 86400,
    wct=wct,
    pvt_model_data={
        "black_oil":
            {"gamma_gas": gamma_gas,
             "gamma_wat": gamma_wat,
             "gamma_oil": gamma_oil,
             "rp": rp}})
inclinometry = {"MD": md,
                "TVD": tvd}
traj = Trajectory(inclinometry=inclinometry)
amb = AmbientTemperatureDistribution(ambient_temperature_distribution=amb_dist)
casing = Pipeline(
    top_depth=md_tubing,
    bottom_depth=md_vdp,
    d=d_casing,
    roughness=roughness_casing,
    fluid=fluid,
    trajectory=traj,
    ambient_temperature_distribution=amb)

tubing = Pipeline(
    top_depth=0,
    bottom_depth=md_tubing,
    d=d_tubing,
    roughness=roughness_tubing,
    fluid=fluid,
    trajectory=traj,
    ambient_temperature_distribution=amb)

# Давление по стволу
q_liq = np.zeros(n)
pt_wh = np.zeros(n)
for i in range(n):
    q_liq[i] = calc_ipr(p_res, p_wf[i], pi)
    pt = casing.calc_pt(
        h_start='bottom',
        p_mes=p_wf[i] * 101325,
        flow_direction=-1,
        q_liq=q_liq[i]/86400,
        extra_output=True)
    pt_wh[i] = tubing.calc_pt(
        h_start='bottom',
        p_mes=pt[0],
        flow_direction=-1,
        q_liq=q_liq[i]/86400,
        extra_output=True
              )[0]


fig, ax = plt.subplots()

ax.plot(q_liq, np.array(pt_wh)/101325, label='Давление на устье')
ax.plot(q_liq, [data["regime"]["p_wh"]]*len(q_liq), label='Заданное буферное давление')
# Поиск оптимального дебита
idx = np.argwhere(np.diff(np.sign(pt_wh/101325 - [data["regime"]["p_wh"]]*n))).flatten()
ax.plot(q_liq[idx], (pt_wh/101325)[idx], 'ro')
ax.grid()
ax.set_xlabel('Дебит жидкости')
ax.set_ylabel('Давление')
ax.legend()

q_liq_intersection = q_liq[idx]
p_wh_intersection = (pt_wh/101325)[idx]
#plt.show()

# Пересчитываем с найденным дебитом
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

# Задание 2
with open("4-2.json") as f:
    data_gl = json.load(f)

md_valve = data_gl["md_valve"]

well = GasLiftWell(
        fluid_data={'q_fluid': 100/86400,
                    'wct': wct,
                    'pvt_model_data': {'black_oil': {'gamma_gas': gamma_gas,
                                                    'gamma_wat': gamma_wat,
                                                    'gamma_oil': gamma_oil,
                                                    'rp': rp}}},
        pipe_data={'casing': {'bottom_depth': md_vdp,
                               'd': d_casing,
                                'roughness': roughness_casing},
                    'tubing': {'bottom_depth': md_tubing,
                               'd': d_tubing,
                               'roughness': roughness_tubing}},
        well_trajectory_data={'inclinometry': {'MD': md,
                                               'TVD': tvd}},
        ambient_temperature_data={'MD': md,
                                  'T': t},
        equipment_data={'gl_system': {'valve1': {'h_mes': md_valve,
                                                 'd': 0.006}}})
# Поиск пересечения
def find_intersection(x, y1, y2):
    first_line = LineString(np.column_stack((x, y1)))
    second_line = LineString(np.column_stack((x, y2)))
    intersection = first_line.intersection(second_line)
    x_intersect, y_intersect = intersection.xy
    return x_intersect[0], y_intersect[0]

from tqdm import tqdm
gaslift_curve = []
q_liq_ = [calc_ipr(p_res, p_wf_i, pi) for p_wf_i in p_wf]
q_gas_injs=range(5000, 150000, 2000)
for q_gas_inj in tqdm(q_gas_injs):
    print('Текущий расход газлифтного газа: ' + str(q_gas_inj) + ' м3/сут')
    gaslift_vlp = []
    for q_liq_i in q_liq_:
        gaslift_vlp.append(well.calc_pwf_pfl(p_fl = p_wh * 101325,
                                                    q_liq = q_liq_i / 86400,
                                                    wct = wct,
                                                    q_gas_inj=q_gas_inj / 86400) / 101325)
    gaslift_curve.append(find_intersection(q_liq_, p_wf, gaslift_vlp)[0])
# Нахождение оптимального значения
q_liq_optimal = max(gaslift_curve)
q_gas_optimal = q_gas_injs[gaslift_curve.index(max(gaslift_curve))]

fig, ax = plt.subplots()
plt.plot(range(5000, 150000, 2000),gaslift_curve)
plt.plot(q_gas_optimal, q_liq_optimal, 'ro')
ax.grid()
plt.show()

with open('output.json', 'w', encoding='utf-8') as file:
    json.dump({"t1": {"pwf": p_wf_result}, "t2": {"q_inj": q_gas_optimal, "q_liq": q_liq_optimal}}, file, indent='\t')

