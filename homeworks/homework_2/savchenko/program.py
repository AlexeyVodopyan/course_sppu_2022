from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory
from shapely.geometry import LineString
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# 1. Определение оптимального режима работы фонтанной скважины
data = pd.read_json\
    ('https://raw.githubusercontent.com/AlexeyVodopyan/course_sppu_2022/main/homeworks/homework_2/input_data/19-1.json')
p_res = data['reservoir']['p_res']
pi = data['reservoir']['pi']
t_res = data["temperature"]["t_res"]
tvd_vdp = data["inclinometry"]["tvd"][-1]
grad_t = data["temperature"]["temp_grad"]
t = []
p_wf = np.linspace(0, p_res, 10)
pt_wh, q_liq = np.zeros(10), np.zeros(10)


def calc_ipr(p_res, p_wf, pi):
    return pi * (p_res-p_wf)


def calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
    return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15


for depth in data["inclinometry"]["tvd"]:
    t.append(calc_amb_temp(t_res, tvd_vdp, grad_t, depth))

fluid = FluidFlow(q_fluid=100/86400, wct=data["fluid"]["wct"],
                   pvt_model_data={"black_oil":
                                   {"gamma_gas": data["fluid"]["gamma_gas"],
                                    "gamma_wat": data["fluid"]["gamma_water"],
                                    "gamma_oil": data["fluid"]["gamma_oil"],
                                    "rp": data["fluid"]["rp"]}})
inclinometry = {"MD": data["inclinometry"]["md"], "TVD": data["inclinometry"]["tvd"]}
traj = Trajectory(inclinometry=inclinometry)

amb_dist = {"MD": data["inclinometry"]["md"], "T": t}

amb = AmbientTemperatureDistribution(ambient_temperature_distribution=amb_dist)

casing = Pipeline(
top_depth = data["pipe"]["tubing"]["md"],
bottom_depth = data["reservoir"]["md_vdp"],
d = data["pipe"]["casing"]["d"],
roughness=data["pipe"]["casing"]["roughness"],
fluid=fluid,
trajectory=traj,
ambient_temperature_distribution=amb
)

tubing = Pipeline(
top_depth = 0,
bottom_depth =  data["pipe"]["tubing"]["md"],
d = data["pipe"]["tubing"]["d"],
roughness=data["pipe"]["tubing"]["roughness"],
fluid=fluid,
trajectory=traj,
ambient_temperature_distribution=amb
)

for i in range(len(p_wf)):
    q_liq[i] = calc_ipr(p_res, p_wf[i], pi)
    pt = casing.calc_pt(
        h_start='bottom',
        p_mes = p_wf[i] * 101325,
        flow_direction=-1,
        q_liq=q_liq[-1]/86400,
        extra_output=True
          )
    pt_wh[i] = tubing.calc_pt(
        h_start='bottom',
        p_mes = pt[0],
        flow_direction=-1,
        q_liq=q_liq[-1]/86400,
        extra_output=True
              )[0]

# Определение точки пересечения (минимального забойного давления фонтанирования)
plt.plot(q_liq, np.array(pt_wh)/101325)
plt.plot(q_liq, [data["regime"]["p_wh"]]*10)
# plt.show()
first_line = LineString(np.column_stack((q_liq, np.array(pt_wh)/101325)))
second_line = LineString(np.column_stack((q_liq, [data["regime"]["p_wh"]]*10)))
intersection = first_line.intersection(second_line)
x, y = intersection.xy
pwf = y[0]
plt.clf()

# 2. Определение оптимального режима работы газовой скважины
data_gas = pd.read_json\
    ('https://raw.githubusercontent.com/AlexeyVodopyan/course_sppu_2022/main/homeworks/homework_2/input_data/19-2.json',
     typ='series')

q_liq_array, q_inj_array =[], []
for k in range(10000, 110000, 10000):
    q_inj_array.append(k)
    casing = Pipeline(
            top_depth=data['pipe']['tubing']['md'],
            bottom_depth=data['reservoir']['md_vdp'],
            d=data['pipe']['casing']['d'],
            roughness=data['pipe']['casing']['roughness'],
            fluid=fluid,
            trajectory=traj,
            ambient_temperature_distribution=amb
        )
    tubing_lower = Pipeline(
            top_depth=data_gas['md_valve'],
            bottom_depth=data['pipe']['tubing']['md'],
            d=data['pipe']['tubing']['d'],
            roughness=data['pipe']['tubing']['roughness'],
            fluid=fluid,
            trajectory=traj,
            ambient_temperature_distribution=amb
        )
    fluid.q_gas_free = k / 86400
    tubing_upper = Pipeline(
            top_depth=0,
            bottom_depth=data_gas['md_valve'],
            d=data['pipe']['tubing']['d'],
            roughness=data['pipe']['tubing']['roughness'],
            fluid=fluid,
            trajectory=traj,
            ambient_temperature_distribution=amb
        )

    pt_casing = np.zeros(len(q_liq))

    for i in range(len(q_liq)):
        pt_tubing_upper = tubing_upper.calc_pt(
                h_start='top',
                p_mes=data['regime']['p_wh'] * 101325,
                flow_direction=-1,
                q_liq=q_liq[i] / 86400,
                extra_output=True
            )
        pt_tubing_lower = tubing_lower.calc_pt(
                h_start='top',
                p_mes=pt_tubing_upper[0],
                flow_direction=1,
                q_liq=q_liq[i] / 86400,
                extra_output=True
            )
        pt_casing[i] = casing.calc_pt(
                h_start='top',
                p_mes=pt_tubing_lower[0],
                flow_direction=1,
                q_liq=q_liq[i] / 86400,
                extra_output=True
                )[0]

    plt.plot(q_liq, np.array(pt_casing) / 101325)
    plt.plot(q_liq, p_wf)
    # plt.show()
    first_line_g = LineString(np.column_stack((q_liq, np.array(pt_casing) / 101325)))
    second_line_g = LineString(np.column_stack((q_liq, p_wf)))
    intersection_g = first_line_g.intersection(second_line_g)
    x, y = intersection_g.xy
    q_liq_array.append(x[0])

q_liq = max(q_liq_array)
q_inj_id = q_liq_array.index(q_liq)
q_inj = q_inj_array[q_inj_id]

# Заносим резы в файл-итог
result = dict(t1 = dict(pwf = pwf), t2 = dict(q_inj = q_inj, q_liq = q_liq))

with open('C:/Users/Vladislav/Desktop/output.json', 'w') as outfile:
    outfile.write(json.dumps(result, indent=4))

