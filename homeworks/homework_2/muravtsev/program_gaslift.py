import numpy as np
import matplotlib.pyplot as plt
import json
from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory


def calc_ipr(p_res, p_wf, pi):
    return pi * (p_res - p_wf)


def calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
    return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15


if __name__ == '__main__':

    with open('../input_data/16-1.json') as f:
        data = json.load(f)

    with open('../input_data/16-2.json') as f1:
        data1 = json.load(f1)

    T = []
    t_res = data['temperature']['t_res']
    tvd_vdp = data['inclinometry']['tvd'][-1]
    grad_t = data['temperature']['temp_grad']

    for depth in data['inclinometry']['tvd']:
        T.append(calc_amb_temp(t_res, tvd_vdp, grad_t, depth))

    fluid = FluidFlow(
        q_fluid=100/86400,
        wct=data['fluid']['wct'],
        pvt_model_data={"black_oil": {"gamma_gas": data['fluid']['gamma_gas'],
                                      "gamma_wat": data['fluid']['gamma_water'],
                                      "gamma_oil":  data['fluid']['gamma_oil'],
                                      "rp": data['fluid']['rp']}}
    )

    diff = (data['inclinometry']['tvd'][2] - data['inclinometry']['tvd'][1]) - \
           (data['inclinometry']['md'][2] - data['inclinometry']['md'][1])
    if diff > 0:
        data['inclinometry']['md'][2:] = [i + 1.1 * diff for i in data['inclinometry']['md'][2:]]

    inclinometry = {'MD': data['inclinometry']['md'], 'TVD': data['inclinometry']['tvd']}
    traj = Trajectory(inclinometry=inclinometry)

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

    tubing_lower = Pipeline(
        top_depth=data1['md_valve'],
        bottom_depth=data['pipe']['tubing']['md'],
        d=data['pipe']['tubing']['d'],
        roughness=data['pipe']['tubing']['roughness'],
        fluid=fluid,
        trajectory=traj,
        ambient_temperature_distribution=amb
    )

    fluid.q_gas_free = 5 / 86400
    tubing_upper = Pipeline(
        top_depth=0,
        bottom_depth=data1['md_valve'],
        d=data['pipe']['tubing']['d'],
        roughness=data['pipe']['tubing']['roughness'],
        fluid=fluid,
        trajectory=traj,
        ambient_temperature_distribution=amb
    )

    p_res = data['reservoir']['p_res']
    pi = data['reservoir']['pi']
    q_liq = np.linspace(0.1, 150, 10)
    pt_casing = np.zeros(len(q_liq))

    for i in range(len(q_liq)):
        pt_tubing_upper = tubing_upper.calc_pt(
            h_start='top',
            p_mes=data['regime']['p_wh'],
            flow_direction=1,
            q_liq=q_liq[i]/86400,
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

    plt.plot(q_liq, pt_casing / 101325)
    plt.show()
