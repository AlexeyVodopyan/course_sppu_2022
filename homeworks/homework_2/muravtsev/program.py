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

    p_res = data['reservoir']['p_res']
    pi = data['reservoir']['pi']
    p_wf = np.linspace(0.1, p_res, 10)
    q_liq = [calc_ipr(p_res, p_wf_i, pi) for p_wf_i in p_wf]

    # plt.plot(q_liq, p_wf)
    # plt.show()

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

    tubing = Pipeline(
        top_depth=0,
        bottom_depth=data['pipe']['tubing']['md'],
        d=data['pipe']['tubing']['d'],
        roughness=data['pipe']['tubing']['roughness'],
        fluid=fluid,
        trajectory=traj,
        ambient_temperature_distribution=amb
    )

    pt = casing.calc_pt(
        h_start='bottom',
        p_mes=150*101325,
        flow_direction=-1,
        extra_output=True
    )

    tubing.calc_pt(
        h_start='bottom',
        p_mes=pt[0],
        flow_direction=-1,
        extra_output=True
    )

    plt.plot(casing.distributions['p'] / 101325, -casing.distributions['depth'])
    plt.plot(tubing.distributions['p'] / 101325, -tubing.distributions['depth'])
    plt.show()

    plt.plot(casing.distributions['gas_fraction'], -casing.distributions['depth'])
    plt.plot(tubing.distributions['gas_fraction'], -tubing.distributions['depth'])
    plt.show()

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

        pt_wh[i] = tubing.calc_pt(
            h_start='bottom',
            p_mes=pt[0],
            flow_direction=-1,
            q_liq=q_liq[i]/86400,
            extra_output=True
        )[0]

    plt.plot(q_liq, pt_wh / 101325)
    plt.plot(q_liq, [data['regime']['p_wh']] * len(q_liq))
    plt.show()
