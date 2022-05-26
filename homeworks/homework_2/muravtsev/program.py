import numpy as np
import matplotlib.pyplot as plt
import json
from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory


def find_intersection(x, y1, y2):
    idx = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    idx = idx[0]
    return x[idx], y1[idx]


def calc_ipr(p_res, p_wf, pi):
    return pi * (p_res - p_wf)


def calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
    return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15


if __name__ == '__main__':

    with open('../input_data/16-1.json') as f:
        data = json.load(f)

    data['inclinometry']['tvd'][1] = 1400  # исправление данных инклинометрии

    p_res = data['reservoir']['p_res']
    pi = data['reservoir']['pi']
    p_wf = np.linspace(0.1, p_res, 150)
    q_liq = [calc_ipr(p_res, p_wf_i, pi) for p_wf_i in p_wf]

    # plot IPR
    # plt.plot(q_liq, p_wf)
    # plt.show()

    # earth temperature distribution
    T = []
    t_res = data['temperature']['t_res']
    tvd_vdp = data['inclinometry']['tvd'][-1]
    grad_t = data['temperature']['temp_grad']

    for depth in data['inclinometry']['tvd']:
        T.append(calc_amb_temp(t_res, tvd_vdp, grad_t, depth))

    # fluid object
    fluid = FluidFlow(
        q_fluid=100/86400,
        wct=data['fluid']['wct'],
        pvt_model_data={"black_oil": {"gamma_gas": data['fluid']['gamma_gas'],
                                      "gamma_wat": data['fluid']['gamma_water'],
                                      "gamma_oil":  data['fluid']['gamma_oil'],
                                      "rp": data['fluid']['rp']}}
    )

    # inclinometry object
    inclinometry = {'MD': data['inclinometry']['md'],
                    'TVD': data['inclinometry']['tvd']}
    traj = Trajectory(inclinometry=inclinometry)

    # temperature distribution object
    amb_dist = {'MD': data['inclinometry']['md'], 'T': T}
    amb = AmbientTemperatureDistribution(ambient_temperature_distribution=amb_dist)

    # casing object
    casing = Pipeline(
        top_depth=data['pipe']['tubing']['md'],
        bottom_depth=data['reservoir']['md_vdp'],
        d=data['pipe']['casing']['d'],
        roughness=data['pipe']['casing']['roughness'],
        fluid=fluid,
        trajectory=traj,
        ambient_temperature_distribution=amb
    )

    # tubing object
    tubing = Pipeline(
        top_depth=0,
        bottom_depth=data['pipe']['tubing']['md'],
        d=data['pipe']['tubing']['d'],
        roughness=data['pipe']['tubing']['roughness'],
        fluid=fluid,
        trajectory=traj,
        ambient_temperature_distribution=amb
    )

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
    min_q_liq, min_p_wh = find_intersection(q_liq, pt_wh / 101325, [data['regime']['p_wh']] * len(q_liq))

    pt = casing.calc_pt(
        h_start='top',
        p_mes=min_p_wh*101325,
        flow_direction=1,
        q_liq=min_q_liq/86400,
        extra_output=True
    )

    p_wf_result = tubing.calc_pt(
        h_start='top',
        p_mes=pt[0],
        flow_direction=1,
        q_liq=min_q_liq/86400,
        extra_output=True
    )[0]

    p_wf_result /= 101325
    print(p_wf_result)

    to_json_file = {
        't1': {
            'pwf': p_wf_result
        }
    }
    with open('output.json', 'w', encoding='UTF-8') as f:
        json.dump(to_json_file, f, indent='\t')
    plt.show()
