import numpy as np
import matplotlib.pyplot as plt
import json
from shapely.geometry import LineString
from unifloc.well.gaslift_well import GasLiftWell


def find_intersection(x, y1, y2):
    first_line = LineString(np.column_stack((x, y1)))
    second_line = LineString(np.column_stack((x, y2)))
    intersection = first_line.intersection(second_line)
    x_intersect, y_intersect = intersection.xy
    return x_intersect[0], y_intersect[0]


def calc_ipr(p_res, p_wf, pi):
    return pi * (p_res - p_wf)


def calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
    return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15


if __name__ == '__main__':

    # считывание данных
    with open('../input_data/16-1.json') as f:
        data = json.load(f)
    with open('../input_data/16-2.json') as f1:
        data1 = json.load(f1)
    data['inclinometry']['tvd'][1] = 1400  # исправление данных инклинометрии

    # распределение температуры
    T = []
    t_res = data['temperature']['t_res']
    tvd_vdp = data['inclinometry']['tvd'][-1]
    grad_t = data['temperature']['temp_grad']
    for depth in data['inclinometry']['tvd']:
        T.append(calc_amb_temp(t_res, tvd_vdp, grad_t, depth))

    # объект газлифтной скважины
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

    # расчёт
    p_res = data['reservoir']['p_res']
    pi = data['reservoir']['pi']
    p_wf = np.linspace(0.1, p_res, 50)
    q_liqs = [calc_ipr(p_res, p_wf_i, pi) for p_wf_i in p_wf]

    q_gas_injs = range(5000, 150000, 10000)

    gaslift_curve = []

    for q_gas_inj in q_gas_injs:
        print('Текущий расход газлифтного газа: ' + str(q_gas_inj) + ' м3/сут')
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

    print('Уточнение значений вблизи найденного максимума')
    # уточняю данные вблизи найденного максимума (расчёт с меньшим шагом q_gas_injs вблизи найденного максимума)
    q_gas_injs = range(q_gas_optimal - 10000, q_gas_optimal + 10000, 2000)
    gaslift_curve_accurate = []

    for q_gas_inj in q_gas_injs:
        print('Текущий расход газлифтного газа: ' + str(q_gas_inj) + ' м3/сут')
        gaslift_vlp = []
        for q_liq in q_liqs:
            gaslift_vlp.append(
                gaslift_well.calc_pwf_pfl(
                    p_fl=data['regime']['p_wh'] * 101325,
                    q_liq=q_liq / 86400,
                    wct=data['fluid']['wct'],
                    q_gas_inj=q_gas_inj / 86400
                ) / 101325)
        gaslift_curve_accurate.append(
            find_intersection(q_liqs, p_wf, gaslift_vlp)[0]
        )

    q_liq_optimal_accurate = max(gaslift_curve_accurate)
    q_gas_optimal_accurate = q_gas_injs[gaslift_curve_accurate.index(max(gaslift_curve_accurate))]

    plt.plot(q_gas_injs, gaslift_curve_accurate)

    print(q_gas_optimal_accurate)
    print(q_liq_optimal_accurate)

    with open('output.json', 'r', encoding='UTF-8') as f:
        to_json_file = json.load(f)

    to_json_file['t2'] = {'q_inj': q_gas_optimal_accurate,
                          'q_liq': q_liq_optimal_accurate}

    with open('output.json', 'w', encoding='UTF-8') as f:
        to_json_file = json.dump(to_json_file, f, indent='\t')

    plt.show()

'''
    # попытка решить без использования GasLiftWell
    # fluid object
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

    fluid.q_gas_free = 50000 / 86400
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
'''
