import json
from os import PathLike
from pathlib import Path
from typing import Union, Any
from functools import lru_cache
from typing import Tuple
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize
from scipy.interpolate import interp1d

from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory, GasLiftWell
import unifloc.tools.units_converter as uc

#1 часть 2 ДЗ
class HWWell1:

    def __init__(self, fluid_data: dict, inclinometry_data: dict, temp_dist_data: dict,
        casing_data: dict, tubing_data: dict, reservoir_data: dict,) -> None:

        _fluid = FluidFlow(**fluid_data)
        _trajectory = Trajectory(**inclinometry_data)
        _temp_dist = AmbientTemperatureDistribution(**temp_dist_data)

        self.casing = Pipeline(
            **casing_data,
            trajectory=_trajectory,
            fluid=_fluid,
            ambient_temperature_distribution=_temp_dist )
        self.tubing = Pipeline(**tubing_data, trajectory=_trajectory,
        fluid=_fluid, ambient_temperature_distribution=_temp_dist)

        self.p_res = reservoir_data["p_res"]
        self.prod_index = reservoir_data["pi"]

        self.ipr_object = self.calc_ipr()

    def calc_q_ipr_pwf(self, p_wf: Union[float, NDArray[float]]) -> Union[float, NDArray[float]]:
        """
        Расчета дебита из пласта.
        :param p_wf: давление фонтанирования скважины, Па;
        :return: Q_ipr, м3/с
        """
        return calc_q_ipr_gen(prod_index=self.prod_index, p_res=self.p_res, p_wf=p_wf)

    def calc_ipr(self) -> pd.DataFrame:
        """
        Метод расчета IPR.
        :return: IPR
        """
        _p_wf_array = np.linspace(self.p_res, 1e-12, 100)

        return pd.DataFrame({"Qliq": self.calc_q_ipr_pwf(_p_wf_array), "Pwf": _p_wf_array})

    def calc_p_wh_p_wf(self, q_liq: float, p_wf: float) -> float:
        """
        Рассчет буферного давление по забойному.
        """
        _p_cas, *_ = self.casing.calc_pt(h_start="bottom", p_mes=p_wf, flow_direction=-1, q_liq=q_liq, extra_output=True)
        _p_wh, *_ = self.tubing.calc_pt(h_start="bottom", p_mes=_p_cas, flow_direction=-1, q_liq=q_liq, extra_output=True)

        return _p_wh

    def calc_p_wf_min_fountain( self, p_wh_aim: float,) -> Union[float, Any]:
        """
        Определение минимального забойного давления фонтанирования, при заданном буферном давлении.
        :param p_wh_aim: буферное давление, Па;
        :return: величина минимального забойного давления при котором, скважина начинает фонтанировать.
        """
        def p_wh_diff(p_wf_guess: float) -> float:
            """
            Рассчет разницы между целевым и расчетным буферным давлением.
            """
            nonlocal p_wh_aim

            _q_liq = self.calc_q_ipr_pwf(p_wf_guess)
            _p_wh = self.calc_p_wh_p_wf(p_wf=p_wf_guess, q_liq=_q_liq)

            return p_wh_aim - _p_wh  # -> 0

        p_wf_solution = optimize.brentq(f=p_wh_diff, a=self.ipr_object["Pwf"].values[-1], b=self.ipr_object["Pwf"].values[0])

        return p_wf_solution


class HWWell2(GasLiftWell):

    def __init__(self, *args, reservoir_data: dict, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        self.p_res = reservoir_data["p_res"]
        self.prod_index = reservoir_data["pi"]

        self.ipr_object = self.calc_ipr()
        self.ipr_func = interp1d(
            x=self.ipr_object["Qliq"].values,
            y=self.ipr_object["Pwf"].values,
            fill_value="extrapolate",
        )

    def calc_q_ipr_pwf(self,p_wf: Union[float, NDArray[float]]) -> Union[float, NDArray[float]]:
        """
        Расчета дебита из пласта.
        :param p_wf: давление фонтанирования скважины, Па;
        :return: Q_ipr, м3/с
        """
        return calc_q_ipr_gen(
            prod_index=self.prod_index,
            p_res=self.p_res,
            p_wf=p_wf,
        )

    def calc_ipr(self) -> pd.DataFrame:
        """
        Метод расчета IPR.
        :return: IPR
        """
        _p_wf_array = np.linspace(self.p_res, 1e-12, 100)

        return pd.DataFrame(
            {
                "Qliq": self.calc_q_ipr_pwf(_p_wf_array),
                "Pwf": _p_wf_array,
            },
        )

    @lru_cache(maxsize=1)
    def calc_q_liq_pwh(self,p_wh: float, q_gas_inj: float) -> float:

        def calc_q_diff(q_vlp_guess: float) -> float:
            nonlocal p_wh, q_gas_inj
            _p_wf_ipr = self.ipr_func(q_vlp_guess)
            _p_wf_vlp = self.calc_pwf_pfl(p_fl=p_wh, q_liq=q_vlp_guess, wct=self.fluid.wct, q_gas_inj=q_gas_inj)

            return _p_wf_vlp - _p_wf_ipr  # -> 0

        q_liq_sol = optimize.brentq(f=calc_q_diff, a=self.ipr_object["Qliq"].values[0],
                                    b=self.ipr_object["Qliq"].values[-1])

        return q_liq_sol # noqa

    def optimal_q_inj(self, p_wh: float) -> Tuple[float, float]:

        def neg_q_liq(q_gas_inj_guess: float) -> float:
            nonlocal p_wh

            _q_liq = self.calc_q_liq_pwh( p_wh=p_wh, q_gas_inj=q_gas_inj_guess)

            return -_q_liq

        solution = optimize.minimize_scalar(
            fun=neg_q_liq,
            bounds=(uc.convert_rate(5000, "m3/day", "m3/s"), uc.convert_rate(150000, "m3/day", "m3/s")),
            method='bounded')

        q_inj_opt: float = solution.x
        q_vlp_max: float = self.calc_q_liq_pwh(p_wh=p_wh, q_gas_inj=q_inj_opt)

        return q_vlp_max, q_inj_opt


def calc_q_ipr_gen(prod_index: float, p_res: float, p_wf: float) -> float:
    _p_diff_atm = uc.convert_pressure(
        (p_res - p_wf),
        "pa",
        "atm",
    )
    _q_ipr_m3_day = prod_index * _p_diff_atm
    return uc.convert_rate(_q_ipr_m3_day, "m3/day", "m3/s")


def calc_temp(t_res: float, t_grad: float, tvd_h: float, tvd_vdp: float) -> float:

    return t_res - (tvd_vdp - tvd_h) / 100 * t_grad

def homework_part_1(
    fluid_data: dict,
    inclinometry_data: dict,
    temp_dist_data: dict,
    casing_data: dict,
    tubing_data: dict,
    reservoir_data: dict,
    regime_data: dict,
) -> Dict[str, float]:

    well_obj = HWWell1(
        fluid_data=fluid_data,
        inclinometry_data=inclinometry_data,
        temp_dist_data=temp_dist_data,
        casing_data=casing_data,
        tubing_data=tubing_data,
        reservoir_data=reservoir_data,
    )

    p_wf_min_fountain = well_obj.calc_p_wf_min_fountain(
        p_wh_aim=regime_data["p_wh"],
    )

    hw_1_solution = uc.convert_pressure(
        p_wf_min_fountain,
        "pa",
        "atm",
    )

    return {"pwf": round(hw_1_solution, 2)}


def homework_part_2(
    fluid_data: dict,
    inclinometry_data: dict,
    temp_dist_data: dict,
    casing_data: dict,
    tubing_data: dict,
    reservoir_data: dict,
    regime_data: dict,
    equipment_data: dict,
) -> Dict[str, float]:

    well_obj = HWWell2(
        fluid_data=fluid_data,
        pipe_data={
            'casing': casing_data,
            'tubing': tubing_data,
        },
        well_trajectory_data=inclinometry_data,
        ambient_temperature_data=(
            temp_dist_data['ambient_temperature_distribution']
        ),
        equipment_data=equipment_data,
        reservoir_data=reservoir_data,
    )

    q_liq_opt, q_gas_inj_opt = well_obj.optimal_q_inj(
        p_wh=regime_data["p_wh"]
    )

    final_solution = {
        "q_inj": round(
            uc.convert_rate(
                q_gas_inj_opt,
                "m3/s",
                "m3/day"
            ), 2),
        "q_liq": round(
            uc.convert_rate(
                q_liq_opt,
                "m3/s",
                "m3/day",
            ), 2),
    }

    return final_solution


def arrange_input_data(
    file_1: Union[str, Path, PathLike],
    file_2: Union[str, Path, PathLike],
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:

    with open(file_1, 'r', encoding='utf-8') as task_1_file, \
            open(file_2, 'r', encoding='utf-8') as task_2_file:
        task_1_data: dict = json.load(task_1_file)
        task_2_data: dict = json.load(task_2_file)

    fluid_data: dict = {
        'q_fluid': uc.convert_rate(
            100,  # случайная величина
            'm3/day',
            'm3/s',
        ),
        'wct': task_1_data['fluid']['wct'],
        'pvt_model_data': {
            "black_oil":
                {
                    "gamma_gas": task_1_data['fluid']['gamma_gas'],
                    "gamma_wat": task_1_data['fluid']['gamma_water'],
                    "gamma_oil": task_1_data['fluid']['gamma_oil'],
                    "rp": task_1_data['fluid']['rp'],
                },
        },
    }

    inclinometry_data: dict = {
        'inclinometry': pd.DataFrame(
            {
                "MD": task_1_data["inclinometry"]["md"],
                "TVD": task_1_data["inclinometry"]["tvd"],
            },
        ),
    }

    temp_dist_data: dict = {
        'ambient_temperature_distribution': {
            "MD": inclinometry_data["inclinometry"]["MD"].to_list(),
            "T": [
                calc_temp(
                    tvd_vdp=inclinometry_data["inclinometry"]["TVD"].iloc[-1],
                    tvd_h=tvd_h_i,
                    t_res=uc.convert_temperature(
                        task_1_data["temperature"]["t_res"],
                        "celsius",
                        "kelvin",
                    ),
                    t_grad=task_1_data["temperature"]["temp_grad"],
                )
                for tvd_h_i in inclinometry_data["inclinometry"]["TVD"]
            ],
        },
    }

    casing_data: dict = {
        'top_depth': task_1_data["pipe"]["tubing"]["md"],
        'bottom_depth': task_1_data['reservoir']["md_vdp"],
        'd': task_1_data["pipe"]["casing"]["d"],
        'roughness': task_1_data["pipe"]["casing"]["roughness"],
    }

    tubing_data: dict = {
        'top_depth': 0,
        'bottom_depth': task_1_data["pipe"]["tubing"]["md"],
        'd': task_1_data["pipe"]["tubing"]["d"],
        'roughness': task_1_data["pipe"]["tubing"]["roughness"],
    }

    reservoir_data: dict = {
        "md_vdp": task_1_data['reservoir']["md_vdp"],
        "p_res": uc.convert_pressure(
            task_1_data['reservoir']["p_res"],
            "atm",
            "pa",
        ),
        "pi": task_1_data['reservoir']["pi"],
    }

    regime_data: dict = {
        "p_wh": uc.convert_pressure(
            task_1_data['regime']["p_wh"],
            "atm",
            "pa",
        ),
    }

    equipment_data = {
        "gl_system": {
            "valve1": {
                "h_mes": task_2_data["md_valve"],
                "d": 0.006
            }
        }
    }

    return (
        fluid_data,
        inclinometry_data,
        temp_dist_data,
        casing_data,
        tubing_data,
        reservoir_data,
        regime_data,
        equipment_data,
    )


def save_results(obj_to_save: Union[dict, List[Dict], Any], save_path: Union[str, Path, PathLike],) -> None:

    with open(save_path, 'w', encoding='utf-8') as dump_file:
        json.dump(
            obj=obj_to_save,
            fp=dump_file,
        )


def main(file_1_name: str = '21_1.json', file_2_name: str = '21_2.json', save_name: str = 'output.json') -> None:
    base_path = Path(__file__).parent
    file_1_path = base_path.joinpath(file_1_name)
    file_2_path = base_path.joinpath(file_2_name)
    save_path = base_path.joinpath(save_name)

    (
        fluid_data,
        inclinometry_data,
        temp_dist_data,
        casing_data,
        tubing_data,
        reservoir_data,
        regime_data,
        equipment_data,
    ) = arrange_input_data(
        file_1=file_1_path,
        file_2=file_2_path,
    )

    hw_1_solution = homework_part_1(
        fluid_data=fluid_data,
        inclinometry_data=inclinometry_data,
        temp_dist_data=temp_dist_data,
        casing_data=casing_data,
        tubing_data=tubing_data,
        reservoir_data=reservoir_data,
        regime_data=regime_data,
    )
    hw_2_solution = homework_part_2(
        fluid_data=fluid_data,
        inclinometry_data=inclinometry_data,
        temp_dist_data=temp_dist_data,
        casing_data=casing_data,
        tubing_data=tubing_data,
        regime_data=regime_data,
        reservoir_data=reservoir_data,
        equipment_data=equipment_data,
    )

    final_results = {
        "a1": hw_1_solution,
        "a2": hw_2_solution,
    }

    print(final_results)


if __name__ == '__main__':
    main()