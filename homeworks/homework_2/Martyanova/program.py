import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import copy

from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory

with open(r".\input_data\13-1.json") as f:
    data = json.load(f)

with open(r".\input_data\13-2.json") as f:
    data2 = json.load(f)

class Well:
    def __init__(self, data, data2):
        self.data = data
        self.data2 = data2
        self.amb = None
        self.fluid = None
        self.fluid_with_gas = None
        self.traj = None
        self.casing = None
        self.tubing = None
        self.tubing_with_gas = None
        self.t1_p_wf = None
        self.t2_q_gas = None
        self.t2_q_liq = None
    
    @staticmethod
    def _calc_amb_temp(t_res, tvd_vdp, grad_t, tvd_h):
        return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15
    
    def _define_ambient_temp(self):
        data = self.data
        t = []
        t_res = data["temperature"]["t_res"]
        tvd_vdp = data["inclinometry"]["tvd"][-1]
        grad_t = data["temperature"]["temp_grad"]

        for depth in data["inclinometry"]["tvd"]:
            t.append(self._calc_amb_temp(t_res, tvd_vdp, grad_t, depth))
        
        amb_dist = {"MD": data["inclinometry"]["md"],
            "T": t}

        self.amb = AmbientTemperatureDistribution(ambient_temperature_distribution=amb_dist)
        
        
        
    def _define_fluid_model(self):
        data = self.data
        self.fluid =  FluidFlow(q_fluid=1/86400, wct=data["fluid"]["wct"],
                   pvt_model_data={"black_oil": 
                                   {"gamma_gas": data["fluid"]["gamma_gas"], 
                                    "gamma_wat": data["fluid"]["gamma_water"], 
                                    "gamma_oil": data["fluid"]["gamma_oil"],
                                    "rp": data["fluid"]["rp"]}})
        self.fluid_with_gas = copy.deepcopy(self.fluid)
        
    def _define_trajectory(self):     
        inclinometry = {"MD": data["inclinometry"]["md"],
                       "TVD": data["inclinometry"]["tvd"]}

        self.traj = Trajectory(inclinometry=inclinometry)

    def _define_pipes_natural(self):
        data = self.data
        self.casing = Pipeline(
                                top_depth = data["pipe"]["tubing"]["md"],
                                bottom_depth = data["reservoir"]["md_vdp"],
                                d = data["pipe"]["casing"]["d"],
                                roughness=data["pipe"]["casing"]["roughness"],
                                fluid=self.fluid,
                                trajectory=self.traj,
                                ambient_temperature_distribution=self.amb
                                )
        self.tubing = Pipeline(
                                top_depth = 0,
                                bottom_depth =  data["pipe"]["tubing"]["md"],
                                d = data["pipe"]["tubing"]["d"],
                                roughness=data["pipe"]["tubing"]["roughness"],
                                fluid=self.fluid,
                                trajectory=self.traj,
                                ambient_temperature_distribution=self.amb
                                ) 
        
    def _define_pipes_gl(self):
        data = self.data
        self.casing = Pipeline(
                                top_depth = data["pipe"]["tubing"]["md"],
                                bottom_depth = data["reservoir"]["md_vdp"],
                                d = data["pipe"]["casing"]["d"],
                                roughness=data["pipe"]["casing"]["roughness"],
                                fluid=self.fluid,
                                trajectory=self.traj,
                                ambient_temperature_distribution=self.amb
                                )
        self.tubing = Pipeline(
                                top_depth = self.data2['md_valve'],
                                bottom_depth =  data["pipe"]["tubing"]["md"],
                                d = data["pipe"]["tubing"]["d"],
                                roughness=data["pipe"]["tubing"]["roughness"],
                                fluid=self.fluid,
                                trajectory=self.traj,
                                ambient_temperature_distribution=self.amb
                                ) 
        self.tubing_with_gas = Pipeline(
                                top_depth = 0,
                                bottom_depth = self.data2['md_valve'],
                                d = data["pipe"]["tubing"]["d"],
                                roughness=data["pipe"]["tubing"]["roughness"],
                                fluid=self.fluid_with_gas,
                                trajectory=self.traj,
                                ambient_temperature_distribution=self.amb
                                ) 
        
        
    
        
    
    def calc_q_liq_by_ipr(self, p_wf):
        p_res = self.data["reservoir"]["p_res"]
        pi = self.data["reservoir"]["pi"]
        return pi * (p_res-p_wf)
    
    def calc_pwf_by_ipr(self, q_liq):
        p_res = self.data["reservoir"]["p_res"]
        pi = self.data["reservoir"]["pi"]
        return p_res-q_liq / pi
    
    def plot_ipr(self):
        p_wf_list = np.linspace(0.1, self.data["reservoir"]["p_res"], 100)
        q_ipr_list = [self.calc_q_liq_by_ipr(this_p_wf) for this_p_wf in p_wf_list]
        plt.plot(q_ipr_list, p_wf_list)
        plt.xlabel(f"Дебит жидкости, м3/сут")
        plt.ylabel(f"Давление забойное, атм")
        plt.title("IPR")
        plt.show()
    
    def calc_p_wh_from_bottom(self, q_liq, plot=False):
        p_wf = self.calc_pwf_by_ipr(q_liq)
        pt = self.casing.calc_pt(
                                h_start='bottom',
                                p_mes = p_wf * 101325,
                                flow_direction=-1,
                                q_liq=q_liq/86400,
                                extra_output=True
                                  )
        p_wh = self.tubing.calc_pt(
                            h_start='bottom',
                            p_mes = pt[0],
                            flow_direction=-1,
                            q_liq=q_liq/86400,
                            extra_output=True
                                  )[0]
        if plot:
            plt.plot(self.casing.distributions["p"]/101325, self.casing.distributions["depth"] * (-1), label = 'casing')
            plt.plot(self.tubing.distributions["p"]/101325, self.tubing.distributions["depth"] * (-1), label = 'tubing')
            plt.xlabel('Давление, атм')
            plt.ylabel('Глубина, м')
            plt.legend()
            plt.show()
            
            plt.plot(self.casing.distributions["t"] - 273.15, self.casing.distributions["depth"] * (-1), label = 'casing')
            plt.plot(self.tubing.distributions["t"] - 273.15, self.tubing.distributions["depth"] * (-1), label = 'tubing')
            plt.xlabel('Температура, С')
            plt.ylabel('Глубина, м')
            plt.legend()
            plt.show()
            
        return p_wf, p_wh / 101325
    
    def calc_p_wf_from_top(self, q_liq, q_gas, plot=False):
        self.tubing_with_gas.fluid.q_gas_free = q_gas
        pt = self.tubing_with_gas.calc_pt( 
                        h_start='top',
                        p_mes = self.data['regime']['p_wh'] * 101325,
                        flow_direction=1,
                        q_liq=q_liq/86400,
                        extra_output=True
                          )
        pt = self.tubing.calc_pt(
                    h_start='top',
                    p_mes = pt[0],
                    flow_direction=1,
                    q_liq=q_liq/86400,
                    extra_output=True
                          )
        pt = self.casing.calc_pt(
                        h_start='top',
                        p_mes = pt[0],
                        flow_direction=1,
                        q_liq=q_liq/86400,
                        extra_output=True
                          )
        if plot:
            plt.plot(self.casing.distributions["p"]/101325, self.casing.distributions["depth"] * (-1), label = 'casing')
            plt.plot(self.tubing.distributions["p"]/101325, self.tubing.distributions["depth"] * (-1), label = 'tubing')
            plt.plot(self.tubing_with_gas.distributions["p"]/101325, 
                     self.tubing_with_gas.distributions["depth"] * (-1), label = 'tubing')
            plt.xlabel('Давление, атм')
            plt.ylabel('Глубина, м')
            plt.legend()
            plt.show()
        return pt[0]/101325
        
    def build_model(self, is_gl=False):
        self._define_ambient_temp()
        self._define_fluid_model()
        self._define_trajectory()
        if is_gl:
            self._define_pipes_gl()
        else:
            self._define_pipes_natural()

    def solve_nodal_analysis_1_task(self, plot = True):
        q_liq_list = np.linspace(0.1, 200, 200)
        p_wh_list = q_liq_list * 0
        p_wf_list = q_liq_list * 0
        
        for i, this_q_liq in enumerate(q_liq_list):
            this_p_wf, this_p_wh = self.calc_p_wh_from_bottom(this_q_liq, plot=False)
            p_wh_list[i] = this_p_wh
            p_wf_list[i] = this_p_wf
            if this_p_wh  < 1:
                break
                
        q_liq_list = q_liq_list[p_wh_list != 0]
        p_wf_list = p_wf_list[p_wh_list != 0]
        p_wh_list = p_wh_list[p_wh_list != 0]

        p_wh_list_regime = p_wh_list * 0 + self.data['regime']['p_wh']
        
        res = self._find_intersection(q_liq_list, p_wh_list, p_wh_list_regime)
        
        q_liq = res[0][0]
        
        self.t1_p_wf = self.calc_pwf_by_ipr(q_liq)
        #print(self.t1_p_wf)
        #print(self.calc_p_wh_from_bottom(q_liq))
        if plot:
            plt.plot(q_liq_list, p_wh_list)
            plt.plot(q_liq_list, p_wh_list_regime)
            #plt.plot(q_liq_list, p_wf_list)
            plt.xlabel('Дебит, м3/сут')
            plt.ylabel('Давление на устье, атм')
            plt.show()
        return q_liq, self.t1_p_wf
    
    def solve_nodal_analysis_2_task(self, q_gas, plot = True):
        q_liq_list = np.linspace(0.1, 150, 50)
        p_wf_vlp_list = q_liq_list * 0
        p_wf_ipr_list =  q_liq_list * 0
        for i, this_q_liq in enumerate(q_liq_list):
            this_p_wf = self.calc_p_wf_from_top(this_q_liq, q_gas, plot=False)
            p_wf_vlp_list[i] = this_p_wf
            p_wf_ipr_list[i] = self.calc_pwf_by_ipr(this_q_liq)
            if p_wf_ipr_list[i]  < 1:
                break
                
        #q_liq_list = q_liq_list[p_wh_list != 0]
        #p_wf_list = p_wf_list[p_wh_list != 0]
        #p_wh_list = p_wh_list[p_wh_list != 0]

        res = self._find_intersection(q_liq_list, p_wf_vlp_list, p_wf_ipr_list)
        
        if res is None:
            q_liq = None
        else:
            q_liq = res[0][0]
        #print(res)
        
        p_wf = self.calc_pwf_by_ipr(q_liq)
        if plot:
            plt.plot(q_liq_list, p_wf_ipr_list)
            plt.plot(q_liq_list, p_wf_vlp_list)
            plt.xlabel('Дебит, м3/сут')
            plt.ylabel('Давление забое, атм')
            plt.show()
        return q_liq, p_wf
        
    @staticmethod    
    def _find_intersection(x: np.array, f: np.array, g: np.array) -> np.array:
        first_line = LineString(np.column_stack((x, f)))
        second_line = LineString(np.column_stack((x, g)))
        intersection = first_line.intersection(second_line)

        if intersection.geom_type == "MultiPoint":
            return LineString(intersection).xy
        elif intersection.geom_type == "Point":
            return intersection.xy
        else:
            return None
        
    def plot_trajectory(self):
        plt.plot(np.array(self.traj.inclinometry['MD']) - np.array(self.traj.inclinometry['TVD']), 
                 np.array(self.traj.inclinometry['TVD']) * (-1))
        plt.xlabel('Смещение, м')
        plt.ylabel('TVD, м')
        plt.show()
        
    def find_q_gas_opt_2_task(self,q_gas_range = np.linspace(0, 300, 30),  plot=False):
        q_liq_list = np.zeros(len(q_gas_range))
        p_wf_list = np.zeros(len(q_gas_range))

        for i, this_q_gas in enumerate(q_gas_range):
            this_q_liq, this_pwf = self.solve_nodal_analysis_2_task(q_gas = this_q_gas * 10**3 /86400, plot=plot)
            q_liq_list[i] = this_q_liq
            p_wf_list[i] = this_pwf
            
        self.t2_q_gas = q_gas_range[np.argmax(q_liq_list)] * 10**3
        self.t2_q_liq = q_liq_list[np.argmax(q_liq_list)]
        
        if plot:
            plt.plot(q_gas_range, q_liq_list, '-o')
            plt.xlabel('Дебит газа, м3/сут * 1000')
            plt.ylabel('Дебит жидкости, м3/сут')
            plt.show()
        return self.t2_q_gas, self.t2_q_liq, q_gas_range, q_liq_list, p_wf_list
    
    def save_to_json(self):
        with open('output.json', 'w', encoding='utf-8') as output_file:
            json.dump(
                {   
                    "t1": {"pwf": self.t1_p_wf},
                    "t2": {"q_inj": self.t2_q_gas,
                           "q_liq": self.t2_q_liq
                          }
                }, 
                output_file, 
                indent=4)


well = Well(data, data2)
well.build_model()
#well.plot_ipr()
#well.plot_trajectory()
well.solve_nodal_analysis_1_task(plot=False)

well.build_model(is_gl=True)
t2_q_gas, t2_q_liq, q_gas_range, q_liq_list, p_wf_list = well.find_q_gas_opt_2_task(q_gas_range = np.linspace(0, 250, 20), plot=False)
well.save_to_json()


