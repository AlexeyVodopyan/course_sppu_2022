import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from unifloc import FluidFlow, Pipeline, AmbientTemperatureDistribution, Trajectory
from unifloc.well.gaslift_well import GasLiftWell
import unifloc.tools.units_converter as uc

with open("8-1.json") as f:
    data = json.load(f)
    
with open("8-2.json") as f:
    data2 = json.load(f)
    

class Calc:
    
    def __init__(self, data, data2) -> None:
        # Задача № 1
        
        # for ipr
        self.p_res = data['reservoir']['p_res']
        self.pi = data['reservoir']['pi']
        self.md_vdp = data['reservoir']['md_vdp']
        self.ipr_p_wf = np.linspace(0, self.p_res, 40)
        self.q_liq_list = self.calc_ipr_list()
        
        # for vlp
        self.wct = data["fluid"]["wct"]
        self.init_well(data)
        self.p_wh = data["regime"]["p_wh"]
        self.find_min_pwh()
        self.min_p_wf = self.calc_p_wf(self.min_q_liq, self.min_p_wh) # ответ на задачу
        self.vlp_p_wf = [self.calc_p_wf(_, self.min_p_wh) for _ in self.q_liq_list]
        self.ax2.plot(self.q_liq_list, self.ipr_p_wf)
        self.ax2.plot(self.q_liq_list, self.vlp_p_wf)
        self.ax2.set(xlabel='дебита закачиваемой жидкости, м3/сут', ylabel='забойного давления, атм')
        
        # Задание № 2
        self.md_valve = data2['md_valve']
        self.init_gl_well(data)
        self.q_inj_opt, self.q_liq_opt = self.find_max_gl_solution()
        
    def calc_ipr(self, p_res, p_wf, pi):
        return pi* (p_res-p_wf)
        
    def calc_ipr_list(self):
        return [self.calc_ipr(self.p_res, p_wf_i, self.pi) for p_wf_i in self.ipr_p_wf]
    
    def calc_amb_temp(self, t_res, tvd_vdp, grad_t, tvd_h):
        return t_res - grad_t * (tvd_vdp - tvd_h) / 100 + 273.15
        
    def init_well(self, data):
        fluid =  FluidFlow(
            q_fluid=100/86400, 
            wct=self.wct,
            pvt_model_data={
               "black_oil": 
                           {"gamma_gas": data["fluid"]["gamma_gas"], 
                            "gamma_wat": data["fluid"]["gamma_water"], 
                            "gamma_oil": data["fluid"]["gamma_oil"],
                            "rp": data["fluid"]["rp"]
                            }
            }
        )
        self.inclinometry = {"MD": data["inclinometry"]["md"],
                       "TVD": data["inclinometry"]["tvd"]}
        traj = Trajectory(inclinometry=self.inclinometry)
        
        
        t = []
        t_res = data["temperature"]["t_res"]
        tvd_vdp = data["inclinometry"]["tvd"][-1]
        grad_t = data["temperature"]["temp_grad"]

        for depth in data["inclinometry"]["tvd"]:
            t.append(self.calc_amb_temp(t_res, tvd_vdp, grad_t, depth))
        self.amb_dist = {"MD": data["inclinometry"]["md"],
                    "T": t}

        amb = AmbientTemperatureDistribution(ambient_temperature_distribution=self.amb_dist)

        self.casing = Pipeline(
            top_depth = data["pipe"]["tubing"]["md"],
            bottom_depth = data["reservoir"]["md_vdp"],
            d = data["pipe"]["casing"]["d"],
            roughness=data["pipe"]["casing"]["roughness"],
            fluid=fluid,
            trajectory=traj,
            ambient_temperature_distribution=amb
        )
        self.tubing = Pipeline(
            top_depth = 0,
            bottom_depth =  data["pipe"]["tubing"]["md"],
            d = data["pipe"]["tubing"]["d"],
            roughness=data["pipe"]["tubing"]["roughness"],
            fluid=fluid,
            trajectory=traj,
            ambient_temperature_distribution=amb
        )
       
    def find_min_pwh(self):
        pt_wh = np.zeros(len(self.ipr_p_wf))
        for i in range(len(self.ipr_p_wf)):
            pt = self.casing.calc_pt(
                h_start='bottom',
                p_mes = self.ipr_p_wf[i] * 101325,
                flow_direction=-1,
                q_liq=self.q_liq_list[i]/86400,
                extra_output=True
                  )
            pt_wh[i] = self.tubing.calc_pt(
                h_start='bottom',
                p_mes = pt[0],
                flow_direction=-1,
                q_liq=self.q_liq_list[i]/86400,
                extra_output=True
            )[0]
        fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)
        self.ax1.plot(self.q_liq_list, 
            [self.p_wh for _ in range(len(self.q_liq_list))])
        self.ax1.plot(self.q_liq_list,
            np.array(pt_wh)/101325)
        self.ax1.set(xlabel='дебита закачиваемой жидкости, м3/сут', ylabel='устьевое давления, атм')
   
        self.min_q_liq, self.min_p_wh = self.find_sol(
            self.q_liq_list, 
            [self.p_wh for _ in range(len(self.q_liq_list))], 
            self.q_liq_list,
            np.array(pt_wh)/101325
        )
        
    def calc_p_wf(self, min_q, min_p_wh):
        pt = self.casing.calc_pt(
            h_start='top',
            p_mes = min_p_wh * 101325,
            flow_direction=1,
            q_liq=min_q/86400,
            extra_output=True
              )
        return self.tubing.calc_pt(
            h_start='top',
            p_mes = pt[0],
            flow_direction=1,
            q_liq=min_q/86400,
            extra_output=True
        )[0]/101325
        
    def find_sol(self, f_x: list, f_y: list, s_x: list, s_y: list):
        """
        Функция поиска точки пересечения
        
        ----------
        """
        first_line = LineString(np.column_stack((f_x, f_y)))
        second_line = LineString(np.column_stack((s_x, s_y)))
        try:
            intersection = first_line.intersection(second_line)
        except:
            first_line = first_line.buffer(0)
            intersection = first_line.intersection(second_line)
        if intersection.type == "MultiPoint":
            results = [(p.x, p.y) for p in intersection]
            return results[-1]
        try:
            x, y = intersection.xy
            return x[0], y[0]
        except (NotImplementedError, AttributeError) as e:
            return None, None
        
    def init_gl_well(self, data):
        pipe_data = {
            'casing': {'bottom_depth': data["reservoir"]["md_vdp"], 'd': data["pipe"]["casing"]["d"],
                      'roughness': data["pipe"]["casing"]["roughness"]
                      },
            'tubing': {'bottom_depth': data["pipe"]["tubing"]["md"], 'd': data["pipe"]["tubing"]["d"],
                      'roughness': data["pipe"]["tubing"]["roughness"]
                      }
        }
        well_trajectory_data = {'inclinometry': self.inclinometry}
        fluid_data = {
            'q_fluid': 100/86400, 
            'wct': self.wct,
            'pvt_model_data': {
                'black_oil': {
                    "gamma_gas": data["fluid"]["gamma_gas"], 
                        "gamma_wat": data["fluid"]["gamma_water"], 
                        "gamma_oil": data["fluid"]["gamma_oil"],
                        "rp": data["fluid"]["rp"]              
                }
            }

        }
        equipment_data = {
            "gl_system": {
                "valve1": {
                    "h_mes": self.md_valve,
                    "d": 0.006
                }
            }
        }
        self.well = GasLiftWell(
            fluid_data, 
            pipe_data, 
            well_trajectory_data,
            self.amb_dist, 
            equipment_data
        )
        
    def calc_gl_p_wf(self, q_liq, q_gas_inj):
        return self.well.calc_pwf_pfl(
            self.p_wh*101325, uc.convert_rate(q_liq, "m3/day", "m3/s"), 
            self.wct,
            q_gas_inj=uc.convert_rate(q_gas_inj, "m3/day", "m3/s")
        )/101325
        
    def calc_gl_vlp(self, q_gas_inj):
        return [self.calc_gl_p_wf(q_liq, q_gas_inj) for q_liq in self.q_liq_list]
        
    def calc_gl_curve(self):
        self.q_gas_inj_list = [_ for _ in range(5000, 150000, 10000)]
        self.gl_curve = []
        for q_gas_inj in self.q_gas_inj_list:
            print(f'расход: {q_gas_inj}')
            self.gl_curve.append(
                self.find_sol(
                    self.q_liq_list,
                    self.ipr_p_wf,
                    self.q_liq_list,
                    self.calc_gl_vlp(q_gas_inj)       
                )[0]
            )
        
    def find_max_gl_solution(self):
        self.calc_gl_curve()
        self.ax3.plot(self.q_gas_inj_list, self.gl_curve)
        self.ax3.plot(self.q_gas_inj_list[self.gl_curve.index(max(self.gl_curve))], max(self.gl_curve), "ro")
        self.ax3.set(xlabel='расход газлифтного газа, м3/сут', ylabel='дебит жидкости, м3/сут')
        return self.q_gas_inj_list[self.gl_curve.index(max(self.gl_curve))], max(self.gl_curve)
        
    
    
calc = Calc(data, data2)

with open('output.json', 'w', encoding='utf-8') as output_file:
        json.dump(
            {   
                "t1": {
            		"pwf": calc.min_p_wf
            	},
            	"t2": {
            		"q_inj": calc.q_inj_opt,
            		"q_liq": calc.q_liq_opt
            	}
            }, 
            output_file, 
            indent=4)
plt.show()

