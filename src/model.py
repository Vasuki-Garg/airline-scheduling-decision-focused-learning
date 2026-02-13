# This code has been adapted from: https://github.com/harrylui1995/ASP_E2EPO , the official GitHub Repo for Lui, G. N., & Demirel, S. (2025). Gradient-based smart predict-then-optimize framework for aircraft arrival scheduling problem. Journal of Open Aviation Science, 2(2). https://doi.org/10.59490/joas.2024.7891


import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from tabulate import tabulate
from pyepo.model.grb import optGrbModel

def WVseparation(A1, A2):
    if A1 == 'J' and A2 == 'H':
        return 120
    elif A1 == 'J' and A2 == 'M':
        return 180
    elif A1 == 'J' and A2 == 'L':
        return 240
    elif A1 == 'H' and A2 == 'H':
        return 120
    elif A1 == 'H' and A2 == 'M':
        return 120
    elif A1 == 'H' and A2 == 'L':
        return 180
    elif A1 == 'M' and A2 == 'L':
        return 180
    else:
        return 120

class ASPmodel(optGrbModel):
    def __init__(self, n_aircraft, E, L, size, T):
        self.n_aircraft = n_aircraft
        self.aircraft = list(range(n_aircraft))

        self.E = E
        self.L = L
        self.size = size
        self.T = T
        self.S = {(i, j): WVseparation(self.size[i], self.size[j])
                  for i in self.aircraft for j in self.aircraft if i != j}
        self.cost_vector = None
        super().__init__()

    @property
    def num_cost(self):
        return len(self.aircraft)

    def copy(self):
        return type(self)(self.n_aircraft, self.E, self.L, self.size, self.T)

    def _getModel(self):
        m = gp.Model("ASP")
        self.x = m.addVars(self.n_aircraft, vtype=GRB.BINARY, name="x")
        self.delta = m.addVars(self.n_aircraft, self.n_aircraft, vtype=GRB.BINARY, name="delta")
        self.y = m.addVars(self.n_aircraft, vtype=GRB.CONTINUOUS, name="y")

        m.modelSense = GRB.MINIMIZE

        m.addConstrs((self.y[i] >= self.E[i] for i in self.aircraft), "earliest_landing_time")
        m.addConstrs((self.y[i] <= self.L[i] for i in self.aircraft), "latest_landing_time")

        bigM = max(self.L.values()) + max(self.S.values())
        m.addConstrs((self.y[i] - self.T[i] <= bigM * self.x[i] for i in self.aircraft), "X_definition_1")
        m.addConstrs((self.y[i] - self.T[i] >= -bigM * (1 - self.x[i]) for i in self.aircraft), "X_definition_2")

        m.addConstrs((self.delta[i, j] + self.delta[j, i] == 1
                      for i in self.aircraft for j in self.aircraft if i != j), "delta_sum")

        m.addConstrs((self.y[j] - self.y[i] >= self.S[i, j] - bigM * self.delta[j, i]
                      for i in self.aircraft for j in self.aircraft if i != j), "separation")

        return m, self.x

    def setObj(self, c):
        if isinstance(c, (float, int)):
            c = [c] * self.num_cost
        elif isinstance(c, list):
            c = np.array(c)

        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match vars.")

        self.cost_vector = c
        obj = gp.quicksum(c[i] * self.x[i] for i in self.aircraft)
        self._model.setObjective(obj)

    def solve(self):
        self._model.update()
        self._model.optimize()

        if self._model.status == GRB.OPTIMAL:
            sol = np.array([self.x[i].x > 0.5 for i in self.aircraft], dtype=np.uint8)
            obj_val = self._model.objVal
            self._solution_data = {
                'landing_times': {i: self.y[i].x for i in self.aircraft},
                'landing_order': sorted(self.aircraft, key=lambda i: self.y[i].x)
            }
            return sol, obj_val
        raise ValueError(f"Optimization failed with status {self._model.status}")

    def fcfs_simulation(self):
        if self.cost_vector is None:
            raise ValueError("Cost vector not set. Please call setObj() before running simulation.")

        sorted_planes = sorted(self.aircraft, key=lambda x: self.E[x])
        landing_times = {}
        total_cost = 0
        current_time = min(self.E.values())

        for i, plane in enumerate(sorted_planes):
            landing_time = max(current_time, self.E[plane])
            landing_times[plane] = landing_time
            if landing_time > self.T[plane]:
                total_cost += self.cost_vector[plane]
            if i < len(sorted_planes) - 1:
                next_plane = sorted_planes[i + 1]
                current_time = landing_time + self.S[plane, next_plane]
        return landing_times, total_cost

    def post_analysis(self):
        if self.cost_vector is None:
            raise ValueError("Cost vector not set. Please call setObj() before running post-analysis.")
        if not hasattr(self, '_solution_data'):
            raise ValueError("Model has not been solved. Please call solve() before running post-analysis.")

        opt_landing_times = self._solution_data['landing_times']
        opt_obj_val = self._model.objVal

        fcfs_landing_times, fcfs_total_cost = self.fcfs_simulation()

        data = []
        for i in self.aircraft:
            opt_late = opt_landing_times[i] > self.T[i]
            fcfs_late = fcfs_landing_times[i] > self.T[i]
            data.append({
                'Plane': i,
                'Opt Landing Time': f"{opt_landing_times[i]:.2f}",
                'FCFS Landing Time': f"{fcfs_landing_times[i]:.2f}",
                'Opt Landed After Target': 'Yes' if opt_late else 'No',
                'FCFS Landed After Target': 'Yes' if fcfs_late else 'No',
                'Time Window': f"[{self.E[i]}, {self.L[i]}]",
                'Target Time': self.T[i],
                'Cost': self.cost_vector[i] if opt_late else 0
            })

        df = pd.DataFrame(data)
        opt_late_landings = sum(1 for i in self.aircraft if opt_landing_times[i] > self.T[i])
        fcfs_late_landings = sum(1 for i in self.aircraft if fcfs_landing_times[i] > self.T[i])

        analysis = {
            'summary': tabulate(df, headers='keys', tablefmt='pretty'),
            'optimized_cost': opt_obj_val,
            'fcfs_cost': fcfs_total_cost,
            'opt_late_landings': opt_late_landings,
            'fcfs_late_landings': fcfs_late_landings
        }

        if fcfs_total_cost != 0 and opt_obj_val != 0:
            cost_diff = fcfs_total_cost - opt_obj_val
            cost_base = max(abs(fcfs_total_cost), abs(opt_obj_val))
            analysis['cost_improvement'] = f"{(cost_diff / cost_base) * 100:.2f}%"
        elif fcfs_total_cost == 0 and opt_obj_val == 0:
            analysis['cost_improvement'] = "0.00% (Both costs are 0)"
        else:
            analysis['cost_improvement'] = "100.00% (One cost is 0)"

        late_diff = fcfs_late_landings - opt_late_landings
        late_base = max(fcfs_late_landings, opt_late_landings, 1)
        analysis['late_landings_improvement'] = f"{(late_diff / late_base) * 100:.2f}%"
        return analysis
