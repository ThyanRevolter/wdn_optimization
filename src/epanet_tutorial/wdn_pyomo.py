""""
This module implements a water distribution network model using Pyomo without any loss or friction terms.
"""

import pyomo.environ as pyo
import numpy as np
from pyomo.opt import SolverFactory
from pyomo.environ import value
from datetime import datetime
from epanet_tutorial.simple_nr import WaterNetwork, Units


class DynamicWaterNetwork():
    """
    A class to represent a simple dynamic water network.
    """
    def __init__(self, inp_file_path:str):
        self.n_time_steps = 24
        self.time_steps = range(self.n_time_steps)
        self.wn = WaterNetwork(inp_file_path, units=Units.IMPERIAL_CFS, round_to=3).wn
        self.model = pyo.ConcreteModel()
        self.create_model_variables()
        self.create_demand_constraints()
        self.create_tank_level_constraints()
        self.create_nodal_flow_balance_constraints()
        self.create_tank_flow_balance_constraints()
        self.create_pump_flow_constraints()
        self.create_power_variables()
        self.create_total_power_constraint()
        self.create_objective()
        self.results = None

    def create_model_variables(self):
        """
        Create a Pyomo model variables for the water network.
        """
        # pipe flow variables for each pipe for each time step
        for pipe in self.wn["links"]:
            if pipe["link_type"] == "Pipe":
                self.model.add_component(f"pipe_flow_{pipe['name']}", pyo.Var(self.time_steps, initialize=0))

        # pump flow variables for each pump for each time step
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                self.model.add_component(f"pump_flow_{pump['name']}", pyo.Var(self.time_steps, initialize=0))

        # tank level variables for each tank for each time step
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                self.model.add_component(f"tank_level_{tank['name']}", pyo.Var(self.time_steps, initialize=0))

        # demand variables for each demand node for each time step
        for demand_node in self.wn["nodes"]:
            if demand_node["node_type"] == "Junction" and demand_node["base_demand"] > 0:
                self.model.add_component(f"demand_{demand_node['name']}", pyo.Var(self.time_steps, initialize=0))


    def create_demand_pattern(self, base_demand:float, pattern_name:str):
        """
        Create a demand pattern for a demand node.
        """
        pattern_data = [pattern["multipliers"] for pattern in self.wn["patterns"] if pattern["name"] == pattern_name][0]
        pattern_values = np.array(pattern_data)
        return base_demand * pattern_values
    
    def create_demand_constraints(self):
        """
        Create constraints for the demand nodes.
        """
        for demand_node in self.wn["nodes"]:
            if demand_node["node_type"] == "Junction" and demand_node["base_demand"] > 0:
                demand_pattern = self.create_demand_pattern(demand_node["base_demand"], demand_node["demand_pattern"])
                self.model.add_component(f"demand_pattern_{demand_node['name']}", pyo.Param(self.time_steps, initialize=demand_pattern, mutable=True))
                for t in self.time_steps:
                    self.model.add_component(
                        f"demand_constraint_{demand_node['name']}_{t}", 
                        pyo.Constraint(expr=(
                            self.model.component(f"demand_{demand_node['name']}")[t] == self.model.component(f"demand_pattern_{demand_node['name']}")[t]
                        )))
                    
    def create_tank_level_constraints(self):
        """
        Create constraints for the tank levels.
        """
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                # initial level constraint
                self.model.add_component(
                    f"tank_level_init_{tank['name']}", 
                    pyo.Constraint(expr=(
                        self.model.component(f"tank_level_{tank['name']}")[0] == tank["init_level"]
                    )))
                for t in self.time_steps:
                    # minimum level constraint
                    self.model.add_component(
                        f"tank_level_min_{tank['name']}_{t}", 
                        pyo.Constraint(expr=(
                            self.model.component(f"tank_level_{tank['name']}")[t] >= tank["min_level"]
                        )))
                    # maximum level constraint
                    self.model.add_component(
                        f"tank_level_max_{tank['name']}_{t}", 
                        pyo.Constraint(expr=(
                            self.model.component(f"tank_level_{tank['name']}")[t] <= tank["max_level"]
                        )))
                    
    def create_nodal_flow_balance_constraints(self):
        """
        Create constraints for the nodal flow balance.
        """
        for node in self.wn["nodes"]:
            if node["node_type"] == "Junction":
                for t in self.time_steps:
                    flow_pipe_in = {}
                    flow_pipe_out = {}
                    # pipes to the node
                    for pipe in self.wn["links"]:
                        if pipe["link_type"] == "Pipe" and pipe["start_node_name"] == node["name"]:
                            flow_pipe_out[pipe["name"]] = self.model.component(f"pipe_flow_{pipe['name']}")[t]
                        elif pipe["link_type"] == "Pipe" and pipe["end_node_name"] == node["name"]:
                            flow_pipe_in[pipe["name"]] = self.model.component(f"pipe_flow_{pipe['name']}")[t]
                    
                    flow_pump_in = {}
                    flow_pump_out = {}
                    # pumps to the node
                    for pump in self.wn["links"]:
                        if pump["link_type"] == "Pump" and pump["start_node_name"] == node["name"]:
                            flow_pump_out[pump["name"]] = self.model.component(f"pump_flow_{pump['name']}")[t]
                        elif pump["link_type"] == "Pump" and pump["end_node_name"] == node["name"]:
                            flow_pump_in[pump["name"]] = self.model.component(f"pump_flow_{pump['name']}")[t]
                    demand = self.model.component(f"demand_{node['name']}")[t] if node["base_demand"] > 0 else 0
                    flow_in = sum(flow_pipe_in.values()) + sum(flow_pump_in.values())
                    flow_out = sum(flow_pipe_out.values()) + sum(flow_pump_out.values())
                    self.model.add_component(
                        f"nodal_flow_balance_{node['name']}_{t}", 
                        pyo.Constraint(expr=(
                            flow_in == flow_out + demand
                        )))
                    
    def create_tank_flow_balance_constraints(self):
        """
        Create constraints for the tank flow balance.
        """
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                self.model.add_component(
                    f"tank_area_{tank['name']}", 
                    pyo.Param(initialize=tank["diameter"]**2 * np.pi / 4, default=tank["diameter"]**2 * np.pi / 4)
                )
                for t in self.time_steps:
                    flow_pipe_in = {}
                    flow_pipe_out = {}
                    # pipes to the tank
                    for pipe in self.wn["links"]:
                        if pipe["link_type"] == "Pipe" and pipe["start_node_name"] == tank["name"]:
                            flow_pipe_out[pipe["name"]] = self.model.component(f"pipe_flow_{pipe['name']}")[t]
                        elif pipe["link_type"] == "Pipe" and pipe["end_node_name"] == tank["name"]:
                            flow_pipe_in[pipe["name"]] = self.model.component(f"pipe_flow_{pipe['name']}")[t]
                    flow_in = sum(flow_pipe_in.values()) * 0.133681 * 60 # convert gallons per second to cubic feet per hour
                    flow_out = sum(flow_pipe_out.values()) * 0.133681 * 60 # convert gallons per second to cubic feet per hour
                    # tank dynamic level constraint
                    if t > 0:
                        self.model.add_component(
                            f"tank_level_dynamic_{tank['name']}_{t}", 
                            pyo.Constraint(expr=(
                                self.model.component(f"tank_level_{tank['name']}")[t] == self.model.component(f"tank_level_{tank['name']}")[t-1] + ((flow_in - flow_out) / self.model.component(f"tank_area_{tank['name']}"))
                            )))
                        
    def create_pump_flow_constraints(self):
        """
        Create constraints for the pump flow.
        """
        self.model.pump_flow_capacity = pyo.Param(initialize=50, default=50)
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                # add binary variable for pump on/off
                self.model.add_component(
                    f"pump_on_status_{pump['name']}", 
                    pyo.Var(self.time_steps, initialize=0, domain=pyo.Binary)
                )
                for t in self.time_steps:
                    # pump flow constraint
                    self.model.add_component(
                        f"pump_flow_{pump['name']}_{t}", 
                        pyo.Constraint(expr=(
                            self.model.component(f"pump_flow_{pump['name']}")[t] == self.model.component(f"pump_on_status_{pump['name']}")[t] * self.model.pump_flow_capacity
                        )))
                    
    def create_power_variables(self):
        """
        Create variables for the power.
        """
        self.model.pump_power_capacity = pyo.Param(initialize=10, default=10)
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                self.model.add_component(
                    f"pump_power_{pump['name']}", 
                    pyo.Var(self.time_steps, initialize=0, domain=pyo.NonNegativeReals)
                )
                for t in self.time_steps:
                    # pump power constraint
                    self.model.add_component(
                        f"pump_power_constraint_{pump['name']}_{t}", 
                        pyo.Constraint(expr=(
                            self.model.component(f"pump_power_{pump['name']}")[t] == self.model.component(f"pump_on_status_{pump['name']}")[t] * self.model.pump_power_capacity
                        )))
        
    def create_total_power_constraint(self):
        """
        Create a variable for the total power at each time step and a constraint for it.
        """
        self.model.add_component(
            "total_power", 
            pyo.Var(self.time_steps, initialize=0, domain=pyo.NonNegativeReals)
        )
        # add all pump power variables to the total power constraint
        for t in self.time_steps:
            total_power_expression = {}
            for pump in self.wn["links"]:
                if pump["link_type"] == "Pump":
                    print("pump power variable", self.model.component(f'pump_power_{pump["name"]}')[t])
                    total_power_expression[pump["name"]] = self.model.component(f"pump_power_{pump['name']}")[t]
            self.model.add_component(
                f"total_power_constraint_{t}", 
                pyo.Constraint(expr=(
                    self.model.component("total_power")[t] == sum(total_power_expression.values())
                ))
            )

    def create_objective(self):
        """
        Create an objective function for the model.
        """
        charge_array = np.array(
            [0.1, 0.1, 0.1, 0.1]
            + [0.15, 0.15, 0.15, 0.15]
            + [0.2, 0.2, 0.2, 0.2]
            + [0.15, 0.15, 0.15, 0.15]
            + [0.1, 0.1, 0.1, 0.1]
            + [0.05, 0.05, 0.05, 0.05]
        )

        def init_charge_array(model, a, data=charge_array):
            return data[a]
        
        self.model.charge_array = pyo.Param(self.time_steps, within=pyo.Reals, initialize=init_charge_array, mutable=True)
        self.model.add_component(
            "objective", 
            pyo.Objective(expr=sum(
                [self.model.component("total_power")[t] * self.model.charge_array[t] for t in self.time_steps]
            ), sense=pyo.minimize)
        )

    def solve(self):
        """
        Solve the model.
        """
        solver = SolverFactory("gurobi")
        solver.options["MIPGap"] = 0.01
        solver.options["TimeLimit"] = 100
        solver.options["OptimalityTol"] = 1e-6
        results = solver.solve(
            self.model,
            tee=True,
        )
        self.results = results

    def print_model_info(self, save_to_file:bool=False):
        """
        Print model information including variables and constraints in a structured way and save it to a text file.
        Args:
            save_to_file (bool): If True, saves the output to a text file named 'model_info.txt'
        """
        # Create output string
        output = []
        output.append("\n" + "="*50)
        output.append("MODEL INFORMATION")
        output.append("="*50)
        
        # Print total number of variables and constraints
        output.append("\nMODEL STATISTICS:")
        output.append("-"*30)
        n_vars = len(self.model.component_map(pyo.Var))
        n_cons = len(self.model.component_map(pyo.Constraint))
        output.append(f"Total Variables: {n_vars}")
        output.append(f"Total Constraints: {n_cons}")
        
        # Print Variables
        output.append("\nVARIABLES:")
        output.append("-"*30)
        for var_name, var in self.model.component_map(pyo.Var).items():
            output.append(f"\n{var_name}:")
            output.append(f"  Type: {type(var).__name__}")
            if hasattr(var, 'index_set'):
                output.append(f"  Index Set: {var.index_set()}")
                # For indexed variables, get domain from first index
                if hasattr(var, '__getitem__'):
                    first_idx = next(iter(var.index_set()))
                    output.append(f"  Domain: {var[first_idx].domain}")
                else:
                    output.append(f"  Domain: {var.domain}")
        
        # Print Constraints
        output.append("\nCONSTRAINTS:")
        output.append("-"*30)
        for con_name, con in self.model.component_map(pyo.Constraint).items():
            output.append(f"\n{con_name}:")
            output.append(f"  Type: {type(con).__name__}")
            if hasattr(con, 'index_set'):
                output.append(f"  Index Set: {con.index_set()}")
            output.append("  Expression:")
            output.append(f"    {con.expr}")
        
        output.append("\n" + "="*50)
        
        # Print to console
        print("\n".join(output))
        
        # Save to file if requested
        if save_to_file:
            with open(f'model_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
                f.write("\n".join(output))

if __name__ == "__main__":
    wdn = DynamicWaterNetwork("data/epanet_networks/simple_pump_tank.inp")
    wdn.print_model_info(save_to_file=True)
    wdn.solve()
    print(wdn.model.component("total_power"))
    power_values = [value(wdn.model.component("total_power")[t]) for t in wdn.time_steps]
    print(power_values)
    for pipe in wdn.wn["links"]:
        print(f"Pipe/Pump {pipe['name']} flowing from {pipe['start_node_name']} to {pipe['end_node_name']}")
        if pipe["link_type"] == "Pipe":
            print("pipe flows")
            print(pipe["name"], [value(wdn.model.component(f"pipe_flow_{pipe['name']}")[t]) for t in wdn.time_steps])
        elif pipe["link_type"] == "Pump":
            print("pump flows")
            print(pipe["name"], [value(wdn.model.component(f"pump_flow_{pipe['name']}")[t]) for t in wdn.time_steps])
    for tank in wdn.wn["nodes"]:
        if tank["node_type"] == "Tank":
            print("tank levels")
            print(tank["name"], [value(wdn.model.component(f"tank_level_{tank['name']}")[t]) for t in wdn.time_steps])
    for demand_node in wdn.wn["nodes"]:
        if demand_node["node_type"] == "Junction" and demand_node["base_demand"] > 0:
            print(f"Demand node {demand_node['name']}")
            print("demand values")
            print(demand_node["name"], [value(wdn.model.component(f"demand_{demand_node['name']}")[t]) for t in wdn.time_steps])

