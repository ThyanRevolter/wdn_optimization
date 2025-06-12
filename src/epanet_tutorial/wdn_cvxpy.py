import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from epanet_tutorial.simple_nr import WaterNetwork, Units
import pandas as pd
from datetime import datetime
from electric_emission_cost import costs

class DynamicWaterNetworkCVX():

    def __init__(self, inp_file_path:str, pump_data_path:str=None, reservoir_data_path:str=None):
        """
        Initialize the DynamicWaterNetworkCVX class.
        """
        self.wn = WaterNetwork(inp_file_path, units=Units.METRIC, round_to=3).wn
        self.start_dt = datetime(2025, 1, 1, 0, 0, 0)
        self.end_dt = datetime(2025, 1, 2, 0, 0, 0)
        self.n_time_steps = int((self.end_dt - self.start_dt).total_seconds() / 3600)
        self.time_steps = range(self.n_time_steps)
        if pump_data_path is not None:
            self.pump_data = pd.read_csv(pump_data_path, sep=",")
        else:
            self.pump_data = None
        if reservoir_data_path is not None:
            self.reservoir_data = pd.read_csv(reservoir_data_path, sep=",")
            self.reservoir_data["reservoir_name"] = self.reservoir_data["reservoir_name"].astype(str)
        else:
            self.reservoir_data = None
        self.rate_df = pd.read_csv("data/operational_data/tariff.csv", sep=",")
        self.binary_pump = False
        self.create_variables(binary_pump=self.binary_pump)
        self.set_demand_pattern_values()
        self.constraints = self.get_constraints()
        self.electricity_cost_objective = self.get_objective()

    def create_variables(self, binary_pump:bool=False):
        """
        Create variables for the model.

        Args:
            binary_pump (bool, optional): Whether to use binary pump constraints. Defaults to False.
        """
        # pipe flow variables for each pipe for each time step
        for pipe in self.wn["links"]:
            if pipe["link_type"] == "Pipe":
                setattr(self, f"pipe_flow_{pipe['name']}", cp.Variable(self.n_time_steps, name=f"pipe_flow_{pipe['name']}"))
        
        # pump flow variables for each pump for each time step
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                setattr(self, f"pump_flow_{pump['name']}", cp.Variable(self.n_time_steps, name=f"pump_flow_{pump['name']}"))
                setattr(self, f"pump_power_{pump['name']}", cp.Variable(self.n_time_steps, name=f"pump_power_{pump['name']}"))

                if binary_pump:
                    setattr(self, f"pump_on_status_var_{pump['name']}", cp.Variable(self.n_time_steps, name=f"pump_on_status_var_{pump['name']}", boolean=True))
                elif self.pump_data is not None:
                    state_powers = self.pump_data[f"pump_{pump['name']}_power"].values
                    state_powers = state_powers[~np.isnan(state_powers)]
                    for s in range(len(state_powers)):
                        setattr(self, f"pump_on_status_var_{pump['name']}_{s}", cp.Variable(self.n_time_steps, name=f"pump_on_status_var_{pump['name']}_{s}", boolean=True))
                else:
                    setattr(self, f"pump_on_status_var_{pump['name']}", cp.Variable(self.n_time_steps, name=f"pump_on_status_var_{pump['name']}"))

        # tank level variables for each tank for each time step
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                setattr(self, f"tank_level_{tank['name']}", cp.Variable(self.n_time_steps, name=f"tank_level_{tank['name']}"))
            elif tank["node_type"] == "Reservoir":
                setattr(self, f"reservoir_flow_{tank['name']}", cp.Variable(self.n_time_steps, name=f"reservoir_flow_{tank['name']}"))

        # total power at each time step
        setattr(self, "total_power", cp.Variable(self.n_time_steps, name="total_power"))

    def get_constraints(self):
        """
        Get all constraints for the model.

        Returns:
            dict: Dictionary containing all model constraints.
        """
        constraints = {}
        constraints.update(self.get_tank_level_constraints())
        constraints.update(self.get_nodal_flow_balance_constraints())
        constraints.update(self.get_tank_flow_balance_constraints())
        if self.pump_data is not None:
            constraints.update(self.get_pump_state_constraints())
            constraints.update(self.get_pump_flow_with_state_constraints())
            constraints.update(self.get_pump_power_with_state_constraints())
        else:
            constraints.update(self.get_pump_flow_constraints(binary_pump=self.binary_pump))
            constraints.update(self.get_pump_power_constraints())
        if self.reservoir_data is not None:
            constraints.update(self.get_reservoir_constraints())
        constraints.update(self.get_total_power_constraint())
        return constraints

    def get_demand_pattern(self, base_demand:float, pattern_name:str):
        """
        Get a demand pattern for a demand node.

        Args:
            base_demand (float): Base demand value.
            pattern_name (str): Name of the pattern to use.

        Returns:
            np.ndarray: Array of demand pattern values.
        """
        # get the pattern data which is for 24 hours and repeat it for the number of time steps
        pattern_data = [pattern["multipliers"] for pattern in self.wn["patterns"] if pattern["name"] == pattern_name][0]
        pattern_values = np.array(pattern_data)
        pattern_values = np.tile(pattern_values, self.n_time_steps // 24)
        return base_demand * pattern_values

    def set_demand_pattern_values(self):
        """
        Set demand pattern values for all demand nodes.
        """
        for demand_node in self.wn["nodes"]:
            if demand_node["node_type"] == "Junction" and demand_node["base_demand"] > 0:
                demand_pattern = self.get_demand_pattern(demand_node["base_demand"], demand_node["demand_pattern"])
                setattr(self, f"demand_pattern_{demand_node['name']}", demand_pattern)

    def get_tank_level_constraints(self):
        """
        Get tank level constraints with minimum, maximum and initial level constraints.

        Returns:
            dict: Dictionary containing tank level constraints.
        """
        tank_level_constraints = {}
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                print("tank['name']", tank["name"])
                print(tank)
                min_tank_level = tank["min_level"]
                max_tank_level = tank["max_level"]
                init_tank_level = tank["init_level"]
                # inital level constraint from inp file
                tank_level_constraints[f"tank_level_init_{tank['name']}"] = getattr(self, f"tank_level_{tank['name']}")[0] == init_tank_level
                tank_level_constraints[f"tank_level_final_{tank['name']}"] = getattr(self, f"tank_level_{tank['name']}")[-1] <= 1.1*init_tank_level
                tank_level_constraints[f"tank_level_final_{tank['name']}"] = getattr(self, f"tank_level_{tank['name']}")[-1] >= 0.9*init_tank_level
                tank_level_constraints[f"tank_level_min_{tank['name']}"] = getattr(self, f"tank_level_{tank['name']}") >= min_tank_level
                tank_level_constraints[f"tank_level_max_{tank['name']}"] = getattr(self, f"tank_level_{tank['name']}") <= max_tank_level
        return tank_level_constraints

    def get_nodal_flow_balance_constraints(self):
        """
        Get constraints for the nodal flow balance.

        Returns:
            dict: Dictionary containing nodal flow balance constraints.
        """
        nodal_flow_balance_constraints = {}
        for node in self.wn["nodes"]:
            if node["node_type"] == "Junction":
                flow_pipe_in = {}
                flow_pipe_out = {}
                flow_pump_in = {}
                flow_pump_out = {}
                for pipe in self.wn["links"]:
                    if pipe["link_type"] == "Pipe" and pipe["start_node_name"] == node["name"]:
                        flow_pipe_out[pipe["name"]] = getattr(self, f"pipe_flow_{pipe['name']}")
                    elif pipe["link_type"] == "Pipe" and pipe["end_node_name"] == node["name"]:
                        flow_pipe_in[pipe["name"]] = getattr(self, f"pipe_flow_{pipe['name']}")
                for pump in self.wn["links"]:
                    if pump["link_type"] == "Pump" and pump["start_node_name"] == node["name"]:
                        flow_pump_out[pump["name"]] = getattr(self, f"pump_flow_{pump['name']}")
                    elif pump["link_type"] == "Pump" and pump["end_node_name"] == node["name"]:
                        flow_pump_in[pump["name"]] = getattr(self, f"pump_flow_{pump['name']}")
                demand = getattr(self, f"demand_pattern_{node['name']}") if node["base_demand"] > 0 else 0
                flow_in = sum(flow_pipe_in.values()) + sum(flow_pump_in.values())
                flow_out = sum(flow_pipe_out.values()) + sum(flow_pump_out.values())
                nodal_flow_balance_constraints[f"nodal_flow_balance_equality_constraint_{node['name']}"] = flow_in == flow_out + demand
        return nodal_flow_balance_constraints

    def get_tank_flow_balance_constraints(self):
        """
        Get constraints for the tank flow balance.

        Returns:
            dict: Dictionary containing tank flow balance constraints.
        """
        tank_flow_balance_constraints = {}
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                tank_area = tank["diameter"]**2 * np.pi / 4
                flow_pipe_in = {}
                flow_pipe_out = {}
                flow_pump_in = {}
                flow_pump_out = {}
                for pipe in self.wn["links"]:
                    if pipe["link_type"] == "Pipe" and pipe["start_node_name"] == tank["name"]:
                        flow_pipe_out[pipe["name"]] = getattr(self, f"pipe_flow_{pipe['name']}")
                    elif pipe["link_type"] == "Pipe" and pipe["end_node_name"] == tank["name"]:
                        flow_pipe_in[pipe["name"]] = getattr(self, f"pipe_flow_{pipe['name']}")
                for pump in self.wn["links"]:
                    if pump["link_type"] == "Pump" and pump["start_node_name"] == tank["name"]:
                        flow_pump_out[pump["name"]] = getattr(self, f"pump_flow_{pump['name']}")
                    elif pump["link_type"] == "Pump" and pump["end_node_name"] == tank["name"]:
                        flow_pump_in[pump["name"]] = getattr(self, f"pump_flow_{pump['name']}")
                flow_in = sum(flow_pipe_in.values()) + sum(flow_pump_in.values())
                flow_out = sum(flow_pipe_out.values()) + sum(flow_pump_out.values())
                tank_flow_balance_constraints[f"tank_flow_balance_equality_constraint_{tank['name']}"] = (
                    getattr(self, f"tank_level_{tank['name']}")[1:] == (getattr(self, f"tank_level_{tank['name']}")[:-1] + ((flow_in - flow_out) / tank_area)[:-1])
                )
        return tank_flow_balance_constraints

    def get_pump_state_constraints(self):
        """
        Get constraints for the pump state such that the pump is on for only one state at each time step.

        Returns:
            dict: Dictionary containing pump state constraints.
        """
        pump_state_constraints = {}
        if self.pump_data is not None:
            for pump in self.wn["links"]:
                if pump["link_type"] == "Pump":
                    state_flows = self.pump_data[f"pump_{pump['name']}_flow"].values
                    state_flows = state_flows[~np.isnan(state_flows)]
                    binary_vars = {}
                    for s, _ in enumerate(state_flows):
                        binary_vars[s] = getattr(self, f"pump_on_status_var_{pump['name']}_{s}")
                    pump_state_constraints[f"pump_state_flow_activation_constraint_{pump['name']}"] = (
                        sum(binary_vars[s] for s, _ in enumerate(state_flows)) == 1
                    )
        return pump_state_constraints

    def get_pump_flow_with_state_constraints(self):
        """
        Get constraints for the pump flow with state.

        Returns:
            dict: Dictionary containing pump flow with state constraints.
        """
        pump_flow_with_state_constraints = {}
        if self.pump_data is not None:
            for pump in self.wn["links"]:
                if pump["link_type"] == "Pump":
                    state_flows = self.pump_data[f"pump_{pump['name']}_flow"].values
                    state_flows = state_flows[~np.isnan(state_flows)]
                    pump_flow_values = {}
                    for s, _ in enumerate(state_flows):
                        pump_flow_values[s] = getattr(self, f"pump_on_status_var_{pump['name']}_{s}") * state_flows[s]
                    pump_flow_with_state_constraints[f"pump_flow_with_state_equality_constraint_{pump['name']}"] = (
                        getattr(self, f"pump_flow_{pump['name']}") == sum(pump_flow_values.values())
                    )
        return pump_flow_with_state_constraints

    def get_pump_flow_constraints(self, binary_pump=False):
        """
        Get constraints for the pump flow.

        Args:
            binary_pump (bool, optional): Whether to use binary pump constraints. Defaults to False.

        Returns:
            dict: Dictionary containing pump flow constraints.
        """
        pump_flow_constraints = {}
        pump_capacity = 700.0
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                if not binary_pump:
                    pump_flow_constraints[f"pump_flow_constraint_on_relaxed_{pump['name']}"] = (
                        getattr(self, f"pump_on_status_var_{pump['name']}") <= 1
                    )                
                pump_flow_constraints[f"pump_flow_equality_constraint_{pump['name']}"] = (
                    getattr(self, f"pump_flow_{pump['name']}") == getattr(self, f"pump_on_status_var_{pump['name']}") * pump_capacity
                )
        return pump_flow_constraints

    def get_pump_power_constraints(self):
        """
        Get constraints for the pump power.

        Returns:
            dict: Dictionary containing pump power constraints.
        """
        pump_power_constraints = {}
        pump_power_capacity = 700.0
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                pump_power_constraints[f"pump_power_constraint_{pump['name']}"] = (
                    getattr(self, f"pump_power_{pump['name']}") == getattr(self, f"pump_on_status_var_{pump['name']}") * pump_power_capacity
                )
        return pump_power_constraints

    def get_pump_power_with_state_constraints(self):
        """
        Get constraints for the pump power with state.

        Returns:
            dict: Dictionary containing pump power with state constraints.
        """
        pump_power_with_state_constraints = {}
        if self.pump_data is not None:
            for pump in self.wn["links"]:
                if pump["link_type"] == "Pump":
                    state_powers = self.pump_data[f"pump_{pump['name']}_power"].values
                    state_powers = state_powers[~np.isnan(state_powers)]
                    pump_power_values = {}
                    for s, _ in enumerate(state_powers):
                        pump_power_values[s] = getattr(self, f"pump_on_status_var_{pump['name']}_{s}") * state_powers[s]
                    pump_power_with_state_constraints[f"pump_power_with_state_constraint_{pump['name']}"] = (
                        getattr(self, f"pump_power_{pump['name']}") == sum(pump_power_values.values())
                    )
        return pump_power_with_state_constraints

    def get_total_power_constraint(self):
        """
        Get constraints for the total power.

        Returns:
            dict: Dictionary containing total power constraints.
        """
        total_power_expression = {}
        pump_power_vars = {}
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                pump_power_vars[pump["name"]] = getattr(self, f"pump_power_{pump['name']}")
        total_power_expression[f"total_power_equality_constraint"] = (
            getattr(self, "total_power") == sum(pump_power_vars.values())
        )
        return total_power_expression

    def get_reservoir_constraints(self):
        """
        Get constraints for the reservoir.
        """
        reservoir_constraints = {}
        for reservoir in self.wn["nodes"]:
            if reservoir["node_type"] == "Reservoir":
                min_volume = self.reservoir_data.loc[self.reservoir_data["reservoir_name"] == reservoir["name"], "min_volume"].values[0]
                max_volume = self.reservoir_data.loc[self.reservoir_data["reservoir_name"] == reservoir["name"], "max_volume"].values[0]
                if ~np.isnan(min_volume):
                    reservoir_constraints[f"reservoir_volume_min_constraint_{reservoir['name']}"] = (
                        sum(getattr(self, f"reservoir_flow_{reservoir['name']}")) >= min_volume
                    )
                if ~np.isnan(max_volume):
                    reservoir_constraints[f"reservoir_volume_max_constraint_{reservoir['name']}"] = (
                        sum(getattr(self, f"reservoir_flow_{reservoir['name']}")) <= max_volume
                    )
        return reservoir_constraints

    def get_objective(self):
        """
        Get an objective function for the model.

        Returns:
            cp.Problem: The optimization problem with objective function.
        """
        self.charge_dict = costs.get_charge_dict(self.start_dt, self.end_dt, self.rate_df, resolution="1h")
        consumption_data_dict = {"electric": getattr(self, "total_power")}
        self.electricity_cost, _ = costs.calculate_cost(
            self.charge_dict,
            consumption_data_dict,
            resolution="1h",
            prev_demand_dict=None,
            prev_consumption_dict=None,
            consumption_estimate=0,
            desired_utility="electric",
            desired_charge_type=None,
        )
        electricity_cost_objective = cp.Minimize(self.electricity_cost)
        return electricity_cost_objective

    def solve(self, **solver_kwargs):
        """
        Solve the optimization problem.

        Returns:
            float: The optimal objective value.
        """
        self.problem = cp.Problem(self.electricity_cost_objective, list(self.constraints.values()))
        result = self.problem.solve(solver="GUROBI", **solver_kwargs)
        return result

    def package_data(self, save_to_csv:bool=False):
        """
        Package the results into a dataframe.

        Args:
            save_to_csv (bool, optional): Whether to save the results to a csv file. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the results.
        """
        time_range = pd.date_range(start=self.start_dt, end=self.end_dt, freq="1h")[:-1]
        results = {"Datetime": time_range}
        for pipe in self.wn["links"]:
            if pipe["link_type"] == "Pipe":
                results[f'pipe_flow_{pipe["name"]}'] = getattr(self, f"pipe_flow_{pipe['name']}").value
            elif pipe["link_type"] == "Pump":
                results[f'pump_flow_{pipe["name"]}'] = getattr(self, f"pump_flow_{pipe['name']}").value
                results[f'pump_power_{pipe["name"]}'] = getattr(self, f"pump_power_{pipe['name']}").value
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                results[f'tank_level_{tank["name"]}'] = getattr(self, f"tank_level_{tank['name']}").value
                results[f'tank_volume_{tank["name"]}'] = getattr(self, f"tank_level_{tank['name']}").value * tank["diameter"]**2 * np.pi / 4
                results[f'tank_min_volume_{tank["name"]}'] = tank["min_level"] * tank["diameter"]**2 * np.pi / 4 * np.ones(self.n_time_steps)
                results[f'tank_max_volume_{tank["name"]}'] = tank["max_level"] * tank["diameter"]**2 * np.pi / 4 * np.ones(self.n_time_steps)
            elif tank["node_type"] == "Junction" and tank["base_demand"] > 0:
                results[f'demand_{tank["name"]}'] = getattr(self, f"demand_pattern_{tank['name']}")
            elif tank["node_type"] == "Reservoir":
                results[f'reservoir_flow_{tank["name"]}'] = getattr(self, f"reservoir_flow_{tank['name']}").value

        results["total_power"] = getattr(self, "total_power").value
        results["electricity_charge"] = sum(self.charge_dict.values())
        results_df = pd.DataFrame(results)
        if save_to_csv:
            results_df.to_csv(f"data/local/operational_data/results_{self.start_dt.strftime('%Y%m%d')}_{self.end_dt.strftime('%Y%m%d')}.csv", index=False)
        return results_df

if __name__ == "__main__":
    # wdn = DynamicWaterNetworkCVX("data/epanet_networks/simple_pump_tank.inp", pump_data_path=None)
    inp_path = "data/epanet_networks/sopron_network.inp"
    pump_data_path = "data/operational_data/sopron_network_pump_data.csv"
    reservoir_data_path = "data/operational_data/sopron_network_reservoir_data.csv"
    wdn = DynamicWaterNetworkCVX(
        inp_path,
        pump_data_path=pump_data_path,
        reservoir_data_path=reservoir_data_path
    )
    wdn.solve(verbose=True, time_limit=30)
    wdn.package_data(save_to_csv=True)
