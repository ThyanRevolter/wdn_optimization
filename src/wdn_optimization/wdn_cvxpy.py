import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from wdn_optimization.simple_nr import WaterNetwork, Units
import pandas as pd
from datetime import datetime
from electric_emission_cost import costs
import json
import os


class DynamicWaterNetworkCVX:
    """A class for optimizing water distribution networks using CVXPY for dynamic operations."""
    
    @staticmethod
    def load_optimization_params(params_path: str) -> dict:
        """
        Load optimization parameters from a JSON file.
        
        Args:
            params_path (str): Path to the JSON file containing optimization parameters.
            
        Returns:
            dict: Dictionary containing optimization parameters.
        """
        with open(params_path, 'r') as f:
            params = json.load(f)
        return params

    def __init__(
        self,
        params_path: str,
    ):
        """
        Initialize the DynamicWaterNetworkCVX class.
        
        Args:
            params_path (str): Path to the optimization parameters JSON file.
        """
        # Load optimization parameters
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Parameters file not found: {params_path}")
            
        self.params = self.load_optimization_params(params_path)
        
        # Load required parameters from JSON
        inp_file_path = self.params.get('network_path')
        if not inp_file_path:
            raise ValueError("network_path must be specified in the parameters file")
            
        pump_data_path = self.params.get('pump_data_path')
        reservoir_data_path = self.params.get('reservoir_data_path')
        self.binary_pump = self.params.get('binary_pump', False)

        self.wn = WaterNetwork(inp_file_path, units=Units.METRIC, round_to=3).wn
        self.start_dt = datetime.strptime(self.params.get('start_date'), '%Y-%m-%d %H:%M:%S')
        self.end_dt = datetime.strptime(self.params.get('end_date'), '%Y-%m-%d %H:%M:%S')
        self.n_time_steps = int((self.end_dt - self.start_dt).total_seconds() / self.params.get('time_step'))
        self.time_steps = range(self.n_time_steps) # time steps in hours
        
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
        self.create_variables(binary_pump=self.binary_pump)
        self.set_demand_pattern_values()
        self.constraints = self.get_constraints()
        self.electricity_cost_objective = self.get_objective()

    def create_variables(self, binary_pump: bool = False):
        """
        Create variables for the model.

        Args:
            binary_pump (bool, optional): Whether to use binary pump constraints. Defaults to False.
        """
        # pipe flow variables for each pipe for each time step
        for pipe in self.wn["links"]:
            if pipe["link_type"] == "Pipe":
                setattr(
                    self,
                    f"pipe_flow_{pipe['name']}",
                    cp.Variable(self.n_time_steps, name=f"pipe_flow_{pipe['name']}"),
                )

        # pump flow variables for each pump for each time step
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                setattr(
                    self,
                    f"pump_flow_{pump['name']}",
                    cp.Variable(self.n_time_steps, name=f"pump_flow_{pump['name']}"),
                )
                setattr(
                    self,
                    f"pump_power_{pump['name']}",
                    cp.Variable(self.n_time_steps, name=f"pump_power_{pump['name']}"),
                )

                if binary_pump:
                    setattr(
                        self,
                        f"pump_on_status_var_{pump['name']}",
                        cp.Variable(
                            self.n_time_steps,
                            name=f"pump_on_status_var_{pump['name']}",
                            boolean=True,
                        ),
                    )
                elif self.pump_data is not None:
                    state_powers = self.pump_data[f"pump_{pump['name']}_power"].values
                    state_powers = state_powers[~np.isnan(state_powers)]
                    for s in range(len(state_powers)):
                        setattr(
                            self,
                            f"pump_on_status_var_{pump['name']}_{s}",
                            cp.Variable(
                                self.n_time_steps,
                                name=f"pump_on_status_var_{pump['name']}_{s}",
                                boolean=True,
                            ),
                        )
                else:
                    setattr(
                        self,
                        f"pump_on_status_var_{pump['name']}",
                        cp.Variable(
                            self.n_time_steps, name=f"pump_on_status_var_{pump['name']}"
                        ),
                    )

        # tank level variables for each tank for each time step
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                setattr(
                    self,
                    f"tank_level_{tank['name']}",
                    cp.Variable(self.n_time_steps, name=f"tank_level_{tank['name']}"),
                )
            elif tank["node_type"] == "Reservoir":
                setattr(
                    self,
                    f"reservoir_flow_{tank['name']}",
                    cp.Variable(
                        self.n_time_steps, name=f"reservoir_flow_{tank['name']}"
                    ),
                )

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
            constraints.update(
                self.get_pump_flow_constraints(binary_pump=self.binary_pump)
            )
            constraints.update(self.get_pump_power_constraints())
        constraints.update(self.get_pump_on_time_constraint())
        constraints.update(self.get_reservoir_constraints())
        constraints.update(self.get_total_power_constraint())
        return constraints

    def get_demand_pattern(self, base_demand: float, pattern_name: str):
        """
        Get a demand pattern for a demand node.

        Args:
            base_demand (float): Base demand value.
            pattern_name (str): Name of the pattern to use.

        Returns:
            np.ndarray: Array of demand pattern values.
        """
        # get the pattern data which is for 24 hours and repeat it for the number of time steps
        pattern_data = [
            pattern["multipliers"]
            for pattern in self.wn["patterns"]
            if pattern["name"] == pattern_name
        ][0]
        pattern_values = np.array(pattern_data)
        pattern_values = np.tile(pattern_values, self.n_time_steps // 24)
        return base_demand * pattern_values

    def set_demand_pattern_values(self):
        """
        Set demand pattern values for all demand nodes.
        """
        for demand_node in self.wn["nodes"]:
            if (
                demand_node["node_type"] == "Junction"
                and demand_node["base_demand"] > 0
            ):
                demand_pattern = self.get_demand_pattern(
                    demand_node["base_demand"], demand_node["demand_pattern"]
                )
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
                min_tank_level = tank["min_level"]
                max_tank_level = tank["max_level"]
                init_tank_level = tank["init_level"]
                # inital level constraint from inp file
                tank_level_constraints[f"tank_level_init_{tank['name']}"] = (
                    getattr(self, f"tank_level_{tank['name']}")[0] == init_tank_level
                )
                tank_level_constraints[f"tank_level_min_{tank['name']}"] = (
                    getattr(self, f"tank_level_{tank['name']}") >= min_tank_level
                )
                tank_level_constraints[f"tank_level_max_{tank['name']}"] = (
                    getattr(self, f"tank_level_{tank['name']}") <= max_tank_level
                )
        return tank_level_constraints

    def get_nodal_flow(self, link_type: str | list, flow_direction: str, node_name: str) -> dict:
        """
        Get flows for a specific node, link type, and flow direction.

        Args:
            link_type (str | list): Type of link(s) to consider (e.g., "Pipe", "Pump", or ["Pipe", "Pump"])
            flow_direction (str): Direction of flow ("in" or "out")
            node_name (str): Name of the node to get flows for

        Returns:
            dict: Dictionary of flows where keys are link names and values are flow variables
        """
        if isinstance(link_type, str):
            link_type = [link_type]
            
        flows = {}
        for link in self.wn["links"]:
            if link["link_type"] in link_type:
                if flow_direction == "in" and link["end_node_name"] == node_name:
                    flows[link["name"]] = getattr(self, f"{link['link_type'].lower()}_flow_{link['name']}")
                elif flow_direction == "out" and link["start_node_name"] == node_name:
                    flows[link["name"]] = getattr(self, f"{link['link_type'].lower()}_flow_{link['name']}")
        return flows

    def get_nodal_flow_balance_constraints(self):
        """
        Get constraints for the nodal flow balance.

        Returns:
            dict: Dictionary containing nodal flow balance constraints.
        """
        nodal_flow_balance_constraints = {}
        for node in self.wn["nodes"]:
            if node["node_type"] == "Junction":
                flow_in = sum(self.get_nodal_flow(["Pipe", "Pump"], "in", node["name"]).values())
                flow_out = sum(self.get_nodal_flow(["Pipe", "Pump"], "out", node["name"]).values())
                demand = (
                    getattr(self, f"demand_pattern_{node['name']}")
                    if node["base_demand"] > 0
                    else 0
                )
                nodal_flow_balance_constraints[
                    f"nodal_flow_balance_equality_constraint_{node['name']}"
                ] = (flow_in == flow_out + demand)
        return nodal_flow_balance_constraints

    def get_tank_flow_balance_constraints(self):
        """
        Get constraints for the tank flow balance.

        Returns:
            dict: Dictionary containing tank flow balance constraints.
        """
        tank_flow_balance_constraints = {}
        final_tank_level_deviation = self.params.get('final_tank_level_deviation', 0.1)
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                tank_area = tank["diameter"] ** 2 * np.pi / 4
                init_tank_level = tank["init_level"]
                flow_in = sum(self.get_nodal_flow(["Pipe", "Pump"], "in", tank["name"]).values())
                flow_out = sum(self.get_nodal_flow(["Pipe", "Pump"], "out", tank["name"]).values())
                tank_flow_balance_constraints[
                    f"tank_flow_balance_equality_constraint_{tank['name']}"
                ] = getattr(self, f"tank_level_{tank['name']}")[1:] == (
                    getattr(self, f"tank_level_{tank['name']}")[:-1]
                    + ((flow_in - flow_out) / tank_area)[:-1]
                )
                # final level and the flow should be within 10% of the initial level
                tank_flow_balance_constraints[
                    f"tank_level_final_max_{tank['name']}"
                ] = (
                    getattr(self, f"tank_level_{tank['name']}")[-1]
                    + ((flow_in - flow_out) / tank_area)[-1]    
                    <= (1 + final_tank_level_deviation) * init_tank_level
                )
                tank_flow_balance_constraints[
                    f"tank_level_final_min_{tank['name']}"
                ] = (
                    getattr(self, f"tank_level_{tank['name']}")[-1]
                    + ((flow_in - flow_out) / tank_area)[-1]
                    >= (1 - final_tank_level_deviation) * init_tank_level
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
                        binary_vars[s] = getattr(
                            self, f"pump_on_status_var_{pump['name']}_{s}"
                        )
                    pump_state_constraints[
                        f"pump_state_flow_activation_constraint_{pump['name']}"
                    ] = (sum(binary_vars[s] for s, _ in enumerate(state_flows)) == 1)
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
                        pump_flow_values[s] = (
                            getattr(self, f"pump_on_status_var_{pump['name']}_{s}")
                            * state_flows[s]
                        )
                    pump_flow_with_state_constraints[
                        f"pump_flow_with_state_equality_constraint_{pump['name']}"
                    ] = getattr(self, f"pump_flow_{pump['name']}") == sum(
                        pump_flow_values.values()
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
        pump_capacity = self.params.get('pump_flow_capacity', 700.0)
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                if not binary_pump:
                    pump_flow_constraints[
                        f"pump_flow_constraint_on_max_{pump['name']}"
                    ] = (getattr(self, f"pump_on_status_var_{pump['name']}") <= 1)
                    pump_flow_constraints[
                        f"pump_flow_constraint_on_min_{pump['name']}"
                    ] = (getattr(self, f"pump_on_status_var_{pump['name']}") >= 0)
                pump_flow_constraints[
                    f"pump_flow_equality_constraint_{pump['name']}"
                ] = (
                    getattr(self, f"pump_flow_{pump['name']}")
                    == getattr(self, f"pump_on_status_var_{pump['name']}")
                    * pump_capacity
                )
        return pump_flow_constraints

    def get_pump_power_constraints(self):
        """
        Get constraints for the pump power.

        Returns:
            dict: Dictionary containing pump power constraints.
        """
        pump_power_constraints = {}
        pump_power_capacity = self.params.get('pump_power_capacity', 700.0)
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                pump_power_constraints[f"pump_power_constraint_{pump['name']}"] = (
                    getattr(self, f"pump_power_{pump['name']}")
                    == getattr(self, f"pump_on_status_var_{pump['name']}")
                    * pump_power_capacity
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
                        pump_power_values[s] = (
                            getattr(self, f"pump_on_status_var_{pump['name']}_{s}")
                            * state_powers[s]
                        )
                    pump_power_with_state_constraints[
                        f"pump_power_with_state_constraint_{pump['name']}"
                    ] = getattr(self, f"pump_power_{pump['name']}") == sum(
                        pump_power_values.values()
                    )
        return pump_power_with_state_constraints

    def get_pump_on_time_constraint(self):
        """
        Get constraints for the pump on time.
        """
        pump_on_time_constraints = {}
        max_pump_on_time_per_day = 24
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                if self.pump_data is not None:
                    flows_states = self.pump_data[f"pump_{pump['name']}_flow"].values
                    flows_states = flows_states[~np.isnan(flows_states)]
                    num_days = self.n_time_steps // 24
                    for day in range(num_days):
                        pump_on_time_constraints[
                            f"pump_on_time_constraint_{pump['name']}_{day}"
                        ] = (
                            sum(
                                sum(
                                    getattr(self, f"pump_on_status_var_{pump['name']}_{s}")[day*24:(day+1)*24] 
                                    for s, _ in enumerate(flows_states)
                                )
                            )
                            <= max_pump_on_time_per_day
                        )
                else:
                    num_days = self.n_time_steps // 24
                    for day in range(num_days):
                        pump_on_time_constraints[
                            f"pump_on_time_constraint_{pump['name']}_{day}"
                        ] = (
                            sum(getattr(self, f"pump_on_status_var_{pump['name']}")[day*24:(day+1)*24])
                            <= max_pump_on_time_per_day
                        )
        return pump_on_time_constraints

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
                pump_power_vars[pump["name"]] = getattr(
                    self, f"pump_power_{pump['name']}"
                )
        total_power_expression[f"total_power_equality_constraint"] = getattr(
            self, "total_power"
        ) == sum(pump_power_vars.values())
        return total_power_expression
    
    def get_reservoir_flow_constraints(self):
        """
        Get constraints for the reservoir flow.
        """
        reservoir_flow_constraints = {}
        for reservoir in self.wn["nodes"]:
            if reservoir["node_type"] == "Reservoir":
                if self.reservoir_data is not None:
                    min_volume = self.reservoir_data.loc[
                        self.reservoir_data["reservoir_name"] == reservoir["name"],
                        "min_volume",
                    ].values[0]
                    max_volume = self.reservoir_data.loc[
                        self.reservoir_data["reservoir_name"] == reservoir["name"],
                        "max_volume",
                    ].values[0]
                    min_flow = self.reservoir_data.loc[
                        self.reservoir_data["reservoir_name"] == reservoir["name"],
                        "min_flow",
                    ].values[0]
                    max_flow = self.reservoir_data.loc[
                        self.reservoir_data["reservoir_name"] == reservoir["name"],
                        "max_flow",
                    ].values[0]
                    if ~np.isnan(min_volume):
                        reservoir_flow_constraints[
                            f"reservoir_volume_min_constraint_{reservoir['name']}"
                        ] = (
                            sum(getattr(self, f"reservoir_flow_{reservoir['name']}"))
                            >= min_volume
                        )
                    if ~np.isnan(max_volume):
                        reservoir_flow_constraints[
                            f"reservoir_volume_max_constraint_{reservoir['name']}"
                        ] = (
                            sum(getattr(self, f"reservoir_flow_{reservoir['name']}"))
                            <= max_volume
                        )
                    if ~np.isnan(min_flow):
                        reservoir_flow_constraints[
                            f"reservoir_min_flow_constraint_{reservoir['name']}"
                        ] = (getattr(self, f"reservoir_flow_{reservoir['name']}") >= min_flow)
                    if ~np.isnan(max_flow):
                        reservoir_flow_constraints[
                            f"reservoir_max_flow_constraint_{reservoir['name']}"
                        ] = (getattr(self, f"reservoir_flow_{reservoir['name']}") <= max_flow)
        return reservoir_flow_constraints
    
    def get_reservoir_constraints(self):
        """
        Get constraints for the reservoir.
        """
        reservoir_constraints = {}
        for reservoir in self.wn["nodes"]:
            if reservoir["node_type"] == "Reservoir":
                # reservoir flow should be equal to the difference between the inflow and outflow
                flow_in = sum(self.get_nodal_flow(["Pipe", "Pump"], "in", reservoir["name"]).values())
                flow_out = sum(self.get_nodal_flow(["Pipe", "Pump"], "out", reservoir["name"]).values())
                reservoir_constraints[
                    f"reservoir_flow_equality_constraint_{reservoir['name']}"
                ] = (
                    getattr(self, f"reservoir_flow_{reservoir['name']}")
                    == flow_out - flow_in
                )
                reservoir_constraints[
                    f"reservoir_flow_positive_constraint_{reservoir['name']}"
                ] = (getattr(self, f"reservoir_flow_{reservoir['name']}") >= 0)
                # reservoir volume should be within the min and max volume and the flow should be within the min and max flow
                if self.reservoir_data is not None:
                    reservoir_constraints.update(self.get_reservoir_flow_constraints())                
        return reservoir_constraints

    def get_objective(self):
        """
        Get an objective function for the model.

        Returns:
            cp.Problem: The optimization problem with objective function.
        """
        self.charge_dict = costs.get_charge_dict(
            self.start_dt, self.end_dt, self.rate_df, resolution="1h"
        )
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
        # Use parameters from JSON if available
        verbose = self.params.get('verbose', solver_kwargs.get('verbose', True))
        time_limit = self.params.get('time_limit', solver_kwargs.get('time_limit', 60))
        
        self.problem = cp.Problem(
            self.electricity_cost_objective, list(self.constraints.values())
        )
        result = self.problem.solve(solver="GUROBI", verbose=verbose, time_limit=time_limit)
        return result

    def package_data(self, save_to_csv: bool = False):
        """
        Package the results into a dataframe.

        Args:
            save_to_csv (bool, optional): Whether to save the results to a csv file. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame containing the results.
        """
        # Use parameter from JSON if available
        save_to_csv = self.params.get('save_to_csv', save_to_csv)
        
        time_range = pd.date_range(start=self.start_dt, end=self.end_dt, freq="1h")[:-1]
        results = {"Datetime": time_range}
        for pipe in self.wn["links"]:
            if pipe["link_type"] == "Pipe":
                results[f'pipe_flow_{pipe["name"]}'] = getattr(
                    self, f"pipe_flow_{pipe['name']}"
                ).value
            elif pipe["link_type"] == "Pump":
                results[f'pump_flow_{pipe["name"]}'] = getattr(
                    self, f"pump_flow_{pipe['name']}"
                ).value
                results[f'pump_power_{pipe["name"]}'] = getattr(
                    self, f"pump_power_{pipe['name']}"
                ).value
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                results[f'tank_level_{tank["name"]}'] = getattr(
                    self, f"tank_level_{tank['name']}"
                ).value
                results[f'tank_volume_{tank["name"]}'] = (
                    getattr(self, f"tank_level_{tank['name']}").value
                    * tank["diameter"] ** 2
                    * np.pi
                    / 4
                )
                results[f'tank_min_volume_{tank["name"]}'] = (
                    tank["min_level"]
                    * tank["diameter"] ** 2
                    * np.pi
                    / 4
                    * np.ones(self.n_time_steps)
                )
                results[f'tank_max_volume_{tank["name"]}'] = (
                    tank["max_level"]
                    * tank["diameter"] ** 2
                    * np.pi
                    / 4
                    * np.ones(self.n_time_steps)
                )
            elif tank["node_type"] == "Junction" and tank["base_demand"] > 0:
                results[f'demand_{tank["name"]}'] = getattr(
                    self, f"demand_pattern_{tank['name']}"
                )
            elif tank["node_type"] == "Reservoir":
                results[f'reservoir_flow_{tank["name"]}'] = getattr(
                    self, f"reservoir_flow_{tank['name']}"
                ).value

        results["total_power"] = getattr(self, "total_power").value
        results["electricity_charge"] = sum(self.charge_dict.values())
        results_df = pd.DataFrame(results)
        if save_to_csv:
            # Create directory if it doesn't exist
            os.makedirs("data/local/operational_data", exist_ok=True)
            results_df.to_csv(
                f"data/local/operational_data/results_{self.start_dt.strftime('%Y%m%d')}_{self.end_dt.strftime('%Y%m%d')}.csv",
                index=False,
            )
        return results_df

    def plot_results(self, packaged_data: pd.DataFrame, save_to_file: bool = False):
        """
        Plot the results.
        """
        # Use parameter from JSON if available
        save_to_file = self.params.get('save_plot_to_file', save_to_file)
        
        fig, axs = plt.subplots(6, 1, figsize=(10, 30))
        for pipe in self.wn["links"]:
            if pipe["link_type"] == "Pipe":
                axs[0].plot(
                    packaged_data["Datetime"],
                    packaged_data[f'pipe_flow_{pipe["name"]}'],
                    label=pipe["name"],
                )
        axs[0].legend()
        axs[0].set_title("Pipe Flows")
        axs[0].set_ylabel("Flow (m³/h)")
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                axs[1].step(
                    packaged_data["Datetime"],
                    packaged_data[f'pump_flow_{pump["name"]}'],
                    label=pump["name"],
                    where="post",
                )
        axs[1].legend()
        axs[1].set_title("Pump Flows")
        axs[1].set_ylabel("Flow (m³/h)")
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                axs[2].plot(
                    packaged_data["Datetime"],
                    packaged_data[f'tank_level_{tank["name"]}'],
                    label=tank["name"],
                )
        axs[2].legend()
        axs[2].set_title("Tank Levels")
        axs[2].set_ylabel("Level (m)")
        for demand_node in self.wn["nodes"]:
            if demand_node["node_type"] == "Junction" and demand_node["base_demand"] > 0:
                axs[3].plot(
                    packaged_data["Datetime"],
                    packaged_data[f'demand_{demand_node["name"]}'],
                    label=demand_node["name"],
                )
        axs[3].legend()
        axs[3].set_title("Demand")
        axs[3].set_ylabel("Demand (m³/h)")
        axs[4].step(
            packaged_data["Datetime"],
            packaged_data["total_power"],
            label="Total Power",
            where="post",
        )
        axs[4].legend()
        axs[4].set_title("Power")
        axs[4].set_ylabel("Power (kW)")
        axs[5].step(
            packaged_data["Datetime"],
            packaged_data["electricity_charge"],
            label="electricity charges",
            where="post",
        )
        axs[5].legend()
        axs[5].set_title("Electricity Charges")
        axs[5].set_ylabel("Electricity Charges ($/kWh)")
        plt.tight_layout()
        if save_to_file:
            plt.savefig(f"data/local/plots/results_{self.start_dt.strftime('%Y%m%d')}_{self.end_dt.strftime('%Y%m%d')}.png")
        else:
            plt.show()
        return fig, axs

    def get_pump_on_times(self, pump_name: str):
        """
        Get the on times for a pump.
        """
        pump_on_times = {}
        for day in range(self.n_time_steps // 24):
            if self.pump_data is not None:
                flow_states = self.pump_data[f"pump_{pump_name}_flow"].values
                flow_states = flow_states[~np.isnan(flow_states)]
                for s, _ in enumerate(flow_states):
                    pump_on_times[f"state: {s} day: {day}"] = sum(getattr(self, f"pump_on_status_var_{pump_name}_{s}").value[day*24:(day+1)*24])
            else:
                for day in range(self.n_time_steps // 24):
                    pump_on_times[f"day: {day}"] = sum(getattr(self, f"pump_on_status_var_{pump_name}").value[day*24:(day+1)*24])
        return pump_on_times

    def print_optimization_result(self):
        """
        Get the optimization result.
        """
        print(f"Optimization result: {self.problem.status}")
        print(f"Objective value: {self.problem.value}")
        print(f"Solution time: {self.problem.solver_stats.solve_time}")

        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                print(f"Pump {pump['name']} on times per day: \n {self.get_pump_on_times(pump['name'])}")

if __name__ == "__main__":
    # Example usage with parameters from JSON
    params_path = "data/simple_pump_tank_network_opt_params.json"
    wdn = DynamicWaterNetworkCVX(params_path=params_path)
    

    print("Constraints:")
    for constraint_name, constraint in wdn.constraints.items():
        print(constraint_name)
        print(constraint)
        print("-"*100)
    print("-"*100)
    print("Objective function:")
    print(wdn.electricity_cost_objective)
    print("-"*100)

    
        
    wdn.solve()
    wdn.print_optimization_result()
    packaged_data = wdn.package_data(save_to_csv=True)
    wdn.plot_results(packaged_data)
