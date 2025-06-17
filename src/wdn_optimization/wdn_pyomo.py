"""
This module implements a water distribution network model using Pyomo without any loss or friction terms.
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.environ import value
from datetime import datetime
from wdn_optimization.simple_nr import WaterNetwork, Units
from electric_emission_cost import costs
import matplotlib.pyplot as plt
import os
import json


class DynamicWaterNetwork:
    """
    A class to represent a simple dynamic water network.
    """

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

    def __init__(self, params_path: str):
        """
        Initialize the DynamicWaterNetwork class.
        
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
        self.model = pyo.ConcreteModel()
        self.create_model_variables()
        self.create_demand_parameters()
        self.create_tank_level_constraints()
        self.create_nodal_flow_balance_constraints()
        self.create_tank_flow_balance_constraints()
        if self.pump_data is not None:
            self.create_pump_state_constraints()
            self.create_pump_flow_with_state_constraints()
            self.create_pump_power_with_state_constraints()
        else:
            self.create_pump_flow_constraints(binary_pump=self.binary_pump)
            self.create_pump_power_constraints()
        self.create_pump_on_time_constraint()
        self.create_total_power_constraint()
        if self.reservoir_data is not None:
            self.create_reservoir_constraints()
        self.results = None
        self.create_objective()

    def create_model_variables(self, binary_pump: bool = False):
        """
        Create a Pyomo model variables for the water network.
        """

        self.model.t = range(self.n_time_steps)

        # pipe flow variables for each pipe for each time step
        for pipe in self.wn["links"]:
            if pipe["link_type"] == "Pipe":
                self.model.add_component(
                    f"pipe_flow_{pipe['name']}", pyo.Var(self.time_steps, initialize=0)
                )

        # pump flow variables for each pump for each time step
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                self.model.add_component(
                    f"pump_flow_{pump['name']}", pyo.Var(self.time_steps, initialize=0)
                )
                self.model.add_component(
                    f"pump_power_{pump['name']}", pyo.Var(self.time_steps, initialize=0)
                )
                if binary_pump:
                    self.model.add_component(
                        f"pump_on_status_var_{pump['name']}",
                        pyo.Var(self.time_steps, initialize=0, domain=pyo.Binary),
                    )
                elif self.pump_data is not None:
                    state_powers = self.pump_data[f"pump_{pump['name']}_power"].values
                    state_powers = state_powers[~np.isnan(state_powers)]
                    for s in range(len(state_powers)):
                        self.model.add_component(
                            f"pump_on_status_var_{pump['name']}_{s}",
                            pyo.Var(self.time_steps, initialize=0, domain=pyo.Binary),
                        )
                else:
                    self.model.add_component(
                        f"pump_on_status_var_{pump['name']}",
                        pyo.Var(
                            self.time_steps, initialize=0, domain=pyo.NonNegativeReals
                        ),
                    )

        # tank level variables for each tank for each time step
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                self.model.add_component(
                    f"tank_level_{tank['name']}", pyo.Var(self.time_steps, initialize=0)
                )

        # reservoir flow variables for each reservoir for each time step
        for reservoir in self.wn["nodes"]:
            if reservoir["node_type"] == "Reservoir":
                self.model.add_component(
                    f"reservoir_flow_{reservoir['name']}", pyo.Var(self.time_steps, initialize=0)
                )

        self.model.add_component(
            "total_power",
            pyo.Var(self.time_steps, initialize=0, domain=pyo.NonNegativeReals),
        )

    def create_demand_pattern(self, base_demand: float, pattern_name: str):
        """
        Create a demand pattern for a demand node.
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

    def create_demand_parameters(self):
        """
        Create constraints for the demand nodes.
        """
        for demand_node in self.wn["nodes"]:
            if (
                demand_node["node_type"] == "Junction"
                and demand_node["base_demand"] > 0
            ):
                demand_pattern = self.create_demand_pattern(
                    demand_node["base_demand"], demand_node["demand_pattern"]
                )
                self.model.add_component(
                    f"demand_pattern_{demand_node['name']}",
                    pyo.Param(self.time_steps, initialize=demand_pattern, mutable=True),
                )

    def create_tank_level_constraints(self):
        """
        Create constraints for the tank levels.
        """
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                self.model.add_component(
                    f"min_tank_level_{tank['name']}",
                    pyo.Param(initialize=tank["min_level"], mutable=True),
                )
                self.model.add_component(
                    f"max_tank_level_{tank['name']}",
                    pyo.Param(initialize=tank["max_level"], mutable=True),
                )
                self.model.add_component(
                    f"init_tank_level_{tank['name']}",
                    pyo.Param(initialize=tank["init_level"], mutable=True),
                )
                # initial level constraint from inp file
                self.model.add_component(
                    f"tank_level_init_{tank['name']}",
                    pyo.Constraint(
                        expr=(
                            self.model.component(f"tank_level_{tank['name']}")[0]
                            == self.model.component(f"init_tank_level_{tank['name']}")
                        )
                    ),
                )
                for t in self.time_steps:
                    # minimum level constraint
                    self.model.add_component(
                        f"tank_level_min_{tank['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"tank_level_{tank['name']}")[t]
                                >= self.model.component(
                                    f"min_tank_level_{tank['name']}"
                                )
                            )
                        ),
                    )
                    # maximum level constraint
                    self.model.add_component(
                        f"tank_level_max_{tank['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"tank_level_{tank['name']}")[t]
                                <= self.model.component(
                                    f"max_tank_level_{tank['name']}"
                                )
                            )
                        ),
                    )

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
                        if (
                            pipe["link_type"] == "Pipe"
                            and pipe["start_node_name"] == node["name"]
                        ):
                            flow_pipe_out[pipe["name"]] = self.model.component(
                                f"pipe_flow_{pipe['name']}"
                            )[t]
                        elif (
                            pipe["link_type"] == "Pipe"
                            and pipe["end_node_name"] == node["name"]
                        ):
                            flow_pipe_in[pipe["name"]] = self.model.component(
                                f"pipe_flow_{pipe['name']}"
                            )[t]

                    flow_pump_in = {}
                    flow_pump_out = {}
                    # pumps to the node
                    for pump in self.wn["links"]:
                        if (
                            pump["link_type"] == "Pump"
                            and pump["start_node_name"] == node["name"]
                        ):
                            flow_pump_out[pump["name"]] = self.model.component(
                                f"pump_flow_{pump['name']}"
                            )[t]
                        elif (
                            pump["link_type"] == "Pump"
                            and pump["end_node_name"] == node["name"]
                        ):
                            flow_pump_in[pump["name"]] = self.model.component(
                                f"pump_flow_{pump['name']}"
                            )[t]
                    demand = (
                        self.model.component(f"demand_pattern_{node['name']}")[t]
                        if node["base_demand"] > 0
                        else 0
                    )
                    flow_in = sum(flow_pipe_in.values()) + sum(flow_pump_in.values())
                    flow_out = sum(flow_pipe_out.values()) + sum(flow_pump_out.values())
                    self.model.add_component(
                        f"nodal_flow_balance_{node['name']}_{t}",
                        pyo.Constraint(expr=(flow_in == flow_out + demand)),
                    )

    def create_tank_flow_balance_constraints(self):
        """
        Create constraints for the tank flow balance.
        """
        self.model.final_tank_level_deviation = pyo.Param(initialize=self.params.get('final_tank_level_deviation', 0.1), mutable=True)
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                self.model.add_component(
                    f"tank_area_{tank['name']}",
                    pyo.Param(
                        initialize=tank["diameter"] ** 2 * np.pi / 4,
                        default=tank["diameter"] ** 2 * np.pi / 4,
                    ),
                )
                for t in self.time_steps:
                    flow_pipe_in = {}
                    flow_pipe_out = {}
                    # pipes to the tank
                    for pipe in self.wn["links"]:
                        if (
                            pipe["link_type"] == "Pipe"
                            and pipe["start_node_name"] == tank["name"]
                        ):
                            flow_pipe_out[pipe["name"]] = self.model.component(
                                f"pipe_flow_{pipe['name']}"
                            )[t]
                        elif (
                            pipe["link_type"] == "Pipe"
                            and pipe["end_node_name"] == tank["name"]
                        ):
                            flow_pipe_in[pipe["name"]] = self.model.component(
                                f"pipe_flow_{pipe['name']}"
                            )[t]
                    flow_in = sum(flow_pipe_in.values())
                    flow_out = sum(flow_pipe_out.values())
                    # tank dynamic level constraint
                    if t < self.n_time_steps - 1:
                        self.model.add_component(
                            f"tank_level_dynamic_{tank['name']}_{t+1}",
                            pyo.Constraint(
                                expr=(
                                    self.model.component(f"tank_level_{tank['name']}")[
                                        t + 1
                                    ]
                                    == self.model.component(
                                        f"tank_level_{tank['name']}"
                                    )[t]
                                    + (
                                        (flow_in - flow_out)
                                        / self.model.component(
                                            f"tank_area_{tank['name']}"
                                        )
                                    )
                                )
                            ),
                        )
                    else:
                        # final level and the flow should be within deviation of the initial level
                        init_level = self.model.component(f"init_tank_level_{tank['name']}")
                        self.model.add_component(
                            f"tank_level_final_max_{tank['name']}",
                            pyo.Constraint(
                                expr=(
                                    self.model.component(f"tank_level_{tank['name']}")[t]
                                    + ((flow_in - flow_out) / self.model.component(f"tank_area_{tank['name']}"))
                                    <= (1 + self.model.final_tank_level_deviation) * init_level
                                )
                            ),
                        )
                        self.model.add_component(
                            f"tank_level_final_min_{tank['name']}",
                            pyo.Constraint(
                                expr=(
                                    self.model.component(f"tank_level_{tank['name']}")[t]
                                    + ((flow_in - flow_out) / self.model.component(f"tank_area_{tank['name']}"))
                                    >= (1 - self.model.final_tank_level_deviation) * init_level
                                )
                            ),
                        )

    def create_pump_state_constraints(self):
        """
        Create constraints for the pump state.
        """
        if self.pump_data is None:
            return

        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                state_flows = self.pump_data[f"pump_{pump['name']}_flow"].values
                state_flows = state_flows[~np.isnan(state_flows)]
                state_powers = self.pump_data[f"pump_{pump['name']}_power"].values
                state_powers = state_powers[~np.isnan(state_powers)]
                # create a binary variable for each flow state for each time step
                binary_vars = {}
                for s, flow_state in enumerate(state_flows):
                    self.model.add_component(
                        f"pump_state_flow_value_{pump['name']}_{s}",
                        pyo.Param(initialize=flow_state, mutable=True),
                    )
                    self.model.add_component(
                        f"pump_state_power_value_{pump['name']}_{s}",
                        pyo.Param(initialize=state_powers[s], mutable=True),
                    )
                    binary_vars[s] = self.model.component(
                        f"pump_on_status_var_{pump['name']}_{s}"
                    )
                # create a constraint for each time step so that the sum of the binary variables is 1
                for t in self.time_steps:
                    self.model.add_component(
                        f"pump_flow_state_constraint_{pump['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                sum(binary_vars[s][t] for s in range(len(state_flows)))
                                == 1
                            )
                        ),
                    )

    def create_pump_flow_with_state_constraints(self):
        """
        Create constraints for the pump flow with state.
        """
        if self.pump_data is None:
            return

        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                state_flows = self.pump_data[f"pump_{pump['name']}_flow"].values
                state_flows = state_flows[~np.isnan(state_flows)]
                for t in self.time_steps:
                    self.model.add_component(
                        f"pump_flow_with_state_{pump['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"pump_flow_{pump['name']}")[t]
                                == sum(
                                    self.model.component(
                                        f"pump_on_status_var_{pump['name']}_{s}"
                                    )[t]
                                    * self.model.component(
                                        f"pump_state_flow_value_{pump['name']}_{s}"
                                    )
                                    for s in range(len(state_flows))
                                )
                            )
                        ),
                    )

    def create_pump_flow_constraints(self, binary_pump=False):
        """
        Create constraints for the pump flow.
        """
        self.model.pump_flow_capacity = pyo.Param(initialize=self.params.get('pump_flow_capacity', 700), mutable=True)
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                for t in self.time_steps:
                    if not binary_pump:
                        self.model.add_component(
                            f"pump_on_status_constraint_max_{pump['name']}_{t}",
                            pyo.Constraint(
                                expr=(
                                    self.model.component(
                                        f"pump_on_status_var_{pump['name']}"
                                    )[t]
                                    <= 1
                                )
                            ),
                        )
                        self.model.add_component(
                            f"pump_on_status_constraint_min_{pump['name']}_{t}",
                            pyo.Constraint(
                                expr=(
                                    self.model.component(
                                        f"pump_on_status_var_{pump['name']}"
                                    )[t]
                                    >= 0
                                )
                            ),
                        )
                    # pump flow constraint
                    self.model.add_component(
                        f"pump_flow_constraint_{pump['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"pump_flow_{pump['name']}")[t]
                                == self.model.component(
                                    f"pump_on_status_var_{pump['name']}"
                                )[t]
                                * self.model.pump_flow_capacity
                            )
                        ),
                    )

    def create_pump_on_time_constraint(self):
        """
        Create constraints for the pump on/off status.
        """
        self.model.pump_max_on_time_per_day = pyo.Param(initialize=24, mutable=True)
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                if self.pump_data is not None:
                    flows_states = self.pump_data[f"pump_{pump['name']}_flow"].values
                    flows_states = flows_states[~np.isnan(flows_states)]
                    num_days = self.n_time_steps // 24
                    for day in range(num_days):
                        self.model.add_component(
                            f"max_pump_on_time_constraint_{pump['name']}_{day}",
                            pyo.Constraint(
                                expr=(
                                    sum(
                                        sum(
                                            self.model.component(
                                                f"pump_on_status_var_{pump['name']}_{s}"
                                            )[t]
                                            for s in range(len(flows_states))
                                        )
                                        for t in range(day * 24, (day + 1) * 24)
                                    )
                                    <= self.model.pump_max_on_time_per_day
                                )
                            ),
                        )
                else:
                    num_days = self.n_time_steps // 24
                    for day in range(num_days):
                        self.model.add_component(
                            f"max_pump_on_time_constraint_{pump['name']}_{day}",
                            pyo.Constraint(
                                expr=(
                                    sum(
                                        self.model.component(
                                            f"pump_on_status_var_{pump['name']}"
                                        )[t]
                                        for t in range(day * 24, (day + 1) * 24)
                                    )
                                    <= self.model.pump_max_on_time_per_day
                                )
                            ),
                        )

    def create_pump_power_constraints(self):
        """
        Create constraints for the pump power with binary state
        """
        self.model.pump_power_capacity = pyo.Param(initialize=self.params.get('pump_power_capacity', 700), mutable=True)
        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                for t in self.time_steps:
                    # pump power constraint
                    self.model.add_component(
                        f"pump_power_constraint_{pump['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"pump_power_{pump['name']}")[t]
                                == self.model.component(
                                    f"pump_on_status_var_{pump['name']}"
                                )[t]
                                * self.model.pump_power_capacity
                            )
                        ),
                    )

    def create_pump_power_with_state_constraints(self):
        """
        Create constraints for the pump power with state.
        """
        if self.pump_data is None:
            return

        for pump in self.wn["links"]:
            if pump["link_type"] == "Pump":
                state_powers = self.pump_data[f"pump_{pump['name']}_power"].values
                state_powers = state_powers[~np.isnan(state_powers)]
                for t in self.time_steps:
                    self.model.add_component(
                        f"pump_power_with_state_{pump['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"pump_power_{pump['name']}")[t]
                                == sum(
                                    self.model.component(
                                        f"pump_state_power_value_{pump['name']}_{s}"
                                    )
                                    * self.model.component(
                                        f"pump_on_status_var_{pump['name']}_{s}"
                                    )[t]
                                    for s in range(len(state_powers))
                                )
                            )
                        ),
                    )

    def create_total_power_constraint(self):
        """
        Create a variable for the total power at each time step and a constraint for it.
        """
        # add all pump power variables to the total power constraint
        for t in self.time_steps:
            total_power_expression = {}
            for pump in self.wn["links"]:
                if pump["link_type"] == "Pump":
                    total_power_expression[pump["name"]] = self.model.component(
                        f"pump_power_{pump['name']}"
                    )[t]
            self.model.add_component(
                f"total_power_constraint_{t}",
                pyo.Constraint(
                    expr=(
                        self.model.component("total_power")[t]
                        == sum(total_power_expression.values())
                    )
                ),
            )

    def create_reservoir_constraints(self):
        """
        Create constraints for the reservoir.
        """
        if self.reservoir_data is None:
            return
        
        for reservoir in self.wn["nodes"]:
            if reservoir["node_type"] == "Reservoir":
                # Get reservoir data
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

                # Add volume constraints if specified
                if not np.isnan(min_volume):
                    self.model.add_component(
                        f"reservoir_volume_min_constraint_{reservoir['name']}",
                        pyo.Constraint(
                            expr=(
                                sum(self.model.component(f"reservoir_flow_{reservoir['name']}")[t] for t in self.time_steps)
                                >= min_volume
                            )
                        ),
                    )
                if not np.isnan(max_volume):
                    self.model.add_component(
                        f"reservoir_volume_max_constraint_{reservoir['name']}",
                        pyo.Constraint(
                            expr=(
                                sum(self.model.component(f"reservoir_flow_{reservoir['name']}")[t] for t in self.time_steps)
                                <= max_volume
                            )
                        ),
                    )

                # Add flow constraints
                for t in self.time_steps:
                    flow_in = 0
                    flow_out = 0
                    for pipe in self.wn["links"]:
                        if pipe["link_type"] in ["Pipe", "Pump"]:
                            if pipe["start_node_name"] == reservoir["name"]:
                                flow_out += self.model.component(f"{pipe['link_type'].lower()}_flow_{pipe['name']}")[t]
                            elif pipe["end_node_name"] == reservoir["name"]:
                                flow_in += self.model.component(f"{pipe['link_type'].lower()}_flow_{pipe['name']}")[t]

                    # Reservoir flow balance constraint
                    self.model.add_component(
                        f"reservoir_flow_equality_constraint_{reservoir['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"reservoir_flow_{reservoir['name']}")[t]
                                == flow_out - flow_in
                            )
                        ),
                    )

                    # Non-negative flow constraint
                    self.model.add_component(
                        f"reservoir_flow_positive_constraint_{reservoir['name']}_{t}",
                        pyo.Constraint(
                            expr=(
                                self.model.component(f"reservoir_flow_{reservoir['name']}")[t]
                                >= 0
                            )
                        ),
                    )

                    # Min/max flow constraints if specified
                    if not np.isnan(min_flow):
                        self.model.add_component(
                            f"reservoir_min_flow_constraint_{reservoir['name']}_{t}",
                            pyo.Constraint(
                                expr=(
                                    self.model.component(f"reservoir_flow_{reservoir['name']}")[t]
                                    >= min_flow
                                )
                            ),
                        )
                    if not np.isnan(max_flow):
                        self.model.add_component(
                            f"reservoir_max_flow_constraint_{reservoir['name']}_{t}",
                            pyo.Constraint(
                                expr=(
                                    self.model.component(f"reservoir_flow_{reservoir['name']}")[t]
                                    <= max_flow
                                )
                            ),
                        )

    def create_objective(self):
        """
        Create an objective function for the model.
        """
        self.charge_dict = costs.get_charge_dict(
            self.start_dt, self.end_dt, self.rate_df, resolution="1h"
        )
        consumption_data_dict = {"electric": self.model.total_power}
        self.model.electricity_cost, self.model = costs.calculate_cost(
            self.charge_dict,
            consumption_data_dict,
            resolution="1h",
            prev_demand_dict=None,
            prev_consumption_dict=None,
            consumption_estimate=0,
            desired_utility="electric",
            desired_charge_type=None,
            model=self.model,
        )
        self.model.add_component(
            "objective",
            pyo.Objective(expr=self.model.electricity_cost, sense=pyo.minimize),
        )

    def solve(self):
        """
        Solve the model.
        """
        # Use parameters from JSON if available
        verbose = self.params.get('verbose', True)
        time_limit = self.params.get('time_limit', 60)
        
        solver = SolverFactory("gurobi")
        solver.options["MIPGap"] = self.params.get('mip_gap', 0.01)
        solver.options["TimeLimit"] = time_limit
        solver.options["OptimalityTol"] = self.params.get('optimality_tol', 1e-6)
        results = solver.solve(
            self.model,
            tee=verbose,
        )
        self.results = results
        return results

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
                results[f'pipe_flow_{pipe["name"]}'] = [
                    value(self.model.component(f"pipe_flow_{pipe['name']}")[t])
                    for t in self.time_steps
                ]
            elif pipe["link_type"] == "Pump":
                results[f'pump_flow_{pipe["name"]}'] = [
                    value(self.model.component(f"pump_flow_{pipe['name']}")[t])
                    for t in self.time_steps
                ]
                results[f'pump_power_{pipe["name"]}'] = [
                    value(self.model.component(f"pump_power_{pipe['name']}")[t])
                    for t in self.time_steps
                ]
        for tank in self.wn["nodes"]:
            if tank["node_type"] == "Tank":
                results[f'tank_level_{tank["name"]}'] = [
                    value(self.model.component(f"tank_level_{tank['name']}")[t])
                    for t in self.time_steps
                ]
                results[f'tank_volume_{tank["name"]}'] = [
                    value(
                        self.model.component(f"tank_level_{tank['name']}")[t]
                        * self.model.component(f"tank_area_{tank['name']}")
                    )
                    for t in self.time_steps
                ]
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
                results[f'demand_{tank["name"]}'] = [
                    value(self.model.component(f"demand_pattern_{tank['name']}")[t])
                    for t in self.time_steps
                ]
            elif tank["node_type"] == "Reservoir" and self.reservoir_data is not None:
                results[f'reservoir_flow_{tank["name"]}'] = [
                    value(self.model.component(f"reservoir_flow_{tank['name']}")[t])
                    for t in self.time_steps
                ]

        results["total_power"] = [
            value(self.model.component("total_power")[t]) for t in self.time_steps
        ]
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

    def plot_results(self, save_to_file: bool = False):
        """
        Plot the results.
        """
        # Use parameter from JSON if available
        save_to_file = self.params.get('save_plot_to_file', save_to_file)
        
        packaged_data = self.package_data()
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

    def print_model_info(self, save_to_file: bool = False):
        """
        Print model information including variables and constraints in a structured way and save it to a text file.
        Args:
            save_to_file (bool): If True, saves the output to a text file named 'model_info.txt'
        """
        # Create output string
        output = []
        output.append("\n" + "=" * 50)
        output.append("MODEL INFORMATION")
        output.append("=" * 50)

        # Print total number of variables and constraints
        output.append("\nMODEL STATISTICS:")
        output.append("-" * 30)
        n_vars = len(self.model.component_map(pyo.Var))
        n_cons = len(self.model.component_map(pyo.Constraint))
        output.append(f"Total Variables: {n_vars}")
        output.append(f"Total Constraints: {n_cons}")

        # Print Variables
        output.append("\nVARIABLES:")
        output.append("-" * 30)
        for var_name, var in self.model.component_map(pyo.Var).items():
            output.append(f"\n{var_name}:")
            output.append(f"  Type: {type(var).__name__}")
            if hasattr(var, "index_set"):
                output.append(f"  Index Set: {var.index_set()}")
                # For indexed variables, get domain from first index
                if hasattr(var, "__getitem__"):
                    first_idx = next(iter(var.index_set()))
                    output.append(f"  Domain: {var[first_idx].domain}")
                else:
                    output.append(f"  Domain: {var.domain}")

        # print Parameters
        output.append("\nPARAMETERS:")
        output.append("-" * 30)
        for param_name, param in self.model.component_map(pyo.Param).items():
            output.append(f"\n{param_name}:")
            output.append(f"  Type: {type(param).__name__}")
            if hasattr(param, "index_set"):
                output.append(f"  Index Set: {param.index_set()}")
                # For indexed parameters, print value for first index
                if hasattr(param, "__getitem__"):
                    first_idx = next(iter(param.index_set()))
                    output.append(f"  Value (first index): {param[first_idx]}")
                else:
                    output.append(f"  Value: {param.value}")
            else:
                output.append(f"  Value: {param.value}")

        # Print Constraints
        output.append("\nCONSTRAINTS:")
        output.append("-" * 30)
        for con_name, con in self.model.component_map(pyo.Constraint).items():
            output.append(f"\n{con_name}:")
            output.append(f"  Type: {type(con).__name__}")
            if hasattr(con, "index_set"):
                output.append(f"  Index Set: {con.index_set()}")
                # For indexed constraints, print expression for first index
                if hasattr(con, "__getitem__"):
                    first_idx = next(iter(con.index_set()))
                    output.append("  Expression (first index):")
                    output.append(f"    {con[first_idx].expr}")
                else:
                    output.append("  Expression:")
                    output.append(f"    {con.expr}")
            else:
                output.append("  Expression:")
                output.append(f"    {con.expr}")

        # print the objective function
        output.append("\nOBJECTIVE FUNCTION:")
        output.append("-" * 30)
        output.append(f"  Expression: {self.model.objective.expr}")

        output.append("\n" + "=" * 50)

        # Save to file if requested
        if save_to_file:
            with open(
                f'data/local/model_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                "w",
            ) as f:
                f.write("\n".join(output))
        else:
            print("\n".join(output))


if __name__ == "__main__":
    # Example usage with parameters from JSON
    params_path = "data/simple_pump_tank_network_opt_params.json"
    wdn = DynamicWaterNetwork(params_path=params_path)
    wdn.solve()
    wdn.print_model_info(save_to_file=True)
    packaged_data = wdn.package_data()
    wdn.plot_results()
