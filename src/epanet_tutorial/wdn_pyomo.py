""""
This module implements a water distribution network model using Pyomo without any loss or friction terms.
"""

import pyomo.environ as pyo
from .simple_nr import WaterNetwork, Units


class DynamicWaterNetwork():
    """
    A class to represent a simple dynamic water network.
    """
    def __init__(self, inp_file_path:str):
        self.n_time_steps = 24
        self.time_steps = range(self.n_time_steps)
        self.wn = WaterNetwork(inp_file_path, units=Units.IMPERIAL, round_to=3)
        self.model = pyo.ConcreteModel()
        self.create_model_variables()

    def create_model_variables(self):
        """
        Create a Pyomo model variables for the water network.
        """
        # pipe flow variables for each pipe for each time step
        for pipe in self.wn.wn["links"]:
            if pipe["link_type"] == "Pipe":
                self.model.pipe_flow[pipe["name"]] = pyo.Var(self.time_steps, initialize=0)

        # pump flow variables for each pump for each time step
        for pump in self.wn.wn["links"]:
            if pump["link_type"] == "Pump":
                self.model.pump_flow[pump["name"]] = pyo.Var(self.time_steps, initialize=0)

        # tank level variables for each tank for each time step
        for tank in self.wn.wn["nodes"]:
            if tank["node_type"] == "Tank":
                self.model.tank_level[tank["name"]] = pyo.Var(self.time_steps, initialize=0)

        # demand variables for each demand node for each time step
        for demand_node in self.wn.wn["nodes"]:
            if demand_node["node_type"] == "Junction":
                self.model.demand[demand_node["name"]] = pyo.Var(self.time_steps, initialize=0)       
        

if __name__ == "__main__":
    wdn = DynamicWaterNetwork("epanet_networks/dynamic_simple_model.inp")
