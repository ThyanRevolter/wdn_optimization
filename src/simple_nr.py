""" This module implements the Hazen-Williams formula for water distribution network analysis and runs the Newton-Raphson method to solve the system of nonlinear equations.

The Hazen-Williams equation is an empirical formula that relates the flow of water in a pipe
to the physical properties of the pipe and the pressure drop caused by friction. The formula is:

h_f = 4.73 * L * Q^1.852 / (C^1.852 * D^4.87)

where:
- h_f = head loss due to friction (ft)
- L = length of pipe (ft)
- Q = flow rate (gpm)
- C = Hazen-Williams roughness coefficient
- D = diameter of pipe (in)
"""

import numpy as np
import wntr
from enum import Enum

class Units(Enum):
    """
    Enum for the units of the network.
    
    Attributes:
        IMPERIAL: Uses feet (ft) for length, gallons per minute (gpm) for flow
        METRIC: Uses meters (m) for length, cubic meters per second (m³/s) for flow
    """
    IMPERIAL = "Imperial"
    METRIC = "Metric"

class WaterNetwork:
    """
    Class for analyzing water distribution networks using the Hazen-Williams formula.

    This class implements simple the Newton-Raphson method for solving the system of nonlinear
    equations that describe the flow and pressure distribution in a water network.

    Attributes:
        inp_file_path (str): Path to the EPANET INP file
        wn (dict): Network dictionary containing nodes and links
        units (Units): System of units (Imperial or Metric)
        links (list): List of pipe links in the network
        nodes (list): List of nodes (junctions and reservoirs) in the network
        n_junctions (int): Number of junction nodes
        n_links (int): Number of pipe links
    """
    def __init__(self, inp_file_path, units=Units.IMPERIAL):
        """
        Initialize the WaterNetwork object.
        Args:
            inp_file_path (str): Path to the INP file
        """
        self.inp_file_path = inp_file_path
        self.wn = self.read_inp_file(inp_file_path, as_dict=True)
        self.units = units
        self.convert_link_units(Units.METRIC, self.units)
        self.convert_node_units(Units.METRIC, self.units)
        self.set_link_k_values()
        self.links = self.wn["links"]
        self.nodes = self.wn["nodes"]
        self.n_junctions = len([node for node in self.wn["nodes"] if node["node_type"] == "Junction"])
        self.n_links = len(self.wn["links"])

    def read_inp_file(self, file_path, as_dict=False):
        """
        Read the INP file and return the network model.
        Args:
            file_path (str): Path to the INP file
        Returns:
            wntr.network.WaterNetworkModel: Network model
        """
        if as_dict:
            return wntr.network.to_dict(wntr.network.WaterNetworkModel(file_path))
        else:
            return wntr.network.WaterNetworkModel(file_path)
    
    def calculate_k(self, L, D, C):
        """
        Calculate the loss coefficient K for the Hazen-Williams formula.

        The K coefficient is calculated as:
        K = 4.73 * L / (C^1.85 * (D/12)^4.87)

        where:
        - L = length of pipe (ft)
        - D = diameter of pipe (in)
        - C = Hazen-Williams coefficient

        Args:
            L (float): Length of the pipe (ft)
            D (float): Diameter of the pipe (in)
            C (float): Hazen-Williams coefficient

        Returns:
            float: K factor used in head loss calculations
        """
        return 4.73 * L / (C ** 1.85 * (D/12) ** 4.87)
    
    def convert_link_units(self, from_units, to_units):
        """
        Convert the units of the network.
        Args:
            from_units (Units): From unit
            to_units (Units): To unit
        """
        if from_units == Units.METRIC and to_units == Units.IMPERIAL:
            for link in self.wn["links"]:
                link["length"] = np.round(link["length"] * 3.28084, 3) # convert m to ft
                link["diameter"] = np.round(link["diameter"] * 39.3701, 3) # convert m to in

        elif from_units == Units.IMPERIAL and to_units == Units.METRIC:
            for link in self.wn["links"]:
                link["length"] = np.round(link["length"] / 3.28084, 3) # convert ft to m
                link["diameter"] = np.round(link["diameter"] / 39.3701, 3) # convert in to m

    def convert_node_units(self, from_units, to_units):
        """
        Convert the units of the nodes.
        Args:
            from_units (Units): From unit
            to_units (Units): To unit
        """
        if from_units == Units.METRIC and to_units == Units.IMPERIAL:
            for node in self.wn["nodes"]:
                if node["node_type"] == "Junction":
                    node["elevation"] = np.round(node["elevation"] * 3.28084, 3) # convert m to ft
                    node["base_demand"] = np.round(node["base_demand"] * 15850.32, 3) # convert m^3/s to gal/min
                elif node["node_type"] == "Reservoir":
                    node["base_head"] = np.round(node["base_head"] * 3.28084, 3) # convert m to ft

        elif from_units == Units.IMPERIAL and to_units == Units.METRIC:
            for node in self.wn["nodes"]:
                if node["node_type"] == "Junction":
                    node["elevation"] = np.round(node["elevation"] / 3.28084, 3) # convert ft to m
                    node["base_demand"] = np.round(node["base_demand"] / 15850.32, 3) # convert gal/min to m^3/s
                elif node["node_type"] == "Reservoir":
                    node["base_head"] = np.round(node["base_head"] / 3.28084, 3) # convert ft to m
    
    def set_link_k_values(self):
        """
        Set the k values for the links.
        """
        for link in self.wn["links"]:
            link["k"] = np.round(self.calculate_k(link["length"], link["diameter"], link["roughness"]), 3)

    def get_link_k_values(self):
        """
        Get the k values for the links.
        """
        return {link["name"]: link["k"] for link in self.wn["links"]}
    
    def get_node_base_demand(self):
        """
        Get the base demand for the nodes.
        """
        return {node["name"]: node["base_demand"] for node in self.wn["nodes"] if node["node_type"] == "Junction"}
    
    def get_node_base_head(self):
        """
        Get the base head for the nodes.
        """
        return {node["name"]: node["base_head"] for node in self.wn["nodes"] if node["node_type"] == "Reservoir"}
    
    def get_node_elevation(self):
        """
        Get the elevation for the nodes.
        """
        return {node["name"]: node["elevation"] for node in self.wn["nodes"] if node["node_type"] == "Junction"}
    
    def set_initial_flow(self, initial_flow:np.ndarray):
        """
        Set the initial flow for the links.
        Args:
            initial_flow (np.ndarray): Initial flow for the links
        """
        if initial_flow.shape[0] != len(self.wn["links"]):
            raise ValueError("The initial flow array must have the same number of elements as the number of links in the network.")
        self.initial_flow = initial_flow

    def set_initial_head(self, initial_head:np.ndarray):
        """
        Set the initial head for the nodes.
        """
        if initial_head.shape[0] != len([node for node in self.wn["nodes"] if node["node_type"] == "Junction"]):
            raise ValueError("The initial head array must have the same number of elements as the number of junctions in the network.")
        self.initial_head = initial_head
    
    def head_loss_difference_matrix(self):
        """
        Calculate the head loss matrix for the network.

        The head loss matrix is a diagonal matrix where each element (i,i) represents
        the head loss for link i, calculated as:
        \Delta h_f = 1.852 * K * Q^0.852

        where:
        - K = loss coefficient
        - Q = flow rate

        Returns:
            numpy.ndarray: Diagonal matrix of head losses
        """
        K = np.array(list(self.get_link_k_values().values()))
        self.head_loss = 1.852 * np.multiply(K, np.power(self.initial_flow, 0.852))
        self.head_loss = np.diag(self.head_loss)
        return self.head_loss

    def flow_adjacency_matrix(self):
        """
        Calculate the adjacency matrix of the network.

        The adjacency matrix A has dimensions (n_junctions × n_links) where:
        - A[i,j] = -1 if link j enters junction i
        - A[i,j] = 1 if link j leaves junction i
        - A[i,j] = 0 otherwise

        This matrix represents the network topology and is used in the continuity equations.

        Returns:
            numpy.ndarray: Transpose of the adjacency matrix
        """
        junction_names = [node["name"] for node in self.wn["nodes"] if node["node_type"] == "Junction"]
        junction_indices = {name: index for index, name in enumerate(junction_names)}
        adjacency_matrix = np.zeros((self.n_junctions, self.n_links))
        for i, link in enumerate(self.wn["links"]):
            start_node = link["start_node_name"]
            end_node = link["end_node_name"]
            if start_node in junction_indices:
                adjacency_matrix[junction_indices[start_node], i] = -1
            if end_node in junction_indices:
                adjacency_matrix[junction_indices[end_node], i] = 1
        return adjacency_matrix.T
    
    def get_reservoir_link_vector(self):
        """
        Get the reservoir link vector for the network.
        """
        reservoir_link_vector = np.zeros(self.n_links)
        for i, link in enumerate(self.wn["links"]):
            if link["start_node_name"] in self.get_node_base_head():
                reservoir_link_vector[i] = self.get_node_base_head()[link["start_node_name"]]
        return reservoir_link_vector

    def get_demand_node_vector(self):
        """
        Get the demand node vector for the network.
        """
        demand_node_vector = np.zeros(self.n_junctions)
        for i, node in enumerate([node for node in self.wn["nodes"] if node["node_type"] == "Junction"]):
            demand_node_vector[i] = node["base_demand"]
        return demand_node_vector
    
    def get_nodal_balance_error(self):
        """
        Calculate the nodal balance error for the network.

        The nodal balance error represents the violation of the energy conservation principle
        at each node. For each node, the sum of head losses around any loop must equal zero:

        ∑(h_f) + ∑(ΔH) - H_reservoir = 0

        where:
        - h_f = head loss in each pipe
        - ΔH = head difference between nodes
        - H_reservoir = reservoir head

        Returns:
            numpy.ndarray: Vector of nodal balance errors
        """
        K = np.array(list(self.get_link_k_values().values()))
        link_head_loss_vector = np.multiply(K, np.power(self.initial_flow, 1.852))
        head_difference = self.flow_adjacency_matrix() @ self.initial_head
        return -(link_head_loss_vector + head_difference - self.get_reservoir_link_vector())
    
    def get_link_flow_error(self):
        """
        Calculate the link flow error for the network.

        The link flow error represents the violation of the continuity principle at each junction.
        For each junction, the sum of incoming flows must equal the sum of outgoing flows plus demand:

        ∑(Q_in) - ∑(Q_out) - Q_demand = 0

        where:
        - Q_in = incoming flow rates
        - Q_out = outgoing flow rates
        - Q_demand = junction demand

        Returns:
            numpy.ndarray: Vector of link flow errors
        """
        return -(self.flow_adjacency_matrix().T @ self.initial_flow - self.get_demand_node_vector())
    
    def get_lhs_matrix(self):
        """
        Calculate the lhs matrix for the network.
        """
        lhs_matrix = np.hstack(
            [
                np.vstack(
                    [
                        self.head_loss_difference_matrix(),
                        self.flow_adjacency_matrix().T
                    ]
                ),
                np.vstack(
                    [
                    self.flow_adjacency_matrix(),
                    np.zeros((self.n_junctions, self.n_junctions))
                    ]
                )
            ]
        )
        return lhs_matrix
    
    def get_rhs_vector(self):
        """
        Calculate the rhs vector for the network.
        """
        return np.hstack((self.get_nodal_balance_error(), self.get_link_flow_error()))
    
    def get_update_vector(self):
        """
        Calculate the update vector for the network.
        """
        return np.round(np.linalg.solve(self.get_lhs_matrix(), self.get_rhs_vector()), 3)
    
    def update_flow_and_head(self, update_vector:np.ndarray):
        """
        Update the flow and head for the network.
        """
        self.initial_flow += update_vector[:self.n_links]
        self.initial_head += update_vector[self.n_links:]

    def run_newton_raphson(self, initial_flow:np.ndarray, initial_head:np.ndarray, max_iter:int=100, tol:float=1e-6):
        """
        Run the Newton-Raphson method to solve the network equations.

        The Newton-Raphson method iteratively solves the system of nonlinear equations:
        1. Energy equations: ∑(h_f) + ∑(ΔH) - H_reservoir = 0
        2. Continuity equations: ∑(Q_in) - ∑(Q_out) - Q_demand = 0

        The method updates the flow and head values using:
        [ΔQ, ΔH] = -J⁻¹ * [E_energy, E_continuity]

        where:
        - J = Jacobian matrix
        - E_energy = energy balance errors
        - E_continuity = continuity balance errors

        Args:
            initial_flow (np.ndarray): Initial flow rates for all links
            initial_head (np.ndarray): Initial heads for all junctions
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance

        Returns:
            tuple: (final_flow, final_head) arrays
        """
        self.set_initial_flow(initial_flow)
        self.set_initial_head(initial_head)
        for _ in range(max_iter):
            update_vector = self.get_update_vector()
            self.update_flow_and_head(update_vector)
            if np.linalg.norm(update_vector) < tol:
                break
        return self.initial_flow, self.initial_head

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_file_path", type=str, default="epanet_networks/chapter_5_3_4_example.inp")
    parser.add_argument("--units", type=Units, default=Units.IMPERIAL)
    args = parser.parse_args()

    wn = WaterNetwork(args.inp_file_path, units=args.units)
    initial_flow = np.array([4.5, 2, 2, 0.5])
    initial_head = np.array([40, 35, 30], dtype=float)
    max_iter = 100
    tol = 1e-6
    flow, head = wn.run_newton_raphson(initial_flow, initial_head, max_iter, tol)
    print(f"Flow: {flow}")
    print(f"Head: {head}")
