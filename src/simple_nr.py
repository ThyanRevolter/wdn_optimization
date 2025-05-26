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

np.set_printoptions(suppress=True)

def solve_quadratic_coefficients(points):
    """
    Solve for coefficients (a, b, c) of quadratic equation y = ax² + bx + c given three points.
    
    Args:
        points (list): List of three points [(x1,y1), (x2,y2), (x3,y3)]
        
    Returns:
        tuple: Coefficients (a, b, c) of the quadratic equation
        
    Example:
        >>> points = [(0, 100), (100, 80), (200, 50)]
        >>> a, b, c = solve_quadratic_coefficients(points)
        >>> print(f"y = {a}x² + {b}x + {c}")
    """
    if len(points) != 3:
        raise ValueError("Exactly three points are required")
    
    # Extract x and y coordinates
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # Create the coefficient matrix A for the system Ax = b
    A = np.column_stack([x**2, x, np.ones(3)])
    
    # Solve the system using numpy's linear algebra solver
    a, b, c = np.linalg.solve(A, y)
    
    return np.round(a, 3), np.round(b, 3), np.round(c, 3)

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
    def __init__(self, inp_file_path, units=Units.IMPERIAL, round_to=3):
        """
        Initialize the WaterNetwork object.
        Args:
            inp_file_path (str): Path to the INP file
            units (Units): System of units (Imperial or Metric)
            round_to (int): Number of decimal places to round to
        """
        self.inp_file_path = inp_file_path
        self.wn = self.read_inp_file(inp_file_path, as_dict=True)
        self.units = units
        self.round_to = round_to
        self.convert_link_units(Units.METRIC, self.units)
        self.convert_node_units(Units.METRIC, self.units)
        self.convert_curve_units(Units.METRIC, self.units)
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
        if self.units == Units.IMPERIAL:
            return 4.73 * L / (C ** 1.85 * (D/12) ** 4.87)
        elif self.units == Units.METRIC:
            return 10.67 * L / (C ** 1.85 * D ** 4.87)
    
    def convert_link_units(self, from_units, to_units):
        """
        Convert the units of the network.
        Args:
            from_units (Units): From unit
            to_units (Units): To unit
        """
        if from_units == Units.METRIC and to_units == Units.IMPERIAL:
            for link in self.wn["links"]:
                if link["link_type"] == "Pipe":
                    link["length"] = np.round(link["length"] * 3.28084, self.round_to) # convert m to ft
                    link["diameter"] = np.round(link["diameter"] * 39.3701, self.round_to) # convert m to in
                else:
                    print(f"This is a pump: {link}")

        elif from_units == Units.IMPERIAL and to_units == Units.METRIC:
            for link in self.wn["links"]:
                if link["link_type"] == "Pipe":
                    link["length"] = np.round(link["length"] / 3.28084, self.round_to) # convert ft to m
                    link["diameter"] = np.round(link["diameter"] / 39.3701, self.round_to) # convert in to m

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
                    node["elevation"] = np.round(node["elevation"] * 3.28084, self.round_to) # convert m to ft
                    node["base_demand"] = np.round(node["base_demand"] * 15850.32, self.round_to) # convert m^3/s to gal/min
                elif node["node_type"] == "Reservoir":
                    node["base_head"] = np.round(node["base_head"] * 3.28084, self.round_to) # convert m to ft

        elif from_units == Units.IMPERIAL and to_units == Units.METRIC:
            for node in self.wn["nodes"]:
                if node["node_type"] == "Junction":
                    node["elevation"] = np.round(node["elevation"] / 3.28084, self.round_to) # convert ft to m
                    node["base_demand"] = np.round(node["base_demand"] / 15850.32, self.round_to) # convert gal/min to m^3/s
                elif node["node_type"] == "Reservoir":
                    node["base_head"] = np.round(node["base_head"] / 3.28084, self.round_to) # convert ft to m
    
    def convert_curve_units(self, from_units, to_units):
        """
        Convert the units of the curves.
        Args:
            from_units (Units): From unit
            to_units (Units): To unit
        """
        if from_units == Units.IMPERIAL and to_units == Units.METRIC:
            for curve in self.wn["curves"]:
                converted_points = []
                for point in curve["points"]:
                    # (flow, head)
                    converted_points.append(
                        (np.round(point[0] / 15850.32, self.round_to),  # convert gal/min to m^3/s
                         np.round(point[1] / 3.28084, self.round_to))    # convert ft to m
                    )
                curve["points"] = converted_points
                curve["quadratic_coefficients"] = dict(zip(["a", "b", "c"], solve_quadratic_coefficients(converted_points)))
        elif from_units == Units.METRIC and to_units == Units.IMPERIAL:
            for curve in self.wn["curves"]:
                converted_points = []
                for point in curve["points"]:
                    # (flow, head)
                    converted_points.append(
                        (np.round(point[0] * 15850.32, self.round_to),  # convert m^3/s to gal/min
                         np.round(point[1] * 3.28084, self.round_to))    # convert m to ft
                    )
                curve["points"] = converted_points
                curve["quadratic_coefficients"] = dict(zip(["a", "b", "c"], solve_quadratic_coefficients(converted_points)))
    def set_link_k_values(self):
        """
        Set the k values for the links.
        """
        for link in self.wn["links"]:
            if link["link_type"] == "Pipe":
                link["k"] = np.round(self.calculate_k(link["length"], link["diameter"], link["roughness"]), self.round_to)

    def get_link_k_values(self):
        """
        Get the k values for the links.
        """
        k_values = {}
        for link in self.wn["links"]:
            if link["link_type"] == "Pipe":
                k_values[link["name"]] = link["k"]
            elif link["link_type"] == "Pump":
                k_values[link["name"]] = 0
        return k_values
    
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
        self.initial_flow = initial_flow.astype(float)

    def set_initial_head(self, initial_head:np.ndarray):
        """
        Set the initial head for the nodes.
        """
        if initial_head.shape[0] != len([node for node in self.wn["nodes"] if node["node_type"] == "Junction"]):
            raise ValueError("The initial head array must have the same number of elements as the number of junctions in the network.")
        self.initial_head = initial_head.astype(float)
    
    def find_pump_curve(self, pump_curve_name):
        """
        Find the pump curve for the given pump curve name.
        """
        for curve in self.wn["curves"]:
            if curve["name"] == pump_curve_name:
                return curve
        return None
    
    def get_pump_head_difference(self):
        """
        Get the head difference for the pumps.
        """
        difference_vector = np.zeros(self.n_links)
        pump_indices = [i for i, link in enumerate(self.wn["links"]) if link["link_type"] == "Pump"]
        for i in pump_indices:
            pump_curve = self.find_pump_curve(self.wn["links"][i]["pump_curve_name"])
            a = pump_curve["quadratic_coefficients"]["a"]
            b = pump_curve["quadratic_coefficients"]["b"]
            flow = self.initial_flow[i]
            delta_head = 2*a*flow + b
            difference_vector[i] = delta_head
        return np.diag(difference_vector)
    
    def head_loss_difference_matrix(self):
        """
        Calculate the head loss matrix for the network.

        The head loss matrix is a diagonal matrix where each element (i,i) represents
        the head loss for link i, calculated as:
        \Delta h_f = 1.852 * K * abs(Q)^0.852 * sign(Q)

        where:
        - K = loss coefficient
        - Q = flow rate

        Returns:
            numpy.ndarray: Diagonal matrix of head losses
        """
        K = np.array(list(self.get_link_k_values().values()))
        self.head_loss = np.multiply(
            np.sign(self.initial_flow),
            np.round(1.852 * np.multiply(K, np.power(np.abs(self.initial_flow), 0.852)), self.round_to)
        )
        self.head_loss = np.diag(self.head_loss)
        return self.head_loss - self.get_pump_head_difference()

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
    
    def get_reservoir_link_head_vector(self):
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
    
    def get_pump_head_vector(self):
        """
        Get the pump head for the network from the pump curve.
        """
        pump_head_vector = np.zeros(self.n_links)
        for i, link in enumerate(self.wn["links"]):
            if link["link_type"] == "Pump":
                pump_curve = self.find_pump_curve(link["pump_curve_name"])
                a = pump_curve["quadratic_coefficients"]["a"]
                b = pump_curve["quadratic_coefficients"]["b"]
                c = pump_curve["quadratic_coefficients"]["c"]
                flow = self.initial_flow[i]
                pump_head_vector[i] = a*flow**2 + b*flow + c
        return pump_head_vector
    
    def get_link_head_loss_vector(self):
        """
        Get the link head loss vector for the network.
        """
        K = np.array(list(self.get_link_k_values().values()))
        return np.multiply(K, np.power(self.initial_flow, 1.852))
    
    def get_link_head_difference_vector(self):
        """
        Get the link head difference vector for the network.
        """
        return self.flow_adjacency_matrix() @ self.initial_head

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
        return -(
            self.get_link_head_loss_vector()
            + self.get_link_head_difference_vector()
            - self.get_reservoir_link_head_vector()
            - self.get_pump_head_vector())
    
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
        return np.round(np.linalg.solve(self.get_lhs_matrix(), self.get_rhs_vector()), self.round_to)
    
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
    parser.add_argument("--inp_file_path", type=str, default="epanet_networks/chapter_5_4_example.inp")
    parser.add_argument("--units", type=Units, default=Units.IMPERIAL)
    args = parser.parse_args()

    wn = WaterNetwork(args.inp_file_path, units=args.units, round_to=5)
    initial_flow = np.array([20, 9, 11, 6, 5.5, 3.5, 0.5, 0.5, 1, 1, 8])
    initial_head = np.array([198, 193, 195, 175, 188, 190, 184], dtype=float)

    # initial_head = np.array([40, 35, 30])
    # initial_flow = np.array([4.5, 2, 2, 0.5])

    max_iter = 100
    tol = 1e-6
    final_flow, final_head = wn.run_newton_raphson(initial_flow, initial_head, max_iter, tol)
    print(final_flow)
    print(final_head)
