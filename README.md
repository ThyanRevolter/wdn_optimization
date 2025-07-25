# Water Distribution Network Optimization and Control

[![validate](https://github.com/ThyanRevolter/epanet_example/actions/workflows/python-tests.yml/badge.svg)](https://github.com/ThyanRevolter/epanet_example/actions/workflows/python-tests.yml)
[![coverage](https://github.com/ThyanRevolter/epanet_example/blob/badges/.github/badges/coverage.svg)](https://github.com/ThyanRevolter/epanet_example)

This repository contains tools for water distribution network analysis and optimization, including both optimization capabilities and the Newton-Raphson method for solving water distribution networks using the Hazen-Williams formula.

## Features

- Advanced optimization of water distribution networks
- Implementation of the Newton-Raphson method for water network analysis
- Model Predictive Control for water distribution networks with demand variability  

## Requirements

- Python 3.10 or higher
- Poetry
- Gurobi Optimizer (for advanced optimization capabilities)
  - Download from [Gurobi's website](https://www.gurobi.com/downloads/)
  - Free academic license available for academic users

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ThyanRevolter/wdn_optimization.git
```

2. cd to repository location
```bash
cd wdn_optimization
```

3. Install Gurobi Optimizer: (Optional if you haven't installed it already)
   - Download and install Gurobi from [Gurobi's website](https://www.gurobi.com/downloads/)
   - Get a free academic license if you're an academic user
   - Add Gurobi to your system PATH

4. Activate the Poetry shell:
```bash
eval $(poetry env activate)
```

5. Install dependencies using Poetry (recommended):
```bash
poetry install
```


## Water Distribution Network Optimization

The project includes advanced optimization capabilities for water distribution networks using both Pyomo and CVXPY solvers. These optimization tools help in:

- Minimizing operational costs
- Optimizing pump operations
- Managing tank levels
- Handling time-varying demands
- Considering electricity tariffs
- Supporting both binary and continuous pump operations

### Optimization Implementations

The optimization is implemented using two different optimization frameworks:

1. **CVXPY Implementation** (`wdn_cvxpy.py`)
   - Convex optimization capabilities
   - Efficient linear programming
   - Simplified pump modeling
   - [CVXPY Documentation](https://www.cvxpy.org/)

2. **Pyomo Implementation** (`wdn_pyomo.py`)
   - Mixed-integer linear programming (MILP) support
   - Advanced pump control capabilities
   - [Pyomo Documentation](https://pyomo.readthedocs.io/)

### Optimization Features

1. **Dynamic Network Optimization**
   - Time-step based optimization
   - Support for multiple time periods
   - Demand pattern integration
   - Tank level management
   - Reservoir constraints

2. **Pump Optimization**
   - Binary pump control (on/off)
   - Continuous pump control
   - Multiple pump states
   - Power consumption optimization
   - Minimum runtime constraints

3. **Cost Optimization**
   - Electricity cost minimization
   - Time-of-use tariff support
   - Operational cost tracking
   - Power consumption monitoring

### Usage Example

#### CVXPY Implementation
```python
from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX
from utils import utils as ut

# Initialize the optimization model
params = ut.load_json_file("data/soporon_network_opt_params.json")
wn_opt = DynamicWaterNetworkCVX(params=params)

# Print constraints and objective function for debugging
print("Constraints:")
for constraint_name, constraint in wn_opt.constraints.items():
    print(constraint_name)
    print(constraint)
    print("-"*100)
print("-"*100)
print("Objective function:")
print(wn_opt.electricity_cost_objective)
print("-"*100)

# Solve the optimization problem
wn_opt.solve()

# Print optimization summary
wn_opt.print_optimization_result()

# Get optimization results
packaged_data = wn_opt.package_data(save_to_csv=True)

# Visualize results
ut.plot_results(wn_opt.wn, packaged_data)
```

The CVXPY implementation provides:
- Support for various solvers (ECOS, GUROBI, MOSEK, CBC, SCIPY)
- Detailed pump operation analysis

#### Pyomo Implementation
```python
from wdn_optimization.wdn_pyomo import DynamicWaterNetwork

# Initialize the optimization model
wn_opt = DynamicWaterNetwork("path/to/optimization_params.json")

# Solve the optimization problem
wn_opt.solve()

# Get and visualize results
wn_opt.plot_results(save_to_file=True)
wn_opt.package_data(save_to_csv=True)
```

### Optimization Parameters

The optimization requires a JSON configuration file with the following parameters:

```json
{
    "start_date": "2025-01-01 00:00:00",
    "end_date": "2025-01-02 00:00:00",
    "time_step": 3600,
    "network_path": "data/epanet_networks/simple_pump_tank.inp",
    "pump_data_path": null,
    "reservoir_data_path": null,
    "final_tank_level_deviation": 0,
    "binary_pump": false,
    "pump_flow_capacity": 700,
    "pump_power_capacity": 700,
    "verbose": true,
    "time_limit": 600,
    "save_to_csv": true,
    "save_plot_to_file": true
}
```

### Output and Visualization

The optimization results include:
- Pump operation schedules
- Tank level variations
- Power consumption profiles
- Cost breakdowns

Results can be exported to CSV files and visualized using built-in plotting functions.

## Model Predictive Control (MPC)

The project implements Model Predictive Control for water distribution networks, enabling real-time optimization with demand uncertainty and system disturbances.

### MPC Features

- **Rolling Horizon Control**: Continuously updates optimization based on current system state
- **Demand Uncertainty Handling**: Incorporates demand variations and forecasting errors
- **Tank Level Management**: Maintains optimal tank levels across prediction horizons
- **Cost Comparison**: Compares MPC performance against prescient (perfect foresight) control
- **Real-time Adaptation**: Updates control decisions based on actual system measurements

### Running MPC

#### Basic MPC Execution
```python
from mpc.mpc_wrapper import MPCWrapper
from datetime import datetime
from utils import utils as ut

# Load optimization parameters
params = ut.load_json_file("data/soporon_network_opt_params.json")

# Configure MPC parameters
mpc_params = {
    "optimization_params": params,
    "simulation_start_date": datetime.strptime("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    "simulation_end_date": datetime.strptime("2025-01-02 00:00:00", "%Y-%m-%d %H:%M:%S"),
    "simulation_time_step": 3600,  # 1 hour
    "model_update_interval": 6*3600,  # 6 hours
    "model_prediction_horizon": 24  # 24 hours
}

# Initialize and run MPC
mpc_wrapper = MPCWrapper(mpc_params)
results = mpc_wrapper.run_mpc()

# Get actual operations and compare with prescient
actual_operations = mpc_wrapper.get_actual_operations(results)
prescient_operations = mpc_wrapper.get_prescient_operations()

# Calculate costs
actual_cost = ut.get_electricity_cost(actual_operations, rate_df)
prescient_cost = ut.get_electricity_cost(prescient_operations, rate_df)
```

#### MPC Visualization
```python
# Run the interactive MPC visualization
poetry run python mpc_visualization.py
```

The visualization provides:
- Interactive plots of pump flows, tank levels, and demands
- Comparison between MPC and prescient control
- Time series analysis across prediction horizons
- Cost performance metrics

### MPC Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `simulation_time_step` | Time resolution for simulation | 3600 (1 hour) |
| `model_update_interval` | Frequency of MPC updates | 6*3600 (6 hours) |
| `model_prediction_horizon` | Length of prediction window | 24 (24 hours) |

## Water Distribution Network Analysis

This project implements the Hazen-Williams formula for analyzing water distribution networks using the Newton-Raphson method. It provides tools for calculating flow rates, head losses, and pressure distributions in water networks.

### Basic Example

```python
from wdn_optimization.simple_nr import WaterNetwork, Units
import numpy as np

# Create a water network object
wn = WaterNetwork("path/to/your/network.inp", units=Units.IMPERIAL)

# Set initial conditions
initial_flow = np.array([4.5, 2, 2, 0.5])  # Initial flow rates for each link
initial_head = np.array([40, 35, 30])      # Initial heads for each junction

# Run the Newton-Raphson method
flow, head = wn.run_newton_raphson(
    initial_flow=initial_flow,
    initial_head=initial_head,
    max_iter=100,
    tol=1e-6
)

print(f"Final flow rates: {flow}")
print(f"Final heads: {head}")
```

### Command Line Interface

You can also run the analysis from the command line:

```bash
poetry run python -m wdn_optimization.simple_nr --inp_file_path "path/to/your/network.inp" --units Imperial
```

## Project Structure

```
project-root/
├── src/
│   └── epanet_tutorial/
│       ├── __init__.py
│       ├── simple_nr.py
│       └── wdn_pyomo.py
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Version Management

This project uses [bumpver](https://github.com/mbarkhau/bumpver) for consistent versioning. To bump the version, use:

```bash
poetry run bumpver update --patch   # or --minor, --major
```

This will update the version in all relevant files and create a git commit and tag.

## Input File Format

The program expects an EPANET INP file format. The INP file should contain:

- [JUNCTIONS] section with node elevations and demands
- [PIPES] section with pipe properties (length, diameter, roughness)
- [RESERVOIRS] section with reservoir heads
- [COORDINATES] section for network layout

Example INP file structure:
```
[JUNCTIONS]
;ID               Elevation    Demand
J1                100          0
J2                95           100
J3                90           150

[PIPES]
;ID               Node1        Node2        Length      Diameter    Roughness
P1                R1           J1           1000        12          100
P2                J1           J2           800         10          100
P3                J2           J3           600         8           100

[RESERVOIRS]
;ID               Head
R1                150
```

## Units

The program supports both Imperial and Metric units:

### Imperial Units
- Length: feet (ft)
- Diameter: inches (in)
- Flow: gallons per minute (gpm)
- Head: feet (ft)

### Metric Units
- Length: meters (m)
- Diameter: meters (m)
- Flow: cubic meters per second (m³/s)
- Head: meters (m)

## Mathematical Background

The program uses the Hazen-Williams equation for calculating head losses:

h_f = 4.73 * L * Q^1.852 / (C^1.852 * D^4.87)

where:
- h_f = head loss due to friction
- L = length of pipe
- Q = flow rate
- C = Hazen-Williams roughness coefficient
- D = diameter of pipe

The Newton-Raphson method is used to solve the system of nonlinear equations that describe the network's flow and pressure distribution.

## Newton-Raphson Algorithm

The Newton-Raphson method is used to solve the system of nonlinear equations that govern the water distribution network. The algorithm is implemented as follows:

### 1. System of Equations

The network is described by two sets of equations:

#### Energy Equations
For each loop in the network:
```
\sum(h_f) + \sum(\Delta H) - H_{reservoir} = 0
```
where:
- h_f = head loss in each pipe (from Hazen-Williams equation)
- \Delta H = head difference between nodes
- H_{reservoir} = reservoir head (from RESERVOIRS section)

#### Continuity Equations
For each junction:
```
\sum(Q_{in}) - \sum(Q_{out}) - Q_{demand} = 0
```
where:
- Q_{in} = incoming flow rates
- Q_{out} = outgoing flow rates
- Q_{demand} = junction demand

### 2. Algorithm Steps

1. **Initialization**
   - Set initial flow rates (Q₀) for all pipes
   - Set initial heads (H₀) for all junctions
   - Define convergence tolerance (tol) and maximum iterations (max_iter)

2. **Iteration Process**
   For each iteration k:
   
   a. Calculate the Jacobian matrix J:
   ```
   J = [∂E_{energy}/∂Q    ∂E_{energy}/∂H]
       [∂E_{continuity}/∂Q ∂E_{continuity}/∂H]
   ```
   
   b. Calculate the error vector E:
   ```
   E = [E_{energy}]
       [E_{continuity}]
   ```
   
   c. Solve the linear system for the update vector \Delta X:
   ```
   J * \Delta X = -E
   ```
   
   d. Update the variables:
   ```
   Q_{k+1} = Q_k + \Delta Q
   H_{k+1} = H_k + \Delta H
   ```

3. **Convergence Check**
   - If ||\Delta X|| < tol, the solution has converged
   - If k > max_iter, stop and report non-convergence
   - Otherwise, continue to next iteration

### 3. Implementation Details

The algorithm is implemented in the `run_newton_raphson` method with the following components:

- **Head Loss Matrix**: Diagonal matrix containing head losses for each pipe
- **Flow Difference Matrix**: Adjacency matrix representing network topology
- **Error Vectors**: 
  - Nodal balance error (energy equations)
  - Link flow error (continuity equations)
- **Update Vector**: Combined flow and head updates

### 4. Convergence Properties

- The algorithm typically converges quadratically near the solution
- Convergence depends on:
  - Quality of initial guess
  - Network topology
  - Pipe properties
  - Demand distribution

### 5. Example

For a simple network with 3 pipes and 2 junctions:

```python
from wdn_optimization.simple_nr import WaterNetwork, Units
import numpy as np

# Create a water network object
wn = WaterNetwork("path/to/your/network.inp", units=Units.IMPERIAL)

# Initial conditions
initial_flow = np.array([4.5, 2, 2])  # Flow rates for 3 pipes
initial_head = np.array([40, 35])     # Heads for 2 junctions

# Run Newton-Raphson
flow, head = wn.run_newton_raphson(
    initial_flow=initial_flow,
    initial_head=initial_head,
    max_iter=100,
    tol=1e-6
)
```

The algorithm will iteratively solve for the flow rates and heads that satisfy both the energy and continuity equations.


