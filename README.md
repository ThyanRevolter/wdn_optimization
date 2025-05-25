# Water Distribution Network Analysis

This project implements the Hazen-Williams formula for analyzing water distribution networks using the Newton-Raphson method. It provides tools for calculating flow rates, head losses, and pressure distributions in water networks.

## Features

- Read and parse EPANET INP files
- Support for both Imperial and Metric units
- Implementation of the Hazen-Williams equation
- Newton-Raphson method for solving network equations
- Automatic unit conversion between Imperial and Metric systems
- Calculation of head losses, flow rates, and pressure distributions

## Requirements

- Python 3.7 or higher
- NumPy
- WNTR (Water Network Tool for Resilience)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.simple_nr import WaterNetwork, Units

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
python src/simple_nr.py --inp_file_path "path/to/your/network.inp" --units Imperial
```

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


