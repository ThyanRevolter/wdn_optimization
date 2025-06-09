import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

n = 24
demand = np.array([100, 100, 100, 100,
                   200, 400, 500, 600, 
                   400, 400, 400, 400, 
                   300, 200, 300, 400, 
                   600, 600, 400, 200,
                   200, 100, 100, 100])
cost = np.array([1, 1, 1, 1,
                 1, 2, 2, 2,
                 2, 4, 4, 4,
                 4, 4, 2, 2,
                 2, 2, 2, 1,
                 1, 1, 1, 1])

pump_flow_capacity = 700 # m3/h
min_tank_level = 100 # m3
max_tank_level = 2000 # m3

tank_volume = cp.Variable(n)
pump_flow = cp.Variable(n)

constraints = []
for i in range(n):
    constraints.append(tank_volume[i] >= min_tank_level)
    constraints.append(tank_volume[i] <= max_tank_level)
    constraints.append(pump_flow[i] <= pump_flow_capacity)
    constraints.append(pump_flow[i] >= 0)
    if i == n-1:
        print(f"V_{i} = V_{0} + pump_flow_{i} - demand_{i}")
        constraints.append(tank_volume[0] == tank_volume[i] + pump_flow[i] - demand[i])
    else:
        print(f"V_{i+1} = V_{i} + pump_flow_{i} - demand_{i}")
        constraints.append(tank_volume[i+1] == tank_volume[i] + pump_flow[i] - demand[i])

objective = cp.Minimize(cost.T @ pump_flow)
problem = cp.Problem(objective, constraints)
problem.solve()
print(f"Optimal pump flow: {pump_flow.value}")
print(f"Optimal tank volume: {tank_volume.value}")
# objective function
print(f"Objective function: {problem.value}")

# plot the results
plt.plot(pump_flow.value)
plt.show()
