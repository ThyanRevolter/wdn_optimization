"""
This module uses the MPC class to wrap around the water network model and the optimization problem. 

Features:
- Reads the network model from the INP file
- Reads the operational data from the CSV files
- Runs the MPC simulation
"""

import json
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX



class MPCWrapper:

    def __init__(
            self, 
            params: dict, 
            simulation_start_date: datetime, 
            simulation_end_date: datetime, 
            simulation_time_step: int, 
            update_interval: int, 
            prediction_horizon: int
        ):
        self.simulation_start_date = simulation_start_date
        self.simulation_end_date = simulation_end_date
        self.simulation_time_step = simulation_time_step
        self.update_interval = update_interval # time step for updating the optimization problem
        self.prediction_horizon = prediction_horizon # time step for the prediction horizon
        self.params = params
        self.mpc_time_horizon = self.create_mpc_time_horizon()

    def create_mpc_time_horizon(self):
        """
        Create the MPC time horizon.
        """
        num_of_optimization_steps = int((self.simulation_end_date - self.simulation_start_date).total_seconds() / self.update_interval)
        mpc_time_horizon_data = {}
        for i in range(num_of_optimization_steps):
            mpc_time_horizon_data[i] = {
                "start_date": self.simulation_start_date + timedelta(hours=i * self.update_interval/3600),
                "end_date": min(self.simulation_end_date, self.simulation_start_date + timedelta(hours=(i * self.update_interval/3600) + self.prediction_horizon)),
                "time_step": self.simulation_time_step
            }
        return mpc_time_horizon_data
    
    def update_params(self, params, time_horizon):
        """
        Update the parameters for the MPC simulation.
        """
        params["start_date"] = time_horizon["start_date"].strftime("%Y-%m-%d %H:%M:%S")
        params["end_date"] = time_horizon["end_date"].strftime("%Y-%m-%d %H:%M:%S")
        params["time_step"] = time_horizon["time_step"]
    
    def update_init_tank_level(self, wdn, tank_levels):
        """
        Update the initial tank level for the MPC simulation.
        """
        for tank in wdn.wn["nodes"]:
            if tank["node_type"] == "Tank":
                current_tank_level = tank_levels[tank["name"]]
                wdn.constraints[f"tank_level_init_{tank['name']}"] = (
                    getattr(wdn, f"tank_level_{tank['name']}")[0] == current_tank_level
                )
    
    def get_tank_levels(self, wdn, time_stamp):
        """
        Get the tank levels for the MPC simulation.
        """
        tank_levels = {}
        for tank in wdn.wn["nodes"]:
            if tank["node_type"] == "Tank":
                tank_levels[tank["name"]] = wdn.get_tank_levels(tank["name"], time_stamp)
        return tank_levels
    
    def update_demand_data(self, wdn, demand_data):
        """
        Update the demand data for the MPC simulation.
        """
        # add some deviations to the demand data
        for column in demand_data.columns:
            if column != "Datetime":
                demand_data[column] = demand_data[column] * (1 + np.random.randn(len(demand_data)) * 0.05)
        wdn.set_demand_data(demand_data)
    
    def run_mpc(self):
        """
        Run the MPC simulation.
        """
        opt_results = {}
        tank_levels = {}
        for i, time_horizon in self.mpc_time_horizon.items():
            params = self.params.copy()
            self.update_params(params, time_horizon)
            wdn = DynamicWaterNetworkCVX(params)
            if i > 0:
                self.update_init_tank_level(wdn, tank_levels)
                self.update_demand_data(wdn, wdn.demand_data)
                wdn.build_optimization_model()
            wdn.solve()
            if i < len(self.mpc_time_horizon) - 1:
                next_step_start_date = self.mpc_time_horizon[i + 1]["start_date"]
                tank_levels = self.get_tank_levels(wdn, next_step_start_date)
            opt_results[i] = wdn.package_data(save_to_csv=False)
            print(f"opt results from {opt_results[i]['Datetime'].iloc[0]} to {opt_results[i]['Datetime'].iloc[-1]}")
        return opt_results

    def plot_tank_level(self, opt_results, tank_name):
        """
        Plot the tank levels.
        """
        for i, _ in self.mpc_time_horizon.items():
            plt.plot(opt_results[i]["Datetime"], opt_results[i][f"tank_level_{tank_name}"], label=f"Tank {i}")
        plt.legend()
        plt.show()

    def concat_opt_results(self, opt_results, column_name):
        """
        Concatenate the mpc optimization results based on "Datetime" column,
        creating a wide format with separate columns for each MPC horizon.
        """
        # Start with the first dataframe
        opt_results_df = opt_results[0][["Datetime"]].copy()
        opt_results_df[f"{column_name}_0"] = opt_results[0][column_name]
        
        # Add columns for subsequent horizons
        for i in range(1, len(opt_results)):
            horizon_df = opt_results[i][["Datetime", column_name]].copy()
            horizon_df = horizon_df.rename(columns={column_name: f"{column_name}_{i}"})
            
            # Merge with the main dataframe based on datetime
            opt_results_df = pd.merge(opt_results_df, horizon_df, on="Datetime", how="outer")
        
        # Sort by datetime
        opt_results_df = opt_results_df.sort_values(by="Datetime")
        
        return opt_results_df

if __name__ == "__main__":
    simulation_start_date = datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    simulation_end_date = datetime.strptime(
        "2025-01-02 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    simulation_time_step = 3600
    update_interval = 3600
    prediction_horizon = 24

    params_path = "data/simple_pump_tank_network_opt_params.json"
    params = DynamicWaterNetworkCVX.load_optimization_params(params_path)
    mpc_wrapper = MPCWrapper(params, simulation_start_date, simulation_end_date, simulation_time_step, update_interval, prediction_horizon)
    results = mpc_wrapper.run_mpc()
    opt_results_df = mpc_wrapper.concat_opt_results(results, "tank_level_TANK")
    # save the opt_results_df to a csv file
    opt_results_df.to_csv("opt_results_tank_level_TANK.csv", index=False)
    

