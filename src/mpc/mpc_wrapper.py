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
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX



class MPCWrapper:

    def __init__(
            self, 
            mpc_params: dict
        ):
        self.simulation_start_date = mpc_params["simulation_start_date"]
        self.simulation_end_date = mpc_params["simulation_end_date"]
        self.simulation_time_step = mpc_params["simulation_time_step"]
        self.model_update_interval = mpc_params["model_update_interval"] # time step for updating the optimization problem
        self.model_prediction_horizon = mpc_params["model_prediction_horizon"] # time step for the prediction horizon
        self.params = mpc_params["optimization_params"]
        self.mpc_time_horizon = self.create_mpc_time_horizons()

    def create_mpc_time_horizons(self):
        """
        Create the MPC time horizon.
        """
        mpc_time_horizon_data = {}
        i = 0
        while self.simulation_start_date + timedelta(hours=(i * self.model_update_interval/3600)) < self.simulation_end_date:
            # print(f"i: {i}, optimization_start_time: {self.simulation_start_date + timedelta(hours=i * self.model_update_interval/3600)}, optimization_end_time: {self.simulation_start_date + timedelta(hours=(i * self.model_update_interval/3600) + self.model_prediction_horizon)}")
            mpc_time_horizon_data[i] = {
                "optimization_start_time": self.simulation_start_date + timedelta(hours=i * self.model_update_interval/3600),
                "optimization_end_time": self.simulation_start_date + timedelta(hours=(i * self.model_update_interval/3600) + self.model_prediction_horizon),
                "optimization_time_step": self.simulation_time_step
            }
            i += 1
        return mpc_time_horizon_data
    
    def update_time_horizon(self, params, time_horizon):
        """
        Update the parameters for the MPC simulation.
        """
        params["optimization_start_time"] = time_horizon["optimization_start_time"].strftime("%Y-%m-%d %H:%M:%S")
        params["optimization_end_time"] = time_horizon["optimization_end_time"].strftime("%Y-%m-%d %H:%M:%S")
        params["optimization_time_step"] = time_horizon["optimization_time_step"]
    
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
        start_time = time.time()
        for i, time_horizon in self.mpc_time_horizon.items():
            params = self.params.copy()
            self.update_time_horizon(params, time_horizon)
            wdn = DynamicWaterNetworkCVX(params)
            if i > 0:
                # self.update_demand_data(wdn, wdn.demand_data)
                wdn.build_optimization_model()
                self.update_init_tank_level(wdn, tank_levels)
            wdn.solve()
            if i < len(self.mpc_time_horizon) - 1:
                next_step_start_date = self.mpc_time_horizon[i + 1]["optimization_start_time"]
                tank_levels = self.get_tank_levels(wdn, next_step_start_date)
            opt_results[i] = wdn.package_data(save_to_csv=False)
            print(f"opt results from {opt_results[i]['Datetime'].iloc[0]} to {opt_results[i]['Datetime'].iloc[-1]}")
        end_time = time.time()
        print(f"MPC simulation time: {end_time - start_time} seconds")
        return opt_results

    def plot_tank_level(self, opt_results, tank_name):
        """
        Plot the tank levels.
        """
        for i, _ in self.mpc_time_horizon.items():
            plt.plot(opt_results[i]["Datetime"], opt_results[i][f"tank_level_{tank_name}"], label=f"Tank level at {i}th horizon")
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

    def get_actual_operations(self, opt_results):
        """
        Get the actual operations from the opt_results.
        """
        actual_operations = []
        operation_steps = int(self.model_update_interval / self.simulation_time_step)
        print(f"operation_steps: {operation_steps}")
        for i, _ in self.mpc_time_horizon.items():
            actual_operations += opt_results[i].iloc[:operation_steps].to_dict(orient="records")
        return pd.DataFrame(actual_operations)

    def get_prescient_operations(self):
        """
        Get the prescient operations from the opt_results.
        """
        opt_params = self.params.copy()
        opt_params["optimization_start_time"] = self.simulation_start_date.strftime("%Y-%m-%d %H:%M:%S")
        opt_params["optimization_end_time"] = self.simulation_end_date.strftime("%Y-%m-%d %H:%M:%S")
        opt_params["optimization_time_step"] = self.simulation_time_step
        wdn = DynamicWaterNetworkCVX(opt_params)
        wdn.solve()
        return wdn.package_data(save_to_csv=False)

if __name__ == "__main__":
    simulation_start_date = datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    simulation_end_date = datetime.strptime(
        "2025-01-02 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    simulation_time_step = 3600
    model_update_interval = 3600
    model_prediction_horizon = 24


    params_path = "data\soporon_network_opt_params.json"
    # params_path = "data\simple_pump_tank_network_opt_params.json"
    params = DynamicWaterNetworkCVX.load_optimization_params(params_path)
    mpc_params = {
        "optimization_params": params,
        "simulation_start_date": simulation_start_date,
        "simulation_end_date": simulation_end_date,
        "simulation_time_step": simulation_time_step,
        "model_update_interval": model_update_interval,
        "model_prediction_horizon": model_prediction_horizon
    }
    mpc_wrapper = MPCWrapper(mpc_params)
    results = mpc_wrapper.run_mpc()
    for column in results[0].columns:
        if column != "Datetime":
            opt_results_df = mpc_wrapper.concat_opt_results(results, column)
            # save the opt_results_df to a csv file
            opt_results_df.to_csv(f"data/local/mpc_results/opt_results_{column}.csv", index=False)
    
    actual_operations = mpc_wrapper.get_actual_operations(results)
    actual_operations.to_csv(f"data/local/mpc_results/actual_operations.csv", index=False)
    rate_df = pd.read_csv("data/operational_data/tariff.csv", sep=",")
    # get actual electricity cost
    actual_electricity_cost = DynamicWaterNetworkCVX.get_electricity_cost(actual_operations, rate_df)
    print(f"actual electricity cost: {actual_electricity_cost}")
    prescient_operations = mpc_wrapper.get_prescient_operations()
    prescient_operations.to_csv(f"data/local/mpc_results/prescient_operations.csv", index=False)
    # get prescient electricity cost
    prescient_electricity_cost = DynamicWaterNetworkCVX.get_electricity_cost(prescient_operations, rate_df)
    print(f"prescient electricity cost: {prescient_electricity_cost}")


