"""
Utility functions for the optimization, plotting, and other utilities.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from electric_emission_cost import costs

def load_json_file(params_path: str) -> dict:
    """
    Load a JSON file.
    
    Args:
        params_path (str): Path to the JSON file.
        
    Returns:
        dict: Dictionary containing the JSON file.
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params

def plot_results(wdn: dict, packaged_data: pd.DataFrame, save_to_file: bool = False):
    """
    Plot the results.
    """
    
    fig, axs = plt.subplots(6, 1, figsize=(10, 30))
    for pipe in wdn["links"]:
        if pipe["link_type"] == "Pipe":
            axs[0].plot(
                packaged_data["Datetime"],
                packaged_data[f'pipe_flow_{pipe["name"]}'],
                label=pipe["name"],
            )
    axs[0].legend()
    axs[0].set_title("Pipe Flows")
    axs[0].set_ylabel("Flow (m³/h)")
    for pump in wdn["links"]:
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
    for tank in wdn["nodes"]:
        if tank["node_type"] == "Tank":
            axs[2].plot(
                packaged_data["Datetime"],
                packaged_data[f'tank_level_{tank["name"]}'],
                label=tank["name"],
            )
    axs[2].legend()
    axs[2].set_title("Tank Levels")
    axs[2].set_ylabel("Level (m)")
    for demand_node in wdn["nodes"]:
        if demand_node["node_type"] == "Junction" and f"demand_{demand_node['name']}" in packaged_data.columns:
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
        plt.savefig(f"data/local/plots/results_{packaged_data['Datetime'][0].strftime('%Y%m%d')}_{packaged_data['Datetime'][-1].strftime('%Y%m%d')}.png")
    else:
        plt.show()
    return fig, axs

def get_electricity_cost(operation_data: pd.DataFrame, rate_df: pd.DataFrame, resolution: str = "1h"):
    """
    Get the electricity cost.
    """
    charge_dict = costs.get_charge_dict(
        operation_data["Datetime"].iloc[0], operation_data["Datetime"].iloc[-1] + timedelta(hours=1), rate_df, resolution=resolution
    )
    consumption_data_dict = {"electric": operation_data["total_power"].values}
    electricity_cost, _ = costs.calculate_cost(
        charge_dict,
        consumption_data_dict,
        resolution=resolution,
        prev_demand_dict=None,
        prev_consumption_dict=None,
        consumption_estimate=0,
        desired_utility="electric",
        desired_charge_type=None,
    )
    return electricity_cost
