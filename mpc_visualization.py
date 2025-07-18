import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from mpc.mpc_wrapper import MPCWrapper
    from datetime import datetime, timedelta
    from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX
    from time import sleep
    from electric_emission_cost import costs
    import numpy as np
    import pandas as pd
    import os
    return (
        DynamicWaterNetworkCVX,
        MPCWrapper,
        datetime,
        mo,
        np,
        os,
        pd,
        plt,
        timedelta,
    )


@app.cell
def _(DynamicWaterNetworkCVX, datetime, os):
    simulation_start_date = datetime.strptime(
        "2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    simulation_end_date = datetime.strptime(
        "2025-01-02 00:00:00", "%Y-%m-%d %H:%M:%S"
    )
    simulation_time_step = 3600
    model_update_interval = 6*3600
    model_prediction_horizon = 24


    params_path_file = "soporon_network_opt_params.json"
    # params_path_file = "simple_pump_tank_network_opt_params.json"
    params_path = os.path.join('data', params_path_file)
    params = DynamicWaterNetworkCVX.load_optimization_params(params_path)
    mpc_params = {
        "optimization_params": params,
        "simulation_start_date": simulation_start_date,
        "simulation_end_date": simulation_end_date,
        "simulation_time_step": simulation_time_step,
        "model_update_interval": model_update_interval,
        "model_prediction_horizon": model_prediction_horizon
    }
    return model_update_interval, mpc_params, params, simulation_start_date


@app.cell
def _(MPCWrapper, mpc_params):
    mpc_wrapper = MPCWrapper(mpc_params)
    results = mpc_wrapper.run_mpc()
    return mpc_wrapper, results


@app.cell
def _(mpc_wrapper, results):
    concatinated_results = {}
    for column in results[0].columns:
        if column != "Datetime":
            opt_results_df = mpc_wrapper.concat_opt_results(results, column)
            concatinated_results[column] = opt_results_df
    return (concatinated_results,)


@app.cell
def _(mpc_wrapper, results):
    actual_results = mpc_wrapper.get_actual_operations(results)
    prescient_results = mpc_wrapper.get_prescient_operations()
    return actual_results, prescient_results


@app.cell
def _(DynamicWaterNetworkCVX, actual_results, mo, pd, prescient_results):
    rate_df = pd.read_csv("data/operational_data/tariff.csv", sep=",")
    actual_electricity_cost = DynamicWaterNetworkCVX.get_electricity_cost(actual_results, rate_df)
    prescient_electricity_cost = DynamicWaterNetworkCVX.get_electricity_cost(prescient_results, rate_df)

    mo.md(
        f"""
        ## Water Distribution Network Optimization Results
        Operating under MPC control
        Electricity Cost: ${actual_electricity_cost:.2f}

        Operating under prescient control
        Electricity Cost: ${prescient_electricity_cost:.2f}

        Percent change from prescient operation: {((-prescient_electricity_cost + actual_electricity_cost)/prescient_electricity_cost)*100:.2f}%
        """)
    return


@app.cell
def _(DynamicWaterNetworkCVX, params):
    wdn = DynamicWaterNetworkCVX(params)
    return (wdn,)


@app.cell
def _():
    # fig, ax = DynamicWaterNetworkCVX.plot_results(wdn.wn, actual_results)
    return


@app.cell
def _():
    # fig1, ax1 = DynamicWaterNetworkCVX.plot_results(wdn.wn, prescient_results)
    return


@app.cell
def _(
    actual_results,
    change_idx,
    charge_to_shade,
    charges,
    plt,
    prescient_results,
):
    figx, ax_dual = plt.subplots(figsize=(12, 6))
    ax_dual.plot(prescient_results["Datetime"], prescient_results["total_power"], label="Prescient Total Power", linestyle='--')
    ax_dual.plot(actual_results["Datetime"], actual_results["total_power"], label="Actual Total Power")
    for i, idx in enumerate(change_idx):
        if i == 0:
            start_idx = 0
            end_idx = idx
        elif i == len(change_idx) - 1:
            start_idx = change_idx[i - 1] + 1 
            end_idx = idx + 1
            final_charge = actual_results["electricity_charge"].iloc[end_idx]
            ax_dual.axvspan(
                actual_results["Datetime"].iloc[end_idx],
                actual_results["Datetime"].iloc[-1],
                color=str(charge_to_shade[final_charge]),
                alpha=0.5,
                lw=0

            )
        else:
            start_idx = change_idx[i - 1] + 1
            end_idx = idx + 1
        ax_dual.axvspan(
            actual_results["Datetime"].iloc[start_idx],
            actual_results["Datetime"].iloc[end_idx],
            color=str(charge_to_shade[charges[idx]]),
            alpha=0.75
        )
    # ax2_dual = ax_dual.twinx()
    # ax2_dual.plot(actual_results["Datetime"], actual_results["electricity_charge"], label="Electricity Charge", color='orange', linestyle='--')
    ax_dual.set_xlabel("Datetime")
    ax_dual.set_ylabel("Total Power (kW)")
    ax_dual.legend()
    return


@app.cell
def _(actual_results, np):
    unique_charges = np.sort(actual_results["electricity_charge"].unique())
    charge_shades = np.linspace(1, 0.3, len(unique_charges))
    charge_to_shade = {charge: shade for charge, shade in zip(unique_charges, charge_shades)}
    charges = actual_results["electricity_charge"].values
    changed_loc = list(charges[1:] - charges[:-1] != 0) + [False]
    change_idx = actual_results.iloc[changed_loc]["electricity_charge"].index.values
    return change_idx, charge_to_shade, charges


@app.cell
def _(mo, prescient_results, wdn):
    # wdn.wn["links"]
    pipe_names = [pipe['name'] for pipe in wdn.wn["links"] if pipe['link_type'] == 'Pipe']
    pump_names = [pump['name'] for pump in wdn.wn["links"] if pump['link_type'] == 'Pump']
    junction_names = [junction["name"] for junction in wdn.wn["nodes"] if (junction["node_type"] == "Junction" and f"demand_{junction['name']}" in prescient_results.columns)]
    tank_names = [tank["name"] for tank in wdn.wn["nodes"] if tank["node_type"] == "Tank"]
    reservoir_names = [reservoir["name"] for reservoir in wdn.wn["nodes"] if reservoir["node_type"] == "Reservoir"]

    pipe_selector = mo.ui.dropdown(options=pipe_names, label="Select Pipe", value=pipe_names[0], full_width=True)
    pump_selector = mo.ui.dropdown(options=pump_names, label="Select Pump", value=pump_names[0], full_width=True)
    junction_selector = mo.ui.dropdown(options=junction_names, label="Select Junction", value=junction_names[0], full_width=True)
    tank_selector = mo.ui.dropdown(options=tank_names, label="Select Tank", value=tank_names[0], full_width=True)
    reservoir_selector = mo.ui.dropdown(options=reservoir_names, label="Select Reservoir", value=reservoir_names[0], full_width=True)
    return (
        junction_selector,
        pipe_selector,
        pump_selector,
        reservoir_selector,
        tank_selector,
    )


@app.cell
def _(actual_results, plt, prescient_results):
    def plot_pump_flows(pump_name):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(prescient_results["Datetime"], prescient_results[f"pump_flow_{pump_name}"], label=pump_name + " Prescient Flow", where='post')
        ax.step(actual_results["Datetime"], actual_results[f"pump_flow_{pump_name}"], label=pump_name + " Actual Flow", where='post')
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Pump Flow (m^3/s)")
        ax.set_title("Pump Flow")
        ax.legend()
        return fig, ax

    def plot_pipe_flows(pipe_name):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prescient_results["Datetime"], prescient_results[f"pipe_flow_{pipe_name}"], label=pipe_name + " Prescient Flow")
        ax.plot(actual_results["Datetime"], actual_results[f"pipe_flow_{pipe_name}"], label=pipe_name + " Actual Flow")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Pipe Flow (m^3/s)")
        ax.set_title("Pipe Flow")
        ax.legend()
        return fig, ax

    def plot_junction_demand(junction_name):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prescient_results["Datetime"], prescient_results[f"demand_{junction_name}"], label=junction_name + " Predicted Demand")
        ax.plot(actual_results["Datetime"], actual_results[f"demand_{junction_name}"], label=junction_name + " Actual Demand")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Demand (m^3/s)")
        ax.set_title("Junction Demand")
        ax.legend()
        return fig, ax

    def plot_tank_levels(tank_name):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prescient_results["Datetime"], prescient_results[f"tank_level_{tank_name}"], label=tank_name + " Prescient Level")
        ax.plot(actual_results["Datetime"], actual_results[f"tank_level_{tank_name}"], label=tank_name + " Actual Level")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Level (m)")
        ax.set_title("Tank Levels")
        ax.legend()
        return fig, ax

    def plot_reservoir_flows(reservoir_name):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prescient_results["Datetime"], prescient_results[f"reservoir_flow_{reservoir_name}"], label=reservoir_name + " Prescient Flow")
        ax.plot(actual_results["Datetime"], actual_results[f"reservoir_flow_{reservoir_name}"], label=reservoir_name + " Actual Flow")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Flow (m^3/s)")
        ax.set_title("Reservoir Flow")
        ax.legend()
        return fig, ax
    return (
        plot_junction_demand,
        plot_pipe_flows,
        plot_pump_flows,
        plot_reservoir_flows,
        plot_tank_levels,
    )


@app.cell
def _(model_update_interval, np, plt, simulation_start_date, timedelta):
    def plot_mpc_timeseries(concat_operational_data, label, label_value):
        _timesteps = concat_operational_data.shape[1] -1
        _ymin = np.min(
            concat_operational_data.iloc[:, 1:]
        ) * 0.9
        _ymax = np.max(
            concat_operational_data.iloc[:, 1:]
        ) * 1.1
        _fig, _ax = plt.subplots(_timesteps, 1, figsize=(12, _timesteps), sharex=True)
        for _i in range(_timesteps):
            _ax[_i].plot(
                concat_operational_data["Datetime"],
                concat_operational_data.iloc[:, _i + 1],
                label=f"{label} prediction at {_i}",
            )
            _control_start = timedelta(seconds=_i * model_update_interval) + simulation_start_date
            _ax[_i].axvspan(
                _control_start,
                _control_start + timedelta(seconds=model_update_interval),
                color='lightgray',
                alpha=0.5
            )
            _ax[_i].set_ylim(bottom=_ymin, top=_ymax)
        _fig.supxlabel('DateTime')
        _fig.supylabel(f'{label} for {label_value}')
        _fig.tight_layout()
        return _fig, _ax    
    return (plot_mpc_timeseries,)


@app.cell
def _(
    junction_selector,
    pipe_selector,
    pump_selector,
    reservoir_selector,
    tank_selector,
):
    pipe_value = pipe_selector.value
    pump_value = pump_selector.value
    junction_value = junction_selector.value
    tank_value = tank_selector.value
    reservoir_value = reservoir_selector.value
    return junction_value, pipe_value, pump_value, reservoir_value, tank_value


@app.cell
def _(pump_selector):
    pump_selector
    return


@app.cell
def _(plot_pump_flows, pump_value):
    plot_pump_flows(pump_value)
    return


@app.cell
def _(concatinated_results, plot_mpc_timeseries, pump_value):
    pump_df = concatinated_results[f"pump_flow_{pump_value}"]
    fig_pump, ax_pump = plot_mpc_timeseries(pump_df, "Pump Flow", pump_value)
    fig_pump
    return


@app.cell
def _(pipe_selector):
    pipe_selector
    return


@app.cell
def _(pipe_value, plot_pipe_flows):
    plot_pipe_flows(pipe_value)
    return


@app.cell
def _(concatinated_results, pipe_value, plot_mpc_timeseries):
    pipe_df = concatinated_results[f"pipe_flow_{pipe_value}"]
    fig_pipe, ax_pipe = plot_mpc_timeseries(pipe_df, "Pipe Flow", pipe_value)
    fig_pipe
    return


@app.cell
def _(junction_selector):
    junction_selector
    return


@app.cell
def _(junction_value, plot_junction_demand):
    plot_junction_demand(junction_value)
    return


@app.cell
def _(concatinated_results, junction_value, pipe_value, plot_mpc_timeseries):
    demand_df = concatinated_results[f"demand_{junction_value}"]
    fig_demand, ax_demand = plot_mpc_timeseries(demand_df, "Demand", pipe_value)
    fig_demand
    return


@app.cell
def _(tank_selector):
    tank_selector
    return


@app.cell
def _(plot_tank_levels, tank_value):
    plot_tank_levels(tank_value)
    return


@app.cell
def _(concatinated_results, plot_mpc_timeseries, tank_value):
    tank_df = concatinated_results[f"tank_level_{tank_value}"]
    fig_tank, ax_tank = plot_mpc_timeseries(tank_df, "Tank Level", tank_value)
    fig_tank
    return


@app.cell(hide_code=True)
def _(mo, prescient_results, tank_value):
    mo.md(
        f"""
    Prescient results for Tank {tank_value}

    Initial level of tank : {prescient_results[f"tank_level_{tank_value}"].iloc[0]:.2f} m

    Final level of tank : {prescient_results[f"tank_level_{tank_value}"].iloc[-1]:.2f} m
    """
    )
    return


@app.cell
def _(reservoir_selector):
    reservoir_selector
    return


@app.cell
def _(plot_reservoir_flows, reservoir_value):
    plot_reservoir_flows(reservoir_value)
    return


@app.cell
def _(concatinated_results, plot_mpc_timeseries, reservoir_value):
    reservoir_df = concatinated_results[f"reservoir_flow_{reservoir_value}"]
    fig_reservoir, ax_reservoir = plot_mpc_timeseries(reservoir_df, "Reservoir Flow", reservoir_value)
    fig_reservoir
    return


if __name__ == "__main__":
    app.run()
