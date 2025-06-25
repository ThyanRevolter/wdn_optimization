import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from mpc.mpc_wrapper import MPCWrapper
    from datetime import datetime
    from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX
    from time import sleep
    return DynamicWaterNetworkCVX, MPCWrapper, datetime, mo


@app.cell
def _(DynamicWaterNetworkCVX, datetime):
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
    return (
        params,
        prediction_horizon,
        simulation_end_date,
        simulation_start_date,
        simulation_time_step,
        update_interval,
    )


@app.cell
def _(
    MPCWrapper,
    params,
    prediction_horizon,
    simulation_end_date,
    simulation_start_date,
    simulation_time_step,
    update_interval,
):
    mpc_wrapper = MPCWrapper(params, simulation_start_date, simulation_end_date, simulation_time_step, update_interval, prediction_horizon)
    results = mpc_wrapper.run_mpc()
    opt_results_df = mpc_wrapper.concat_opt_results(results, "tank_level_TANK")
    return


@app.cell
def _(mo):
    mo.ui.slider.from_series()
    return


if __name__ == "__main__":
    app.run()
