import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from epanet_tutorial.wdn_pyomo import DynamicWaterNetwork as dwn
    from electric_emission_cost import costs
    import wntr
    import networkx as nx
    import matplotlib.pyplot as plt
    return costs, dwn, mo, wntr


@app.cell
def _():
    inp_file = "data/epanet_networks/simple_pump_tank.inp"
    return (inp_file,)


@app.cell
def _(mo):
    min_tank_level = mo.ui.slider(start=2.0, stop=5, step=0.1, label="Minimum Tank Level", show_value=True)
    max_tank_level = mo.ui.slider(start=10, stop=15, step=0.1, label="Maximum Tank Level", show_value=True)
    min_tank_level, max_tank_level
    return max_tank_level, min_tank_level


@app.cell
def _(wdn):
    wdn.wn
    return


@app.cell
def _(inp_file, wntr):
    network_file = wntr.network.WaterNetworkModel(inp_file)
    return


@app.cell
def _(wdn):
    sum(wdn.charge_dict.values())
    return


@app.cell
def _(dwn, inp_file):
    wdn = dwn(inp_file)
    wdn.solve()
    return (wdn,)


@app.cell
def _(wdn, x):
    x
    wdn.results.solver.termination_condition.value
    return


@app.cell
def _(max_tank_level, min_tank_level, wdn):
    wdn.model.min_tank_level_TANK = min_tank_level.value
    wdn.model.max_tank_level_TANK = max_tank_level.value
    wdn.solve()
    x = 1
    return (x,)


@app.cell
def _(wdn, x):
    x
    flows_df = wdn.package_flows_results()
    tank_df = wdn.package_tank_results()
    demand_df = wdn.package_demand_results()
    power_df = wdn.package_power_results()
    return (power_df,)


@app.cell
def _(costs, power_df, wdn, x):
    # itemized cost
    x
    power_consumed = power_df["total_power"].values
    itemized_cost = costs.calculate_itemized_cost(
        wdn.charge_dict,
        {"electric": power_consumed},
        resolution="1h"
    )
    itemized_cost
    return


@app.cell
def _(wdn, x):
    x
    fig, axs = wdn.plot_results()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
