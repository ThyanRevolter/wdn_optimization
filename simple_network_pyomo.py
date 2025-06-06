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
    return costs, dwn, wntr


@app.cell
def _():
    inp_file = "data/epanet_networks/simple_pump_tank.inp"
    return (inp_file,)


@app.cell
def _(inp_file, wntr):
    network_file = wntr.network.WaterNetworkModel(inp_file)
    return


@app.cell
def _(dwn, inp_file):
    wdn = dwn(inp_file)
    wdn.solve()
    return (wdn,)


@app.cell
def _(wdn):
    flows_df = wdn.package_flows_results()
    tank_df = wdn.package_tank_results()
    demand_df = wdn.package_demand_results()
    power_df = wdn.package_power_results()
    return (power_df,)


@app.cell
def _(costs, power_df, wdn):
    # itemized cost
    power_consumed = power_df["total_power"].values
    itemized_cost = costs.calculate_itemized_cost(
        wdn.charge_dict,
        {"electric": power_consumed},
        resolution="1h"
    )
    itemized_cost
    return


@app.cell
def _(wdn):
    fig, axs = wdn.plot_results()
    return (fig,)


@app.cell
def _(fig):
    fig
    return


if __name__ == "__main__":
    app.run()
