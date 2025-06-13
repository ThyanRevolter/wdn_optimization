import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from epanet_tutorial.wdn_pyomo import DynamicWaterNetwork as dwn
    from epanet_tutorial.wdn_cvxpy import DynamicWaterNetworkCVX as cpdwn
    from electric_emission_cost import costs
    import wntr
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np

    return cpdwn, mo, plt


@app.cell
def _():
    # inp_file = "data/epanet_networks/simple_pump_tank.inp"
    # pump_data_file_path = None

    inp_file = "data/epanet_networks/sopron_network.inp"
    pump_data_file_path = r"data/operational_data/sopron_network_pump_data.csv"
    reservoir_file_path = r"data/operational_data/sopron_network_reservoir_data.csv"
    return inp_file, pump_data_file_path, reservoir_file_path


@app.cell
def _(cpdwn, inp_file, mo, pump_data_file_path, reservoir_file_path):
    wdn = cpdwn(
        inp_file,
        pump_data_path=pump_data_file_path,
        reservoir_data_path=reservoir_file_path,
    )
    optimal_cost = wdn.solve(verbose=True, time_limit=300)
    results_df = wdn.package_data()
    mo.md(
        f"""
    The optimal cost of the water distribution network is: ${optimal_cost:.2f}
    """
    )
    return results_df, wdn


@app.cell
def _(wdn):
    tanks_list = [
        tank["name"] for tank in wdn.wn["nodes"] if tank["node_type"] == "Tank"
    ]
    junction_list = [
        junction["name"]
        for junction in wdn.wn["nodes"]
        if junction["node_type"] == "Junction"
    ]
    pipe_list = [
        pipe["name"] for pipe in wdn.wn["links"] if pipe["link_type"] == "Pipe"
    ]
    pump_list = [
        pump["name"] for pump in wdn.wn["links"] if pump["link_type"] == "Pump"
    ]
    return junction_list, pipe_list, pump_list, tanks_list


@app.cell
def _(junction_list, mo, pipe_list, pump_list, tanks_list):
    tank_selector = mo.ui.multiselect(
        options=tanks_list,
        label="Select Tanks",
    )
    junction_selector = mo.ui.multiselect(
        options=junction_list,
        label="Select Junctions",
    )
    pipe_selector = mo.ui.multiselect(
        options=pipe_list,
        label="Select Pipes",
    )
    pump_selector = mo.ui.multiselect(
        options=pump_list,
        label="Select Pumps",
    )
    tank_selector, junction_selector, pipe_selector, pump_selector
    return junction_selector, pipe_selector, pump_selector, tank_selector


@app.cell
def _():
    return


@app.cell
def _(
    junction_selector,
    pipe_selector,
    plot_selected_results,
    pump_selector,
    results_df,
    tank_selector,
):
    fig, axs = plot_selected_results(
        results_df, tank_selector, junction_selector, pipe_selector, pump_selector
    )
    fig
    return


@app.cell
def _(plt):
    def plot_selected_results(
        results_df, tank_selector, junction_selector, pipe_selector, pump_selector
    ):
        fig, axs = plt.subplots(3, 2, figsize=(18, 18), sharex=True)
        axs = axs.flatten()

        # Time axis
        time_steps = results_df["Datetime"]

        # 1. Pipe Flows
        for pipe in pipe_selector.value:
            col_name = f"pipe_flow_{pipe}"
            if col_name in results_df.columns:
                axs[0].plot(time_steps, results_df[col_name], label=f"Pipe {pipe}")
        axs[0].legend()
        axs[0].set_title("Pipe Flows")
        axs[0].set_ylabel("Flow (m³/h)")

        # 2. Pump Flows
        for pump in pump_selector.value:
            col_name = f"pump_flow_{pump}"
            if col_name in results_df.columns:
                axs[1].step(
                    time_steps, results_df[col_name], label=f"Pump {pump}", where="post"
                )
        axs[1].legend()
        axs[1].set_title("Pump Flows")
        axs[1].set_ylabel("Flow (m³/h)")

        # 3. Tank Volumes
        for tank in tank_selector.value:
            vol_col = f"tank_volume_{tank}"
            if vol_col in results_df.columns:
                axs[2].plot(time_steps, results_df[vol_col], label=f"Tank {tank}")
        axs[2].legend(loc="upper right")
        axs[2].set_title("Tank Volumes")
        axs[2].set_ylabel("Volume (m³)")

        # 4. Junction Demand
        for junction in junction_selector.value:
            col_name = f"demand_{junction}"
            if col_name in results_df.columns:
                axs[3].plot(
                    time_steps, results_df[col_name], label=f"Junction {junction}"
                )
        axs[3].legend()
        axs[3].set_title("Demand")
        axs[3].set_ylabel("Demand (m³/h)")

        # 5. Power Consumption
        if "total_power" in results_df.columns:
            axs[4].step(
                time_steps, results_df["total_power"], label="Total Power", where="post"
            )
        axs[4].legend()
        axs[4].set_title("Power")
        axs[4].set_ylabel("Power (kW)")

        # 6. Electricity Charges
        if "electricity_charge" in results_df.columns:
            axs[5].step(
                time_steps,
                results_df["electricity_charge"],
                label="Electricity Charge",
                where="post",
            )
        axs[5].legend()
        axs[5].set_title("Electricity Charges")
        axs[5].set_ylabel("Cost ($/kWh)")

        # Shared x-axis label for bottom plots
        axs[4].set_xlabel("Datetime")
        axs[5].set_xlabel("Datetime")

        plt.tight_layout()
        return fig, axs

    return (plot_selected_results,)


if __name__ == "__main__":
    app.run()
