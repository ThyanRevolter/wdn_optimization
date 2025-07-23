import pytest
from datetime import datetime, timedelta
import pandas as pd
from mpc.mpc_wrapper import MPCWrapper
from utils import utils as ut

# Define test cases for MPCWrapper
MPC_TEST_CASES = [
    pytest.param(
        {
            "simulation_start_date": datetime(2025, 1, 1, 0, 0),
            "simulation_end_date": datetime(2025, 1, 2, 0, 0),
            "simulation_time_step": 3600,
            "model_update_interval": 4 * 3600,
            "model_prediction_horizon": 12,
            "params_path": "tests/data/simple_pump_tank/simple_pump_tank_network_opt_params.json"
        },
        {
            "expected_horizons": {
                0: {
                    'optimization_start_time': datetime(2025, 1, 1, 0, 0),
                    'optimization_end_time': datetime(2025, 1, 1, 12, 0),
                    'optimization_time_step': 3600
                },
                1: {
                    'optimization_start_time': datetime(2025, 1, 1, 4, 0),
                    'optimization_end_time': datetime(2025, 1, 1, 16, 0),
                    'optimization_time_step': 3600
                },
                2: {
                    'optimization_start_time': datetime(2025, 1, 1, 8, 0),
                    'optimization_end_time': datetime(2025, 1, 1, 20, 0),
                    'optimization_time_step': 3600
                },
                3: {
                    'optimization_start_time': datetime(2025, 1, 1, 12, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 0, 0),
                    'optimization_time_step': 3600
                },
                4: {
                    'optimization_start_time': datetime(2025, 1, 1, 16, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 4, 0),
                    'optimization_time_step': 3600
                },
                5: {
                    'optimization_start_time': datetime(2025, 1, 1, 20, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 8, 0),
                    'optimization_time_step': 3600
                }
            },
            "expected_columns": [
                'Datetime', 'pipe_flow_P1', 'pipe_flow_P2', 'pump_flow_PUMP1',
                'pump_power_PUMP1', 'demand_J2', 'reservoir_flow_RES',
                'tank_level_TANK', 'tank_volume_TANK', 'tank_min_volume_TANK',
                'tank_max_volume_TANK', 'total_power', 'electricity_charge'
            ],
            "expected_rows": 12,
        },
        id="simple_pump_tank_mpc"
    ),
    pytest.param(
        {
        "simulation_start_date": datetime(2025, 1, 1, 0, 0),
        "simulation_end_date": datetime(2025, 1, 2, 0, 0),
        "simulation_time_step": 3600,
        "model_update_interval": 4 * 3600,
        "model_prediction_horizon": 24,
        "params_path": "tests/data/soporon/soporon_network_opt_params.json"
        },
        {
            "expected_horizons": {
                0: {
                    'optimization_start_time': datetime(2025, 1, 1, 0, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 0, 0),
                    'optimization_time_step': 3600
                },
                1: {
                    'optimization_start_time': datetime(2025, 1, 1, 4, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 4, 0),
                    'optimization_time_step': 3600
                },
                2: {
                    'optimization_start_time': datetime(2025, 1, 1, 8, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 8, 0),
                    'optimization_time_step': 3600
                },
                3: {
                    'optimization_start_time': datetime(2025, 1, 1, 12, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 12, 0),
                    'optimization_time_step': 3600
                },
                4: {
                    'optimization_start_time': datetime(2025, 1, 1, 16, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 16, 0),
                    'optimization_time_step': 3600
                },
                5: {
                    'optimization_start_time': datetime(2025, 1, 1, 20, 0),
                    'optimization_end_time': datetime(2025, 1, 2, 20, 0),
                    'optimization_time_step': 3600
                }
            },
            "expected_columns": [
                'Datetime', 'pipe_flow_12', 'pipe_flow_13', 'pipe_flow_14',
                'pipe_flow_15', 'pipe_flow_16', 'pipe_flow_17', 'pipe_flow_18',
                'pipe_flow_19', 'pipe_flow_20', 'pipe_flow_1', 'pipe_flow_2',
                'pipe_flow_3', 'pump_flow_4', 'pump_power_4', 'pump_flow_5',
                'pump_power_5', 'pump_flow_6', 'pump_power_6', 'pump_flow_7',
                'pump_power_7', 'pump_flow_8', 'pump_power_8', 'pump_flow_9',
                'pump_power_9', 'pump_flow_10', 'pump_power_10', 'pump_flow_11',
                'pump_power_11', 'demand_9', 'demand_10', 'demand_11', 'demand_12',
                'demand_13', 'reservoir_flow_1', 'reservoir_flow_2', 'reservoir_flow_3',
                'reservoir_flow_4', 'reservoir_flow_5', 'tank_level_14',
                'tank_volume_14', 'tank_min_volume_14', 'tank_max_volume_14',
                'tank_level_15', 'tank_volume_15', 'tank_min_volume_15',
                'tank_max_volume_15', 'tank_level_16', 'tank_volume_16',
                'tank_min_volume_16', 'tank_max_volume_16', 'tank_level_17',
                'tank_volume_17', 'tank_min_volume_17', 'tank_max_volume_17',
                'tank_level_18', 'tank_volume_18', 'tank_min_volume_18',
                'tank_max_volume_18', 'tank_level_19', 'tank_volume_19',
                'tank_min_volume_19', 'tank_max_volume_19', 'tank_level_20',
                'tank_volume_20', 'tank_min_volume_20', 'tank_max_volume_20',
                'tank_level_21', 'tank_volume_21', 'tank_min_volume_21',
                'tank_max_volume_21', 'total_power', 'electricity_charge'
            ],
            "expected_rows": 24,
        },
        id="soporon_mpc"
    )
]

@pytest.fixture(scope="module")
def mpc_wrapper_results(request):
    """Fixture to create an MPCWrapper instance for each test case."""
    opt_params = ut.load_json_file(request.param["params_path"])
    # Set up MPC parameters (simulate a 24-hour period, 1-hour steps)
    simulation_start_date = request.param["simulation_start_date"]
    simulation_end_date = request.param["simulation_end_date"]
    simulation_time_step = request.param["simulation_time_step"]
    model_update_interval = request.param["model_update_interval"]
    model_prediction_horizon = request.param["model_prediction_horizon"]
    mpc_params = {
        "optimization_params": opt_params,
        "simulation_start_date": simulation_start_date,
        "simulation_end_date": simulation_end_date,
        "simulation_time_step": simulation_time_step,
        "model_update_interval": model_update_interval,
        "model_prediction_horizon": model_prediction_horizon
    }
    mpc_wrapper = MPCWrapper(mpc_params)
    results = mpc_wrapper.run_mpc()
    return {"mpc_wrapper": mpc_wrapper, "results": results}

@pytest.mark.parametrize("mpc_wrapper_results, expected", MPC_TEST_CASES, indirect=["mpc_wrapper_results"])
class TestMPCWrapper:
    def test_mpc_time_horizons(self, mpc_wrapper_results, expected):
        """Test that the MPC time horizons are created correctly."""
        assert mpc_wrapper_results["mpc_wrapper"].mpc_time_horizon == expected["expected_horizons"]

    def test_mpc_run_shape_and_columns(self, mpc_wrapper_results, expected):
        """Test that the MPC simulation runs and produces a DataFrame with expected columns and rows."""
        results = mpc_wrapper_results["results"]
        assert len(results) == len(expected["expected_horizons"])
        for i, result_df in results.items():
            print(f"columns: {result_df.columns}")
            assert result_df.shape == (expected["expected_rows"], len(expected["expected_columns"]))
            assert result_df["Datetime"].iloc[0] == expected["expected_horizons"][i]["optimization_start_time"]
            assert result_df["Datetime"].iloc[-1] == expected["expected_horizons"][i]["optimization_end_time"] - timedelta(hours=mpc_wrapper_results["mpc_wrapper"].simulation_time_step/3600)

    def test_mpc_tank_level_continuity(self, mpc_wrapper_results, expected):
        """Test that tank levels are continuous between MPC horizons.
        
        For each horizon after the first, the initial tank level should match
        the tank level at the same time from the previous horizon's optimization.
        """
        # Get tank names from the network by creating a temporary WDN instance
        # to access the network structure
        temp_params = mpc_wrapper_results["mpc_wrapper"].params.copy()
        temp_params["optimization_start_time"] = mpc_wrapper_results["mpc_wrapper"].simulation_start_date.strftime("%Y-%m-%d %H:%M:%S")
        temp_params["optimization_end_time"] = mpc_wrapper_results["mpc_wrapper"].simulation_end_date.strftime("%Y-%m-%d %H:%M:%S")
        temp_params["optimization_time_step"] = mpc_wrapper_results["mpc_wrapper"].simulation_time_step
        
        from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX
        temp_wdn = DynamicWaterNetworkCVX(temp_params)
        
        tank_names = []
        for tank in temp_wdn.wn["nodes"]:
            if tank["node_type"] == "Tank":
                tank_names.append(tank["name"])
        
        assert len(tank_names) > 0, "No tanks found in the network"
        
        # Check continuity for each tank
        for tank_name in tank_names:
            tank_level_column = f"tank_level_{tank_name}"
            
            # For each horizon after the first
            for horizon_idx in range(1, len(mpc_wrapper_results["results"])):
                current_horizon = mpc_wrapper_results["results"][horizon_idx]
                previous_horizon = mpc_wrapper_results["results"][horizon_idx - 1]
                
                # Get the start time of the current horizon
                current_start_time = mpc_wrapper_results["mpc_wrapper"].mpc_time_horizon[horizon_idx]["optimization_start_time"]
                
                # Find the tank level at the current horizon start time from the previous horizon
                # We need to find the row in the previous horizon that corresponds to the current start time
                previous_horizon_start = mpc_wrapper_results["mpc_wrapper"].mpc_time_horizon[horizon_idx - 1]["optimization_start_time"]
                time_diff = (current_start_time - previous_horizon_start).total_seconds()
                time_step_seconds = mpc_wrapper_results["mpc_wrapper"].simulation_time_step
                time_index = int(time_diff / time_step_seconds)
                
                # Get the tank level from the previous horizon at the current start time
                if time_index < len(previous_horizon):
                    expected_tank_level = previous_horizon[tank_level_column].iloc[time_index]
                else:
                    # If the time index is beyond the previous horizon, use the last value
                    expected_tank_level = previous_horizon[tank_level_column].iloc[-1]
                
                # Get the initial tank level from the current horizon
                actual_tank_level = current_horizon[tank_level_column].iloc[0]
                
                # Assert that the tank levels match (with some tolerance for floating point precision)
                assert abs(actual_tank_level - expected_tank_level) < 1e-6, (
                    f"Tank level continuity violated for {tank_name} at horizon {horizon_idx}. "
                    f"Expected: {expected_tank_level}, Actual: {actual_tank_level}. "
                    f"Current horizon start: {current_start_time}, "
                    f"Previous horizon start: {previous_horizon_start}"
                )
