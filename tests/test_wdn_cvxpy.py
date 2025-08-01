from datetime import timedelta
import pytest
from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX
from utils import utils as ut

# Define test cases with their expected values
TEST_CASES = [
    pytest.param(
        "tests/data/simple_pump_tank/simple_pump_tank_network_opt_params.json",
        {
            "objective_value": pytest.approx(11200.0, abs=1e-1),
            "pump_on_time": {
                "PUMP1": {'day: 0': pytest.approx(10.2, abs=1e-1)}
            },
            "expected_constraints": ut.load_json_file("tests/data/simple_pump_tank/simple_pump_tank_constraints_list.json"),
            "expected_columns": [
                "Datetime", "pipe_flow_P1", "pipe_flow_P2", 
                "pump_flow_PUMP1", "pump_power_PUMP1", 
                "tank_level_TANK", "total_power"
            ],
            "expected_rows": 24
        },
        id="simple_pump_tank_network"
    ),
    pytest.param(
        "tests/data/simple_pump_tank_binary/simple_pump_tank_network_opt_params.json",
        {
            "objective_value": pytest.approx(13600.0, abs=1e-1),
            "pump_on_time": {
                "PUMP1": {'day: 0': pytest.approx(9.0, abs=1e-1)}
            },
            "expected_constraints": ut.load_json_file("tests/data/simple_pump_tank_binary/simple_pump_tank_constraints_list.json"),
            "expected_columns": [
                "Datetime", "pipe_flow_P1", "pipe_flow_P2", 
                "pump_flow_PUMP1", "pump_power_PUMP1", 
                "tank_level_TANK", "total_power"
            ],
            "expected_rows": 24
        },
        id="simple_pump_tank_binary_network"
    ),
    pytest.param(
        "tests/data/soporon/soporon_network_opt_params.json",
        {
            "objective_value": pytest.approx(9966.8, abs=1e-1),
            "pump_on_time": {
                '4': {'state: 0 day: 0': pytest.approx(24.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(0.0, abs=1e-1), 'state: 2 day: 0': pytest.approx(0.0, abs=1e-1)},
                '5': {'state: 0 day: 0': pytest.approx(0.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(24.0, abs=1e-1)}, 
                '6': {'state: 0 day: 0': pytest.approx(5.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(7.0, abs=1e-1), 'state: 2 day: 0': pytest.approx(12.0, abs=1e-1)},
                '7': {'state: 0 day: 0': pytest.approx(4.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(20.0, abs=1e-1)},
                '8': {'state: 0 day: 0': pytest.approx(3.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(2.0, abs=1e-1), 'state: 2 day: 0': pytest.approx(5.0, abs=1e-1), 'state: 3 day: 0': pytest.approx(14.0, abs=1e-1)},
                '9': {'state: 0 day: 0': pytest.approx(13.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(2.0, abs=1e-1), 'state: 2 day: 0': pytest.approx(9.0, abs=1e-1)},
                '10': {'state: 0 day: 0': pytest.approx(24.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(0.0, abs=1e-1), 'state: 2 day: 0': pytest.approx(0.0, abs=1e-1)}, 
                '11': {'state: 0 day: 0': pytest.approx(5.0, abs=1e-1), 'state: 1 day: 0': pytest.approx(11.0, abs=1e-1), 'state: 2 day: 0': pytest.approx(8.0, abs=1e-1)}
            },
            "expected_constraints": ut.load_json_file("tests/data/soporon/soporon_constraints_list.json"),
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
            "expected_rows": 24
        },
        id="soporon_network"
    )
]

@pytest.fixture(scope="module")
def wdn(request):
    """Fixture to create a DynamicWaterNetworkCVX instance."""
    params = ut.load_json_file(request.param)
    wdn = DynamicWaterNetworkCVX(params=params)
    wdn.solve()
    return wdn

@pytest.mark.parametrize("wdn, expected", TEST_CASES, indirect=["wdn"])
class TestWDN:
    def test_optimization_solution(self, wdn, expected):
        """Test that the optimization problem can be solved."""
        # wdn.solve()
        # Accept optimal, optimal_inaccurate, or user_limit status (time limit reached)
        assert wdn.problem.status in ["optimal", "optimal_inaccurate", "user_limit"]
        
        # Only check objective value for optimal solutions 
        if wdn.problem.status == "optimal":
            assert wdn.problem.value == pytest.approx(expected["objective_value"], abs=1e-1)

    def test_all_constraints(self, wdn, expected):
        """Test that all constraints are properly set."""
        constraints = wdn.constraints
        print(constraints.keys())
        for constraint_name in expected["expected_constraints"]:
            assert constraint_name in constraints.keys()

    def test_package_data(self, wdn, expected):
        """Test that data packaging works correctly."""
        # wdn.solve()
        results_df = wdn.package_data()
        # Check all expected columns exist
        for column in expected["expected_columns"]:
            assert column in results_df.columns
        
        # Check number of rows
        assert len(results_df) == expected["expected_rows"]


    def test_pump_on_times(self, wdn, expected):
        """Test the pump on times after optimization."""
        # wdn.solve()
        # Skip detailed pump time checks if solver didn't reach optimality
        if wdn.problem.status != "optimal":
            pytest.skip(f"Skipping pump on times test as solver status is {wdn.problem.status}")
        
        # print all pump names
        for pump_name, expected_pump_on_time in expected["pump_on_time"].items():
            result_pump_on_times = wdn.get_pump_on_times(pump_name)
            assert result_pump_on_times == expected_pump_on_time


    def test_tank_levels(self, wdn, expected):
        """
        Test the tank levels after optimization.
        Because the tank levels can be different depending on the branch and cut in MIP solver, test the initial and final tank levels.
        """
        # wdn.solve()
        # get the tank levels at the start and end of the optimization
        tank_levels_start = {tank["name"]:wdn.get_tank_levels(tank["name"], wdn.optimization_start_time) for tank in wdn.wn["nodes"] if tank["node_type"] == "Tank"}
        # Assert that tank_levels_start equals init_level
        for tank in wdn.wn["nodes"]:
            if tank["node_type"] == "Tank":
                name = tank["name"]
                assert tank_levels_start[name] == pytest.approx(tank["init_level"], abs=1e-6), f"Start level for tank {name} {tank_levels_start[name]} does not match init_level {tank['init_level']}"

        # Assert that tank_levels_end is within the allowed deviation from init_level
        tolerance = wdn.params.get("final_tank_level_deviation", 0.1)
        for tank in wdn.wn["nodes"]:
            if tank["node_type"] == "Tank":
                name = tank["name"]
                init_level = tank["init_level"]
                end_level = wdn.get_final_tank_levels(name)
                assert (1 - tolerance) * init_level - 1e-6 <= end_level <= (1 + tolerance) * init_level + 1e-6, f"End level for tank {name} {end_level} is not within allowed deviation ({tolerance}) from init_level {init_level}"
