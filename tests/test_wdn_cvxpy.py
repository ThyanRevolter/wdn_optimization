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

@pytest.fixture
def wdn(request):
    """Fixture to create a DynamicWaterNetworkCVX instance."""
    params = ut.load_json_file(request.param)
    return DynamicWaterNetworkCVX(params=params)

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_optimization_solution(wdn, expected):
    """Test that the optimization problem can be solved."""
    result = wdn.solve()
    print(wdn.problem.status)
    print(wdn.problem.value)
    assert result is not None
    assert wdn.problem.status == "optimal"
    assert wdn.problem.value == pytest.approx(expected["objective_value"], abs=1e-1)

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_pump_on_times(wdn, expected):
    """Test the pump on times after optimization."""
    wdn.solve()
    # print all pump names
    for pump_name, expected_pump_on_time in expected["pump_on_time"].items():
        result_pump_on_times = wdn.get_pump_on_times(pump_name)
        assert result_pump_on_times == expected_pump_on_time

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_all_constraints(wdn, expected):
    """Test that all constraints are properly set."""
    constraints = wdn.constraints
    print(constraints.keys())
    for constraint_name in expected["expected_constraints"]:
        assert constraint_name in constraints.keys()

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_package_data(wdn, expected):
    """Test that data packaging works correctly."""
    wdn.solve()
    results_df = wdn.package_data()
    print(results_df.columns)
    # Check all expected columns exist
    for column in expected["expected_columns"]:
        assert column in results_df.columns
    
    # Check number of rows
    assert len(results_df) == expected["expected_rows"]
