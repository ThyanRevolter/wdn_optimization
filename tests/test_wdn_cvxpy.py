import pytest
import numpy as np
from wdn_optimization.wdn_cvxpy import DynamicWaterNetworkCVX

# Define test cases with their expected values
TEST_CASES = [
    pytest.param(
        "data/simple_pump_tank_network_opt_params.json",
        {
            "objective_value": 11200.0,
            "pump_on_time": 10.285714285714285,
            "expected_constraints": {
                "tank_level": ["tank_level_init_TANK", "tank_level_min_TANK", "tank_level_max_TANK"],
                "nodal_flow": ["nodal_flow_balance_equality_constraint_J1", "nodal_flow_balance_equality_constraint_J2"],
                "pump_flow": ["pump_flow_constraint_on_max_PUMP1", "pump_flow_constraint_on_min_PUMP1", "pump_flow_equality_constraint_PUMP1", "pump_power_constraint_PUMP1"],
                "pump_on_time": ["pump_on_time_constraint_PUMP1_0"],
                "total_power": ["total_power_equality_constraint"]
            },
            "expected_columns": [
                "Datetime", "pipe_flow_P1", "pipe_flow_P2", 
                "pump_flow_PUMP1", "pump_power_PUMP1", 
                "tank_level_TANK", "total_power"
            ],
            "expected_rows": 24
        },
        id="simple_pump_tank_network"
    )
]

@pytest.fixture
def wdn(request):
    """Fixture to create a DynamicWaterNetworkCVX instance."""
    params_path = request.param
    return DynamicWaterNetworkCVX(params_path=params_path)

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_optimization_solve(wdn, expected):
    """Test that the optimization problem can be solved."""
    result = wdn.solve()
    assert result is not None
    assert wdn.problem.status == "optimal"
    assert abs(wdn.problem.value - expected["objective_value"]) < 1e-6

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_pump_on_times(wdn, expected):
    """Test the pump on times after optimization."""
    wdn.solve()
    pump_on_times = wdn.get_pump_on_times("PUMP1")
    assert "day: 0" in pump_on_times
    assert abs(pump_on_times["day: 0"] - expected["pump_on_time"]) < 1e-6

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_tank_level_constraints(wdn, expected):
    """Test that tank level constraints are properly set."""
    constraints = wdn.get_tank_level_constraints()
    for constraint_name in expected["expected_constraints"]["tank_level"]:
        assert constraint_name in constraints

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_nodal_flow_balance_constraints(wdn, expected):
    """Test that nodal flow balance constraints are properly set."""
    constraints = wdn.get_nodal_flow_balance_constraints()
    for constraint_name in expected["expected_constraints"]["nodal_flow"]:
        assert constraint_name in constraints

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_pump_flow_constraints(wdn, expected):
    """Test that pump flow constraints are properly set."""
    constraints = wdn.get_pump_flow_constraints()
    for constraint_name in expected["expected_constraints"]["pump_flow"]:
        assert constraint_name in constraints

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_pump_on_time_constraint(wdn, expected):
    """Test that pump on time constraints are properly set."""
    constraints = wdn.get_pump_on_time_constraint()
    for constraint_name in expected["expected_constraints"]["pump_on_time"]:
        assert constraint_name in constraints

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_total_power_constraint(wdn, expected):
    """Test that total power constraint is properly set."""
    constraints = wdn.get_total_power_constraint()
    for constraint_name in expected["expected_constraints"]["total_power"]:
        assert constraint_name in constraints

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_package_data(wdn, expected):
    """Test that data packaging works correctly."""
    wdn.solve()
    results_df = wdn.package_data()
    
    # Check all expected columns exist
    for column in expected["expected_columns"]:
        assert column in results_df.columns
    
    # Check number of rows
    assert len(results_df) == expected["expected_rows"]
