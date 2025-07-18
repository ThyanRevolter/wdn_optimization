import pytest
import numpy as np
import pyomo.environ as pyo
from wdn_optimization.wdn_pyomo import DynamicWaterNetwork

# Define test cases with their expected values
TEST_CASES = [
    pytest.param(
        "tests/data/simple_pump_tank/simple_pump_tank_network_opt_params.json",
        {
            "objective_value": 11200.0,  # Same expected value as CVXPY version
        },
        id="simple_pump_tank_network"
    )
]

@pytest.fixture
def wdn(request):
    """Fixture to create a DynamicWaterNetwork instance."""
    params_path = request.param
    return DynamicWaterNetwork(params_path=params_path)

@pytest.mark.parametrize("wdn,expected", TEST_CASES, indirect=["wdn"])
def test_optimization_solve(wdn, expected):
    """Test that the optimization problem can be solved and objective value matches expected."""
    results = wdn.solve()
    assert results is not None
    assert results.solver.status == "ok"
    assert abs(pyo.value(wdn.model.objective) - expected["objective_value"]) < 1e-6 