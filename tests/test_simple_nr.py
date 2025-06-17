import pytest
import numpy as np
from wdn_optimization.simple_nr import WaterNetwork, Units

# Test data for chapter 5.3.4 example
chapter534_test_cases = [
    pytest.param(
        np.array([5.0, 2.5, 2.5, 0.5]),
        np.array([35.0, 31.0, 31.0]),
        np.array([4.5, 2.24, 2.26, 0.24]),
        np.array([34.845, 30.68, 30.613]),
        id="positive_initial_flow",
    ),
    pytest.param(
        np.array([-5.0, -2.5, -2.5, -0.5]),
        np.array([35.0, 31.0, 31.0]),
        np.array([4.5, 2.24, 2.26, 0.24]),
        np.array([34.845, 30.68, 30.613]),
        id="negative_initial_flow",
    ),
    pytest.param(
        np.array([5.0, -2.5, 2.5, -0.5]),
        np.array([35.0, 31.0, 31.0]),
        np.array([4.5, 2.24, 2.26, 0.24]),
        np.array([34.845, 30.68, 30.613]),
        id="mixed_initial_flow",
    ),
    pytest.param(
        np.array([-4.0, -2.0, -2.0, -0.0]),
        np.array([40.0, 35.0, 30.0]),
        np.array([4.5, 2.24, 2.26, 0.24]),
        np.array([34.845, 30.68, 30.613]),
        id="different_initial_head",
    ),
    pytest.param(
        np.array([10.0, 5.0, 5.0, 1.0]),
        np.array([50.0, 40.0, 40.0]),
        np.array([4.5, 2.24, 2.26, 0.24]),
        np.array([34.845, 30.68, 30.613]),
        id="large_initial_values",
    ),
]

# Test data for chapter 5.4 example
chapter54_test_cases = [
    pytest.param(
        np.array([20.0, 9.0, 11.0, 6.0, 5.5, 3.5, 0.5, 0.5, 1.0, 1.0, 8.0]),
        np.array([198.0, 193.0, 195.0, 175.0, 188.0, 190.0, 184.0]),
        np.array(
            [
                21.269,
                9.844,
                11.426,
                6.029,
                6.066,
                3.388,
                0.359,
                1.066,
                0.185,
                0.545,
                6.731,
            ]
        ),
        np.array([198.32, 193.861, 197.508, 170.577, 185.891, 192.443, 184.144]),
        id="positive_initial_flow",
    ),
    pytest.param(
        np.array([-20.0, -9.0, -11.0, -6.0, -5.5, -3.5, -0.5, -0.5, -1.0, -1.0, -8.0]),
        np.array([198.0, 193.0, 195.0, 175.0, 188.0, 190.0, 184.0]),
        np.array(
            [
                21.269,
                9.844,
                11.426,
                6.029,
                6.066,
                3.388,
                0.359,
                1.066,
                0.185,
                0.545,
                6.731,
            ]
        ),
        np.array([198.32, 193.861, 197.508, 170.577, 185.891, 192.443, 184.144]),
        id="negative_initial_flow",
    ),
    pytest.param(
        np.array([20.0, -9.0, 11.0, -6.0, 5.5, -3.5, 0.5, -0.5, 1.0, -1.0, 8.0]),
        np.array([198.0, 193.0, 195.0, 175.0, 188.0, 190.0, 184.0]),
        np.array(
            [
                21.269,
                9.844,
                11.426,
                6.029,
                6.066,
                3.388,
                0.359,
                1.066,
                0.185,
                0.545,
                6.731,
            ]
        ),
        np.array([198.32, 193.861, 197.508, 170.577, 185.891, 192.443, 184.144]),
        id="mixed_initial_flow",
    ),
    pytest.param(
        np.array([20.0, 9.0, 11.0, 6.0, 5.5, 3.5, 0.5, 0.5, 1.0, 1.0, 8.0]),
        np.array([200.0, 195.0, 197.0, 177.0, 190.0, 192.0, 186.0]),
        np.array(
            [
                21.269,
                9.844,
                11.426,
                6.029,
                6.066,
                3.388,
                0.359,
                1.066,
                0.185,
                0.545,
                6.731,
            ]
        ),
        np.array([198.32, 193.861, 197.508, 170.577, 185.891, 192.443, 184.144]),
        id="different_initial_head",
    ),
    pytest.param(
        np.array([30.0, 15.0, 15.0, 10.0, 10.0, 5.0, 1.0, 1.0, 2.0, 2.0, 12.0]),
        np.array([210.0, 205.0, 207.0, 187.0, 200.0, 202.0, 196.0]),
        np.array(
            [
                21.269,
                9.844,
                11.426,
                6.029,
                6.066,
                3.388,
                0.359,
                1.066,
                0.185,
                0.545,
                6.731,
            ]
        ),
        np.array([198.32, 193.861, 197.508, 170.577, 185.891, 192.443, 184.144]),
        id="large_initial_values",
    ),
]


@pytest.fixture
def wn_chapter534():
    return WaterNetwork(
        "data/epanet_networks/chapter_5_3_4_example.inp",
        units=Units.IMPERIAL_CFS,
        round_to=5,
    )


@pytest.fixture
def wn_chapter54():
    return WaterNetwork(
        "data/epanet_networks/chapter_5_4_example.inp",
        units=Units.IMPERIAL_CFS,
        round_to=5,
    )


@pytest.mark.parametrize(
    "initial_flow,initial_head,expected_flow,expected_head", chapter534_test_cases
)
def test_chapter534(
    wn_chapter534, initial_flow, initial_head, expected_flow, expected_head
):
    """Test cases for chapter 5.3.4 example"""
    final_flow, final_head = wn_chapter534.run_newton_raphson(
        initial_flow, initial_head
    )
    np.testing.assert_array_almost_equal(final_flow, expected_flow, decimal=2)
    np.testing.assert_array_almost_equal(final_head, expected_head, decimal=2)


@pytest.mark.parametrize(
    "initial_flow,initial_head,expected_flow,expected_head", chapter54_test_cases
)
def test_chapter54(
    wn_chapter54, initial_flow, initial_head, expected_flow, expected_head
):
    """Test cases for chapter 5.4 example"""
    final_flow, final_head = wn_chapter54.run_newton_raphson(initial_flow, initial_head)
    np.testing.assert_array_almost_equal(final_flow, expected_flow, decimal=2)
    np.testing.assert_array_almost_equal(final_head, expected_head, decimal=2)
