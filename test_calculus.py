"""
Unit testing module for testing functions in calculus.py
"""

import math
import pytest
import numpy as np
import calculus as calc

def func_1(x):
    """
    A simple function to be used for testing

    Parameters:
    Inputs:
    x (number): the independent variable
    Outputs:
    (number): the value of the function
    """
    return x ** 2

def func_1_prime(x):
    """
    A simple function to be used for testing

    Parameters:
    Inputs:
    x (number): the independent variable
    Outputs:
    (number): the value of the derivative of func_1
    """
    return 2 * x

@pytest.fixture(name = "initial_guess_1")
def func_1_x_0():
    """
    A function that returns a value for x as a first guess to be used for testing
    """
    return 0.0073

def test_dummy():
    """ 
    Unit test for dummy function
    """
    assert calc.dummy() == 0

def test_root_tangent(initial_guess_1):
    """
    A function that tests the wrapper implementation for tangent method for root-finding

    Parameters:
    Inputs:
    func_1 (function): the function to find its root
    func_1_prime (function): the derivative of the function
    x_0 (number): the initial guess for the root
    """
    compare = calc.root_tangent(func_1, func_1_prime, initial_guess_1)
    assert np.isclose(compare, 0.0, atol = 1.0e-6)

def test_trapezoid_numpy():
    '''
    Unit test for numpy implementation of trapezoid method
    '''
    assert np.isclose(calc.trapezoid_numpy(np.sin, 0, np.pi), 2)

def test_trapezoid_scipy():
    '''
    Unit test for scipy implementation of trapezoid method
    '''
    assert np.isclose(calc.trapezoid_scipy(np.sin, 0, np.pi), 2)

# Test data for various cases
test_data_tanh = [
    (math.tanh, -1, 1, 1e-6, 0.0)  # (function, a, b, tol, expected_root)
]

test_data_1_over_sin = [
    (lambda x: 1 / math.sin(x) if math.sin(x) != 0 else float('inf'), 3, 4, 1e-6, math.pi)
]

invalid_interval_data = [
    (math.tanh, -1, -0.5, 1e-6)  # (function, a, b, tol)
]

# Test data: singularities near sin(x) = 0
singularity_data = [
    (lambda x: 1 / math.sin(x) if math.sin(x) != 0 else float('inf'),
     3.141592653589793 - 1e-3, 3.141592653589793 + 1e-3, 1e-6),  # near singularity
    (lambda x: 1 / math.sin(x), 3.1405926535897932, 3.142592653589793,
     1e-6), # Check behavior near zero
]

# SciPy Wrapper Tests
@pytest.mark.parametrize("func, a, b, tol, expected", test_data_tanh + test_data_1_over_sin)
def test_bisection_wrapper(func, a, b, tol, expected):
    """Test SciPy wrapper implementation."""
    root = calc.bisection_wrapper(func, a, b, tol)
    assert math.isclose(root, expected, rel_tol=1e-6), f"Expected {expected}, got {root}"

# Pure Python Implementation Tests
@pytest.mark.parametrize("func, a, b, tol, expected", test_data_tanh + test_data_1_over_sin)
def test_bisection_pure_python(func, a, b, tol, expected):
    """Test pure Python implementation."""
    root = calc.bisection_pure_python(func, a, b, tol)
    assert math.isclose(root, expected, rel_tol=1e-6), f"Expected {expected}, got {root}"

# Invalid Interval Tests
@pytest.mark.parametrize("func, a, b, tol", invalid_interval_data)
def test_invalid_interval_for_bisection(func, a, b, tol):
    """Test for invalid intervals where func(a) and func(b) do not have opposite signs."""
    with pytest.raises(ValueError):
        calc.bisection_wrapper(func, a, b, tol)

    with pytest.raises(ValueError):
        calc.bisection_pure_python(func, a, b, tol)

# Singularity Tests
@pytest.mark.parametrize("func, a, b, tol", singularity_data)
def test_singularities_for_bisection(func, a, b, tol):
    """Test handling of singularities for 1/sin(x)."""
    with pytest.raises(ValueError):
        calc.bisection_wrapper(func, a, b, tol)

    with pytest.raises(ValueError):
        calc.bisection_pure_python(func, a, b, tol)
