"""
Unit testing module for testing functions in calculus.py
"""

import math
import pytest
import numpy as np
import calculus as calc
# Define the function to integrate outside the test function

def test_wrapper_simpson():
    """
    test the scipy implementation for simpson method
    """
    assert np.isclose(calc.wrapper_simpson(np.sin, 0, np.pi), 2)

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

def exp_minus_one_by_x(x):
    '''
    defining a function that raises divide by zero at x=0

    Parameters:
    - x: function argument

    Returns:
    - numerical value of exp(-1/x)
    '''
    return np.exp(-1/x)

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
    assert np.isclose(calc.trapezoid_numpy(exp_minus_one_by_x, 0, 1), 0.148496)

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

@pytest.mark.parametrize("f, a, b, n, expected", [
    (lambda x: x**2, 0, 1, 100, 1/3),
    (lambda x: x ** 2, 0, 1, 100, 1/3),
])
def test_trapezoid(f, a, b, n, expected):
    """Unit test for trapezoid pure python"""
    result = calc.trapezoid(f, a, b, n)
    assert abs(result - expected) < 1e-4, f"Failed for f={f}, a={a}, b={b}, n={n}"

@pytest.mark.parametrize("f, a, b, tol, expected", [
    (lambda x: x**2, 0, 1, 1e-6, 1/3),
    (lambda x: x**2, 0, 1, 1e-6, 1/3),
    (lambda x: 1/(1 + x**2), 0, 1, 1e-6, 3.141592653589793 / 4),
])
def test_adaptive_trap_py(f, a, b, tol, expected):
    """Unit test for adaptive trap pure python"""
    result = calc.adaptive_trap_py(f, a, b, tol)
    assert abs(result - expected) < 1e-6, f"Failed for f={f}, a={a}, b={b}, tol={tol}"

    assert np.isclose(calc.trapezoid_scipy(exp_minus_one_by_x, 0, 1), 0.148496)

def test_secant_pure_matches_scipy():
    '''
    Unit test to check if scipy and pure python implementation of
    secant root finding method yield equivalent results.
    '''

    def dummyfunc(x,a):
        return x-a
    wrap = calc.secant_wrapper(dummyfunc, x0=0, x1 = 4, args=(1,), maxiter = 50)
    pure = calc.secant_pure_python(dummyfunc, x0=0, x1 = 4, args=(1,), maxiter = 50)

    assert np.isclose(wrap['root'], pure['root'])

def test_secant_pure_gets_root():
    '''
    Unit test to check if pure python implementation of
    secant root finding method yields correct value.
    '''

    def dummyfunc(x,a):
        return x-a
    pure = calc.secant_pure_python(dummyfunc, x0=0, x1 = 4, args=(1,), maxiter = 50)

    assert np.isclose(pure['root'],1)

@pytest.mark.filterwarnings("ignore:Tolerance of.*:RuntimeWarning")
def test_secant_wrapper_doesnt_converge():
    '''
    Unit test to check if scipy secant root finder wrapper 
    returns no convergence when there is no root
    '''

    def quadratic(x,a,b,c):
        return a*x**2 + b*x + c
    assert calc.secant_wrapper(quadratic, x0=0, x1 = 1,
                               args=(1,0,1), maxiter = 50)['converged'] is False


def test_trapezoid_python():
    '''
    Unit test for pure python implementation of trapezoid method
    '''
    assert np.isclose(calc.trapezoid_python(np.sin, 0, np.pi), 2)
    assert np.isclose(calc.trapezoid_python(exp_minus_one_by_x, 0, 1), 0.148496)
