"""
Unit testing module for testing functions in calculus.py
"""
import ctypes
import os
import math
import pytest
import numpy as np
import calculus as calc

# Ctypes initialization routine
# Load the shared library
lib_path = os.path.abspath("calculus.dll")
calculus = ctypes.CDLL(lib_path)
# Define argument and return types for the DLL functions
calculus.verify_arguments.argtypes = [ctypes.c_double]
calculus.verify_arguments.restype = ctypes.c_bool

calculus.calculate_square.argtypes = [ctypes.c_double]
calculus.calculate_square.restype = ctypes.c_double

# Define the function to integrate outside the test function

def test_wrapper_simpson():
    """
    test the scipy implementation for simpson method
    """
    assert np.isclose(calc.wrapper_simpson(np.sin, 0, np.pi), 2)

def test_constant_function():
    """Integral of f(x) = 1 from 0 to 1 is 1"""
    result = calc.simpsons_rule(lambda x: 1, 0, 1, 10)
    assert math.isclose(result, 1, rel_tol=1e-5)


def test_linear_function():
    """Integral of f(x) = x from 0 to 1 is 0.5"""
    result = calc.simpsons_rule(lambda x: x, 0, 1, 10)
    assert math.isclose(result, 0.5, rel_tol=1e-5)


def test_quadratic_function():
    """Integral of f(x) = x^2 from 0 to 1 is 1/3"""
    result = calc.simpsons_rule(lambda x: x**2, 0, 1, 10)
    assert math.isclose(result, 1 / 3, rel_tol=1e-5)


def test_sine_function():
    """Integral of f(x) = sin(x) from 0 to pi is 2"""
    result = calc.simpsons_rule(math.sin, 0, math.pi, 100)
    assert math.isclose(result, 2, rel_tol=1e-5)


def test_invalid_subintervals():
    """Testing invalid subintervals"""
    with pytest.raises(ValueError):
        calc.simpsons_rule(lambda x: x, 0, 1, 3)


def test_negative_subintervals():
    """Testing negative subintervals"""
    with pytest.raises(ValueError):
        calc.simpsons_rule(lambda x: x, 0, 1, -2)

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

def test_convergence_tangent():
    """
    Test the tangent_pure_python function for convergence to a positive root.

    This test validates that the function successfully converges to the 
    positive root (x=2) of the quadratic equation x^2 - 4, starting with an
    initial guess near the root (x0=3).
    """
    def func(x):
        return x**2 - 4  # Roots at x=2 and x=-2

    def fprime(x):
        return 2*x

    result = calc.tangent_pure_python(func, fprime, x0=3)
    assert result['converged'] is True
    assert pytest.approx(result['root'], abs=1e-6) == 2

def test_negative_root_tangent():
    """
    Test the tangent_pure_python function for convergence to a negative root.

    This test validates that the function successfully converges to the 
    negative root (x=-2) of the quadratic equation x^2 - 4, starting with an
    initial guess near the root (x0=-3).
    """
    def func(x):
        return x**2 - 4  # Roots at x=2 and x=-2

    def fprime(x):
        return 2*x

    result = calc.tangent_pure_python(func, fprime, x0=-3)
    assert result['converged'] is True
    assert pytest.approx(result['root'], abs=1e-6) == -2

def test_non_convergence_tangent():
    """
    Test the tangent_pure_python function for non-convergence behavior.

    This test ensures that the function correctly identifies when it fails to
    converge within the specified maximum number of iterations (maxiter=5). 
    The initial guess is far from the root.
    """
    def func(x):
        return x**2 - 4  # Roots at x=2 and x=-2

    def fprime(x):
        return 2*x

    result = calc.tangent_pure_python(func, fprime, x0=1000, maxiter=5)
    assert result['converged'] is False
    assert result['iterations'] == 5

def test_zero_div_tangent():
    """
    Test the tangent_pure_python function for division by zero handling.

    This test ensures that the function handles cases where the derivative
    is zero, returning an appropriate error message and no root.
    """
    def func(x):
        return x**3 - 6*x**2 + 11*x - 6  # Roots at x=1, 2, and 3

    def fprime(x):
        return 0*x  # Derivative is zero (forces division by zero)

    result = calc.tangent_pure_python(func, fprime, x0=1)
    assert result['converged'] is False
    assert result['root'] is None
    assert 'message' in result
    assert result['message'] == "Derivative too close to zero, division by zero encountered."

def test_tolerance_control_tangent():
    """
    Test the tangent_pure_python function with a tighter tolerance.

    This test validates that the function respects the specified tighter
    tolerance (1e-10) while successfully converging to the positive root (x=2).
    """
    def func(x):
        return x**2 - 4  # Roots at x=2 and x=-2

    def fprime(x):
        return 2*x

    result = calc.tangent_pure_python(func, fprime, x0=3, tol=1e-10)
    assert result['converged'] is True
    assert pytest.approx(result['root'], abs=1e-10) == 2

def test_zero_root_tangent():
    """
    Test the tangent_pure_python function for a root close to zero.

    This test ensures that the function converges to the root (x=0) of the 
    cubic function x^3, starting from an initial guess near zero (x0=0.1).
    The maximum number of iterations is increased to account for slower 
    convergence near the root.
    """
    def func(x):
        return x**3

    def fprime(x):
        return 3*x**2

    # Increase maxiter to ensure convergence for higher multiplicity roots
    result = calc.tangent_pure_python(func, fprime, x0=0.1, maxiter=100)
    assert result['converged'] is True, "Method did not converge"
    assert pytest.approx(result['root'], abs=1e-5) == 0, f"Expected 0, but got {result['root']}"

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


def d3(x):
    """Derivative of x^3 + 1."""
    return 3 * x**2

def d1(x):
    """Derivative of exp(-1/x)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.exp(-1 / x) / x**2

def d2(x):
    """Derivative of cos(1/x)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sin(1 / x) / x**2

@pytest.mark.parametrize("func, bounds, d, sens, expected", [
    ("x^3+1", [0, 1], 100, 1, 1.25),  # Integral of x^3 + 1 from 0 to 1
    ("exp(-1/x)", [1, 2], 100, 1, 0.5047),  # Approximation
    ("cos(1/x)", [0.1, 0.2], 100, 1, 0.0322),  # Approximation
])
def test_adapt(func, bounds, d, sens, expected):
    """
    Unit test for adaptive integration function

    Parameters:
    func (str): the function to integrate
    bounds (list): integration bounds [lower, upper]
    d (int): number of points
    sens (float): sensitivity of the adaptation
    """
    result = calc.adapt(func, bounds, d, sens)
    assert np.isclose(result, expected, atol=1e-2)

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

# ctypes and c++ unit tests
def test_trapezoid_python():
    '''
    Unit test for pure python implementation of trapezoid method
    '''
    assert np.isclose(calc.trapezoid_python(np.sin, 0, np.pi), 2)
    assert np.isclose(calc.trapezoid_python(exp_minus_one_by_x, 0, 1), 0.148496)

def test_library_loaded():
    """
    Test to ensure the shared library is loaded successfully.
    """
    assert calculus is not None, "Failed to load the shared library."

def test_verify_arguments():
    """
    Test the verify_arguments function.

    This function should return True for non-negative inputs and False for negative inputs.
    """
    # Test a valid, non-negative input
    assert calculus.verify_arguments(10.0), "verify_arguments(10.0) should return True."

    # Test an invalid, negative input
    assert not calculus.verify_arguments(-5.0), "verify_arguments(-5.0) should return False."

def test_calculate_square():
    """
    Test the calculate_square function.

    This function should compute the square of a valid input and return NAN for invalid inputs.
    """
    # Test a valid input
    result = calculus.calculate_square(4.0)
    assert result == 16.0, f"calculate_square(4.0) should return 16.0, got {result}."

    # Test an invalid input (negative number)
    result = calculus.calculate_square(-4.0)
    assert math.isnan(result), "calculate_square(-4.0) should return NAN for invalid input."
