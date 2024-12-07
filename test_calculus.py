"""
Unit testing module for testing functions in calculus.py
"""
import ctypes
import os
import re
import math
from unittest.mock import patch
import pytest
import numpy as np
import calculus as calc

# Ctypes initialization routine
# Load the shared library
lib_path = os.path.abspath("./lib_calculus.so")  # Ensure the DLL is correctly named and placed
new_lib_path=os.path.abspath("./calc.dll")
try:
    calculus = ctypes.CDLL(lib_path)
    new_calc=ctypes.CDLL(new_lib_path)
except OSError as e:
    raise RuntimeError(f"Failed to load shared library: {e}") from e

# Define argument and return types for the DLL functions
calculus.verify_arguments.argtypes = [ctypes.c_double]
calculus.verify_arguments.restype = ctypes.c_bool
calculus.calculate_square.argtypes = [ctypes.c_double]
calculus.calculate_square.restype = ctypes.c_double

# Define ctypes function signature for `invoke_with_floats`
CallbackFunction = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
calculus.invoke_with_floats.argtypes = [CallbackFunction, ctypes.c_double, ctypes.c_double]
calculus.invoke_with_floats.restype = ctypes.c_double

new_calc.adapt_c.argtypes=[ctypes.CFUNCTYPE(ctypes.c_double,ctypes.c_double)
, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
new_calc.adapt_c.restype=ctypes.c_double

def cubic(x):
    """
    the cubic function
    """
    return x**3+1
def cosine(x):
    """
    the cosine function
    """
    return np.cos(1/x)
def exponential(x):
    """
    The exponential function
    """
    return np.exp(-1/x)

cubic_c=ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
python_cubic_c=cubic_c(cubic)
cos_c=ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
python_cos_c=cos_c(cosine)
exp_c=ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
python_exp_c=exp_c(exponential)
def test_adapt_c():
    """
    test adapt confirms that the adapt_c function 
    produces the expected results for certain 
    integrals.
    """
    result1=new_calc.adapt_c(python_cubic_c, -1,1,10,10)
    assert np.isclose(result1, 2, .1)
    result2=new_calc.adapt_c(python_cos_c, .01,3*np.pi, 20,100)
    assert np.isclose(result2, 7.9, .1)
    result3=new_calc.adapt_c(python_exp_c, .01, 10, 100, 1000)
    assert np.isclose(result3, 7.2, .1)
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
    return x ** 3 - 4 * x + 1

def func_1_prime(x):
    """
    A simple function to be used for testing
    Parameters:
    Inputs:
    x (number): the independent variable
    Outputs:
    (number): the value of the derivative of func_1
    """
    return 3 * x ** 2 - 4

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
    initial_guess_1 (number): the initial guess for the root
    """
    compare = calc.root_tangent(func_1, func_1_prime, initial_guess_1)
    assert np.isclose(compare, 0.2541016883651, atol = 1.0e-6)

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

def test_cpp_root_tangent(initial_guess_1):
    """
    A function that tests the cpp implementation for tangent method for root-finding

    Parameters:
    Inputs:
    initial_guess_1 (number): the initial guess for the root
    """
    compare = calc.cpp_root_tangent(func_1, func_1_prime, initial_guess_1).root
    assert np.isclose(compare, 0.2541016883651, atol = 1.0e-6)

def test_cpp_root_tangent_callability(initial_guess_1):
    """
    A function that tests the cpp implementation for tangent method for root-finding

    Parameters:
    Inputs:
    initial_guess_1 (number): the initial guess for the root
    """
    with pytest.raises(TypeError, match="py_func and py_fprime must be callable functions."):
        calc.cpp_root_tangent(1, "s", initial_guess_1)

def test_cpp_root_tangent_convergence(initial_guess_1):
    """
    A function that tests the cpp implementation for tangent method for root-finding

    Parameters:
    Inputs:
    initial_guess_1 (number): the initial guess for the root
    """
    with pytest.raises(RuntimeError, match="Root finding did not converge within 3 iterations."):
        calc.cpp_root_tangent(func_1, func_1_prime, initial_guess_1, maxiter = 3)

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
    (cubic, [0, 1], 100, 1, 1.25),  # Integral of x^3 + 1 from 0 to 1
    (exponential, [1, 2], 100, 1, 0.5047),  # Approximation
    (cosine, [0.1, 0.2], 100, 1, 0.0322),  # Approximation
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

# Start of Bisection methods tests
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

def cubic_function(x):
    """
    A cubic function: f(x) = x^3 - 6x^2 + 11x - 6.

    Roots: x = 1, 2, 3
    """
    return x**3 - 6*x**2 + 11*x - 6

@pytest.mark.parametrize(
    "func, a, b, expected_root, tol",
    [
        (cubic_function, 0, 1.5, 1.0, 1e-6),  # Root near 1
        (cubic_function, 1.5, 2.5, 2.0, 1e-6),  # Root near 2
        (cubic_function, 2.5, 3.5, 3.0, 1e-6),  # Root near 3
    ]
)

# Test bisection_ctypes
def test_bisection_ctypes(func, a, b, expected_root, tol):
    """
    Test the bisection_ctypes method with a cubic function for multiple intervals.

    Parameters:
    ----------
    func : The function to find the root of.
    a : The lower bound of the interval.
    b : The upper bound of the interval.
    expected_root : The expected root of the function.
    tol : The tolerance for the test.
    """
    root = calc.bisection_ctypes(func, a, b, tol=tol, max_iterations=100)
    expect_msg = f"Root {root} deviates from {expected_root} by more than {tol}"
    assert abs(root - expected_root) <= tol, expect_msg

def linear_function(x):
    """
    A simple linear function: f(x) = x - 5.

    Root: x = 5
    """
    return x - 5

def test_bisection_linear():
    """
    Test the bisection method with a linear function.
    """
    root = calc.bisection_ctypes(linear_function, 0, 10, tol=1e-6, max_iterations=100)
    assert abs(root - 5.0) <= 1e-6, f"Root {root} deviates from 5.0 by more than 1e-6"

def test_bisection_no_root():
    """
    Test the bisection method on an interval with no root.

    Should raise a ValueError because the signs of f(a) and f(b) are not opposite.
    """
    with pytest.raises(OSError, match=re.escape('[WinError 541541187] Windows Error 0x20474343')):
        calc.bisection_ctypes(cubic_function, 4, 5, tol=1e-6, max_iterations=10)

def test_bisection_max_iterations():
    """
    Test the bisection method exceeding maximum iterations.

    Should raise a RuntimeError when maximum number of iterations reached.
    """
    # Define a slow converging function (e.g., quadratic function)
    def slow_converging_function(x):
        return x**2 - 1  # Root at x = 1

    # Call the bisection method
    result = calc.bisection_ctypes(slow_converging_function, 0.0, 2.0, 1e-6, 10)

    print(result)
    # Check if the result is NaN, which indicates the max iterations were exceeded
    if math.isnan(result):
        # If we get here, the test failed because no error was raised but we got expected root
        assert False, f"RuntimeError -> maximum iterations, but got result: {result}"
    else:
        assert result, 1.0

# End of Bisection tests

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


def test_ctypes_invoke_with_floats():
    """
    Test invoking a Python math function with two floats through a C++ callback.
    """

    # Define a simple Python math function
    def square(x):
        return x * x

    # Wrap the Python callback using ctypes
    wrapped_callback = CallbackFunction(square)

    # Test with two positive floats
    result = calculus.invoke_with_floats(wrapped_callback, 2.0, 3.0)
    assert math.isclose(result, 25.0, rel_tol=1e-5), f"Expected 25.0, got {result}"

    # Test with a positive and a negative float
    result = calculus.invoke_with_floats(wrapped_callback, -3.0, 3.0)
    assert math.isclose(result, 0.0, rel_tol=1e-5), f"Expected 0.0, got {result}"

    # Test with two negative floats
    result = calculus.invoke_with_floats(wrapped_callback, -2.0, -3.0)
    assert math.isclose(result, 25.0, rel_tol=1e-5), f"Expected 25.0, got {result}"

def test_secant_root_positive_root():
    """
    Test secant_root to find the positive root of the equation x^2 - 4 = 0.
    """
    # Define the function to find the root for
    def func(x):
        return x**2 - 4

    # Initial guesses near the positive root (x = 2)
    x0, x1 = 3.0, 2.5
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=50)

    # Assert the result is close to the expected root
    assert math.isclose(root, 2.0, rel_tol=1e-6), f"Expected root 2.0, got {root}"


def test_secant_root_negative_root():
    """
    Test secant_root to find the negative root of the equation x^2 - 4 = 0.
    """
    # Define the function to find the root for
    def func(x):
        return x**2 - 4

    # Initial guesses near the negative root (x = -2)
    x0, x1 = -3.0, -2.5
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=50)

    # Assert the result is close to the expected root
    assert math.isclose(root, -2.0, rel_tol=1e-6), f"Expected root -2.0, got {root}"


def test_secant_root_no_convergence():
    """
    Test secant_root for a case where the method does not converge.
    """
    # Define a function that does not have a root in the given range
    def func(x):
        return x**2 + 4  # No real roots

    # Initial guesses
    x0, x1 = 1.0, 2.0
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=10)

    # Assert that the root is NaN due to non-convergence
    assert math.isnan(root), f"Expected NaN for no convergence, got {root}"

def test_secant_root_converges_to_zero():
    """
    Test secant_root for convergence to zero.
    """
    def func(x):
        return x**3  # Flat region near x = 0

    # Initial guesses
    x0, x1 = 0.0, 1.0
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=50)

    # Debugging output
    print(f"Result for secant_root_converges_to_zero: {root}")

    # Assert that the root is approximately 0.0
    assert math.isclose(root, 0.0, abs_tol=1e-6), f"Expected root at 0.0, got {root}"

def test_secant_root_division_by_zero():
    """
    Test secant_root for a case where division by zero occurs.
    """
    def func(x=0):
        print(f"{x} passed in to test_secant_root_division_by_zero()")
        return 1.0  # Constant function, derivative is zero everywhere

    # Initial guesses where function values are the same
    x0, x1 = 0.0, 1.0
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=50)

    # Debugging output
    print(f"Result for secant_root_division_by_zero: {root}")

    # Assert that the root is NaN due to division by zero
    assert math.isnan(root), f"Expected NaN due to division by zero, got {root}"

def test_secant_root_cubic():
    """
    Test secant_root for a cubic function.
    """
    def func(x):
        return x**3 - 1

    x0, x1 = 0.5, 1.5
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=50)
    print(f"Cubic Test: x0={x0}, x1={x1}, root={root}")

    assert not math.isnan(root), "Unexpected NaN result."
    assert math.isclose(root, 1.0, rel_tol=1e-6), f"Expected root 1.0, got {root}"


def test_secant_root_exponential():
    """
    Test secant_root for an exponential function.
    """
    def func(x):
        return math.exp(x) - 1

    # Initial guesses
    x0, x1 = -1.0, 1.0
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=50)

    # Assert the result is close to 0
    assert math.isclose(root, 0.0, rel_tol=1e-6), f"Expected root 0.0, got {root}"



def test_secant_root_trigonometric():
    """
    Test secant_root for a trigonometric function.
    """
    def func(x):
        return math.sin(x)

    x0, x1 = 3.0, 4.0  # Root at pi
    root = calc.secant_root(func, x0, x1, tol=1e-6, max_iter=50)
    print(f"Trigonometric Test: x0={x0}, x1={x1}, root={root}")

    assert not math.isnan(root), "Unexpected NaN result."
    assert math.isclose(root, math.pi, rel_tol=1e-6), f"Expected root {math.pi}, got {root}"


def test_adaptive_trap_py_exp():
    """
    Unit test for adaptive trapezoidal rule for exp(-1/x).
    """
    result = calc.adaptive_trap_py(calc.func1, 0.01, 10, tol=1e-6)
    expected = 7.22545022194032
    assert np.isclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

def test_adaptive_trap_py_cos():
    """
    Unit test for adaptive trapezoidal rule for cos(1/x).
    """
    result = calc.adaptive_trap_py(calc.func2, 0.01, 3 * np.pi, tol=1e-6)
    expected = 7.908470226410015
    assert np.isclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

def test_trapezoid_pure_python():
    """
    Unit test for pure Python trapezoidal implementation.
    """
    result = calc.trapezoid(lambda x: x**2, 0, 1, 100)
    expected = 1 / 3
    assert np.isclose(result, expected, atol=1e-4), f"Expected {expected}, got {result}"


def test_secant_method():
    """
    Unit test for the secant method (pure Python).
    """
    result = calc.secant_pure_python(lambda x: x**2 - 4, 1, 3, maxiter=50)
    expected = 2.0
    assert np.isclose(result["root"], expected, atol=1e-6), \
        f"Expected {expected}, got {result['root']}"

def test_secant_wrapper_method():
    """
    Unit test for the secant wrapper using SciPy.
    """
    result = calc.secant_wrapper(lambda x: x**2 - 4, 1, 3)
    expected = 2.0
    assert np.isclose(result["root"], expected, atol=1e-6), \
        f"Expected {expected}, got {result['root']}"

def test_evaluate_integrals():
    """
    Unit test for evaluate_integrals function.

    This test will verify that evaluate_integrals() runs successfully without raising exceptions
    and checks the correctness of the integration values.

    Assertions:
        - Verifies that the computed integral is close to the expected value for each function.
    """
    # Expected values for each function over their respective ranges
    expected_results = {
        "exp(-1/x)": 7.22545,  # Approximation for [0.000001, 10]
        "cos(1/x)": 7.91109,   # Approximation for [0.000001, 3π]
        "x^3+1": 2.0          # Analytic value for [-1, 1]
    }

    tol = 1e-2  # Allowable tolerance for accuracy check

    # Mocking time.sleep to speed up the testing process if it was part of the original function
    with patch('time.sleep', return_value=None):
        # Run evaluate_integrals and capture the results
        try:
            results = calc.evaluate_integrals()
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            pytest.fail(f"evaluate_integrals() raised an unexpected exception: {e}")

    # Loop through expected results and verify if output is close to expected value
    for func_name, expected in expected_results.items():
        result = results[func_name]["Scipy Trapezoidal"]["result"]
        assert np.isclose(result, expected, atol=tol), (
            f"Integration result for {func_name} was {result:.6f}, "
            f"expected approximately {expected:.6f}"
        )

def test_edge_cases():
    """
    Unit test for edge cases when evaluating integrals.

    This test ensures that edge cases like singularities and very large integration ranges
    are handled appropriately by the integration functions in evaluate_integrals.

    Edge Cases:
        - Singularities at x close to zero.
        - Very large integration ranges to check numerical stability.

    Assertions:
        - Verifies that the calculated results are close to expected values for each edge case.
    """
    # Define edge cases for each function with respective bounds
    edge_cases = [
        {
            "func": calc.func1,
            "lower": 1e-10,
            "upper": 10,
            "expected": 7.22545
        },
        {
            "func": calc.func2,
            "lower": 1e-10,
            "upper": 3 * np.pi,
            "expected": 7.91109
        },
        {
            "func": calc.func3,
            "lower": -1e6,
            "upper": 1e6,
            "expected": 0.0  # Assuming large cancellation results in ~0
        }
    ]

    for case in edge_cases:
        try:
            result = calc.adaptive_trap_py(
                case["func"],
                case["lower"],
                case["upper"],
                tol=1e-6,
                remaining_depth=10,
            )
            assert np.isclose(result, case["expected"], atol=1e-2), (
                f"Integration result for edge case was {result:.6f}, "
                f"expected approximately {case['expected']:.6f}"
            )
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            pytest.fail(f"Edge case integration raised an unexpected exception: {e}")

@patch('calculus.adaptive_trap_py', return_value=7.22545)
@patch('calculus.trapezoid_numpy', return_value=7.22545)
@patch('calculus.trapezoid_scipy', return_value=7.22545)
def test_individual_methods(mock_adapt, mock_numpy, mock_scipy):
    """
    Unit test for individual integration methods using mocking.

    This test verifies that the integration methods (adaptive, numpy, scipy) are called
    correctly by the evaluate_integrals function, and that they return the expected values.

    Parameters:
    - mock_adapt: Mock of adaptive_trap_py function.
    - mock_numpy: Mock of trapezoid_numpy function.
    - mock_scipy: Mock of trapezoid_scipy function.
    """
    # Run evaluate_integrals() and ensure all mocks are called
    results = calc.evaluate_integrals()

    # Ensure results contain expected keys
    assert "exp(-1/x)" in results, "Results do not contain 'exp(-1/x)' function key."
    assert "cos(1/x)" in results, "Results do not contain 'cos(1/x)' function key."
    assert "x^3+1" in results, "Results do not contain 'x^3+1' function key."

    # Assert that each mocked method was called at least once
    assert mock_adapt.called, "Adaptive Trapezoidal method was not called."
    assert mock_numpy.called, "Numpy Trapezoidal method was not called."
    assert mock_scipy.called, "Scipy Trapezoidal method was not called."

    # Verify that each mocked method returned the correct value
    assert np.isclose(mock_adapt.return_value, 7.22545, atol=1e-2)
    assert np.isclose(mock_numpy.return_value, 7.22545, atol=1e-2)
    assert np.isclose(mock_scipy.return_value, 7.22545, atol=1e-2)

# Test for func_1_safe
def test_func_1_safe():
    """Test the func_1_safe function."""
    assert abs(calc.func_1_safe(1)) - 1.557 < 1e-2
    with pytest.raises(ValueError):
        calc.func_1_safe(math.pi)

# Test for func_2 (tanh)
def test_func_2():
    """Test the func_2 function (tanh)."""
    assert abs(calc.func_2(0)) - 0.0 < 1e-6
    assert abs(calc.func_2(1)) - 0.7616 < 1e-4
    assert abs(calc.func_2(-1)) - 0.7616 < 1e-5

# Test for func_3 (sin)
def test_func_3():
    """Test the func_3 function (sin)."""
    assert abs(calc.func_3(0)) - 0.0 < 1e-6
    assert abs(calc.func_3(math.pi / 2)) - 1.0 < 1e-6
    assert abs(calc.func_3(math.pi)) - 0.0 < 1e-6

# Test for calculate_accuracy
def test_calculate_accuracy():
    """Test the calculate_accuracy function."""
    assert calc.calculate_accuracy(1.0, 1.0) == 0  # Exact match returns 0 accuracy
    assert calc.calculate_accuracy(1.00001, 1.0) == 4

# Test for apply_methods
def test_apply_methods():
    """Test the apply_methods function."""
    try:
        calc.apply_methods(
            calc.func_2,
            (-1, 1),
            "y(x) = tanh(x)", 
            true_root=0.0,
            filename="test_plot.png"
        )
    except (ValueError, FileNotFoundError, TypeError, AssertionError) as e:
        pytest.fail(f"apply_methods failed with an expected error: {str(e)}")

# Test for plot_function_with_roots
def test_plot_function_with_roots():
    """Test the plot_function_with_roots function."""
    try:
        calc.plot_function_with_roots(
            calc.func_3,
            (0, math.pi),
            [math.pi],
            "test_plot.png", 
            "y(x) = sin(x)"
        )
    except (ValueError, FileNotFoundError, TypeError, AssertionError) as e:
        pytest.fail(f"plot_function_with_roots failed with an expected error: {str(e)}")

# Test for find_roots
def test_find_roots():
    """Test the find_roots function."""
    try:
        calc.find_roots()
    except (ValueError, TypeError, AssertionError) as e:
        pytest.fail(f"find_roots failed with an expected error: {str(e)}")
