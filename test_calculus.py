"""
Unit testing module for testing functions in calculus.py
"""

import pytest
import numpy as np
import calculus as calc

def test_simpson():
    """
    Unit test simpson method
    """
    # Call the simpson function and unpack the result
    result, updated_n = calc.simpson(np.sin, 0, np.pi, 100)

    # Use an assertion to check if the result is close to the expected value
    assert np.isclose(result, 2), f"Expected 2, but got {result}. Updated n: {updated_n}"

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

    def fprime():
        return 0  # Derivative is zero (forces division by zero)

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
    assert pytest.approx(result['root'], abs=1e-6) == 0, f"Expected root 0, but got {result['root']}"

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
