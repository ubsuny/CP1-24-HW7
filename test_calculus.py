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

def test_constant_function(self):
    """Integral of f(x) = 1 from 0 to 1 is 1"""
    result = calc.simpsons_rule(lambda x: 1, 0, 1, 10)
    self.assertAlmostEqual(result, 1, places=5)

def test_linear_function(self):
    """Integral of f(x) = x from 0 to 1 is 0.5"""
    result = calc.simpsons_rule(lambda x: x, 0, 1, 10)
    self.assertAlmostEqual(result, 0.5, places=5)

def test_quadratic_function(self):
    """Integral of f(x) = x^2 from 0 to 1 is 1/3"""
    result = calc.simpsons_rule(lambda x: x**2, 0, 1, 10)
    self.assertAlmostEqual(result, 1/3, places=5)

def test_sine_function(self):
    """Integral of f(x) = sin(x) from 0 to pi is 2"""
    result = calc.simpsons_rule(math.sin, 0, math.pi, 100)
    self.assertAlmostEqual(result, 2, places=5)

def test_invalid_subintervals(self):
    """Testing invalid subintervals"""
    with self.assertRaises(ValueError):
        calc.simpsons_rule(lambda x: x, 0, 1, 3)

def test_negative_subintervals(self):
    """Testing negative subintervals"""
    with self.assertRaises(ValueError):
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
