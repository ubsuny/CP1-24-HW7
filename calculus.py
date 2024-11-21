"""
calculus.py module

This module provides implementations of the bisection root-finding algorithm 
in three ways: using a SciPy wrapper, a pure Python implementation, 
and a C/C++ implementation via ctypes.

Functions:
- bisection_wrapper: A wrapper for SciPy's `bisect` method.
- bisection_pure_python: A pure Python implementation of the bisection method.
- 
"""

import math
from scipy.optimize import bisect

def dummy():
    """ dummy functions for template file
    just the same function that Dr Thomay made"""
    return 0

def bisection_wrapper(func, a, b, tol=1e-6, max_iter=1000):
    """
    Wrapper for SciPy's `bisect` function.

    Parameters:
        func (callable): The function for which to find the root.
        a (float): The start of the interval.
        b (float): The end of the interval.
        tol (float, optional): The tolerance level for convergence. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

    Returns:
        float: The approximate root of the function.

    Raises:
        ValueError: If func(a) and func(b) do not have opposite signs or if
                    the function encounters undefined values (singularities).
    """
    small_value_threshold = 1e-3  # Threshold for detecting singularities in sin(x)

    try:
        # Check if sin(a) or sin(b) are very small (near zero)
        if abs(math.sin(a)) < small_value_threshold:
            raise ValueError(f"Singularity detected: division by zero in function at x = {a}.")
        if abs(math.sin(b)) < small_value_threshold:
            raise ValueError(f"Singularity detected: division by zero in function at x = {b}.")

        # Call the SciPy bisect method if no errors were raised
        root = bisect(func, a, b, xtol=tol, maxiter=max_iter)

    except ValueError as e:
        raise ValueError(f"SciPy bisect failed: {e}") from e

    return root


def bisection_pure_python(func, a, b, tol=1e-6):
    """
    Pure Python implementation of the bisection method.
    Finds the root of func within the interval [a, b].

    Parameters:
        func (callable): The function for which to find the root.
        a (float): The start of the interval.
        b (float): The end of the interval.
        tol (float, optional): The tolerance level for convergence. Defaults to 1e-6.

    Returns:
        float: The approximate root of the function.

    Raises:
        ValueError: If func(a) and func(b) do not have opposite signs or if the function
                    encounters singularities (undefined values at the interval endpoints).
    """
    # Check if the function values at a and b are of opposite signs
    if func(a) * func(b) >= 0:
        raise ValueError("The function must have opposite signs at a and b.")

    # Check for singularity or undefined values in the function at the endpoints
    if abs(math.sin(a)) < 1e-12 or abs(math.sin(b)) < 1e-12:  # Stricter threshold for singularity
        raise ValueError(f"Singularity detected: sin(a) = {math.sin(a)}, sin(b) = {math.sin(b)}")

    root = (a + b) / 2
    print(f"Initial root estimate: {root}")  # Debugging: Initial root estimate

    while (b - a) / 2 > tol:
        root = (a + b) / 2
        value_at_root = func(root)

        # If the function value at root is 0, return the root as an exact solution
        if value_at_root == 0:
            break

        # If func(root) is too large, it indicates a singularity
        if abs(value_at_root) > 1e10:  # Set a threshold for large values
            raise ValueError(f"Singularity detected: func(root) = {value_at_root}")

        # Narrow the interval
        if func(a) * value_at_root < 0:
            b = root
        else:
            a = root

    return root
