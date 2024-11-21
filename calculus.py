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

import ctypes
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
                    the function encounters undefined values.
    """
    try:
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
        ValueError: If func(a) and func(b) do not have opposite signs.
    """
    if func(a) * func(b) >= 0:
        raise ValueError("The function must have opposite signs at a and b.")

    root = (a + b) / 2
    while (b - a) / 2 > tol:
        root = (a + b) / 2
        value_at_root = func(root)
        if value_at_root == 0:
            break  # Exact root found
        if func(a) * value_at_root < 0:
            b = root
        else:
            a = root

    return root

# Define C-compatible function pointers
CFUNC = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

# Load the shared library
lib = ctypes.CDLL('./bisection.so')

# Update ctypes function signature
lib.root_bisection.argtypes = [
    CFUNC,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_bool
]
lib.root_bisection.restype = ctypes.c_double

def bisection_ctypes(func, x0, x1, tol=1e-6, max_iter=1000):
    """
    C++ implementation of the bisection method using ctypes.
    """
    def c_safe_func(x):
        try:
            return func(x)
        except ValueError:
            return float('nan')  # Return NaN for invalid values

    c_func = CFUNC(c_safe_func)
    result = lib.root_bisection(
        c_func, ctypes.c_double(x0),
        ctypes.c_double(x1),
        ctypes.c_double(tol),
        ctypes.c_int(max_iter),
        ctypes.c_bool(False)
    )

    if math.isnan(result):
        raise ValueError("C++ root_bisection failed or encountered undefined values.")

    return result
