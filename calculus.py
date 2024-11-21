"""
calculus.py
this module contain function  to calculate the integral using the simpson
"""

import numpy as np

def dummy():
    """ dummy functions for template file
    just the same function that Dr Thomay made"""
    return 0

def simpson(f, a, b, n):
    """
    Approximate the integral of the function f over the interval [a, b] using Simpson's rule.

    Parameters:
    f (function): The function to integrate.
    a (float): The start of the interval.
    b (float): The end of the interval.
    n (int): The number of intervals (must be even). Default is 100.

    Returns:
    float: The approximate integral of f over [a, b].
    """
    h = (b - a) / n
    i = np.arange(0, n)
    # Ensure n is even
    if n % 2 == 1:
        n += 1

    s = f(a) + f(b)
    s += 4 * np.sum(f(a + i[1::2] * h))
    s += 2 * np.sum(f(a + i[2:-1:2] * h))

    # Compute the integral and return both the result and the updated value of n
    integral = s * h / 3
    return integral, n
