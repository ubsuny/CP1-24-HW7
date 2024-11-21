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

    return s * h / 3


from scipy.integrate import simpson

def exp_minus_1_over_x(x):
    """f: e^(-1/x)"""
    return np.exp(-1/x)

def cos_1_over_x(x):
    """f: cos(1/x)"""
    return np.cos(1/x)

def x3_plus_1(x):
    """f: x^3 + 1"""
    return x**3 + 1

# Example usage:
a = -1  # Start of the interval
b = 1   # End of the interval
n = 100  # Number of intervals (ensure this is odd for Simpson's Rule)

# Create an array of x values from a to b
x = np.linspace(a, b, n+1)

# Use scipy.integrate.simpson to compute the integral of x^3 + 1
result = simpson(x3_plus_1(x), x)

print(f"The integral of x^3 + 1 over [{a}, {b}] is approximately {result:.5f}")
