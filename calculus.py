"""
calculus.py
this module contain implementation of scipy for simpson
"""
import numpy as np
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
A = -1  # Start of the interval
B = 1   # End of the interval
N = 100  # Number of intervals

# Create an array of x values from A to B
x_va;ues = np.linspace(A, B, N+1)

# Use scipy.integrate.simpson to compute the integral of x^3 + 1
result = simpson(x3_plus_1(x_values), x_values)

print(f"The integral of x^3 + 1 over [{a}, {b}] is approximately {result:.5f}")
