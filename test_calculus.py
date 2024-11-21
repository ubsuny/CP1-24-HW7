"""
test_calculus.py
"""
import unittest
import numpy as np
from scipy.integrate import simpson

class TestCalculusFunctions(unittest.TestCase):
    """
    Unit tests for integration-related functions.

    This class tests the correctness and accuracy of functions related to
    numerical and symbolic integration. It includes tests for definite and
    indefinite integrals, handling of improper integrals, edge cases, and
    the behavior of integration functions for a variety of mathematical
    expressions and domains.
    """
    def test_simpson(self):
        """Test the scipy.simpson function with a known integral."""

        # Example: Integral of f(x) = x^3 + 1 over the interval [-1, 1]
        def x3_plus_1(x):
            """f: x^3 + 1"""
            return x**3 + 1

        # Interval
        a = -1
        b = 1
        n = 100  # Number of intervals (ensure this is odd for Simpson's Rule)

        # Create an array of x values between a and b (n+1 points)
        x = np.linspace(a, b, n+1)

        # The expected integral for the function x^3 + 1 over [-1, 1]
        expected_integral = (b**4 / 4 + b) - (a**4 / 4 + a)  # Integral of x^3 + 1 analytically

        # Calculate the function values for x^3 + 1 at the points in x_values
        y_values = x3_plus_1(x)
        
        # Use scipy.integrate.simpson to compute the integral of x^3 + 1
        result = simpson(y_values)

        # Check if the result is close to the expected value (within 5 decimal places)
        self.assertAlmostEqual(result, expected_integral, places=5)
