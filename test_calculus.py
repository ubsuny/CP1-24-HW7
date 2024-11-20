"""
test_calculus.py
"""
import calculus as calc
import unittest # Import the unittest module
import numpy as np
import sympson from calculus

class TestCalculusFunctions(unittest.TestCase):
    """
    Unit tests for integration-related functions.

    This class tests the correctness and accuracy of functions related to
    numerical and symbolic integration. It includes tests for definite and
    indefinite integrals, handling of improper integrals, edge cases, and
    the behavior of integration functions for a variety of mathematical
    expressions and domains.
    """
    def test_simpson(self): # Indent this function definition
        """Test the simpson function with a known integral."""


    # Indent the following functions one level further to be part of the class
    def exp_minus_1_over_x(self, x): # Added 'self' as argument for class methods
        """f: e^(-1/x)"""
        return np.exp(-1/x)

    def cos_1_over_x(self, x): # Added 'self' as argument for class methods
        """f: cos(1/x)"""
        return np.cos(1/x)

    def x3_plus_1(self, x): # Added 'self' as argument for class methods
        """f: x^3 + 1"""
        return x**3 + 1

        # Initial guesses and expected outcome for Function: x^3 + 1
        a = -1
        b = 1
        n = 100
        expected_integral = 2

        # Call the simpson function
        result = simpson(f, a, b, n)

        # Check if the result is close to the expected integral
        self.assertAlmostEqual(result, expected_integral, places=5)
        
def test_dummy():
    """ Unit test for dummy function
    just the same test function that Dr Thomay made"""
    assert calc.dummy() == 0
