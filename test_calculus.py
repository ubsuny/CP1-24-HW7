"""
test_calculus.py
"""

import numpy as np
import calculus as calc
from calculus import scipy_simpson
# Define the function to integrate outside the test function

def test_scipy_simpson():
    """
    test the scipy implementation for simpson method
    """
    # Use an assertion to check if the result is close to the expected value
    assert np.isclose(calc.scipy_simpson(x**3 + 1, -1, 1), 2)
