"""
test_calculus.py
"""

import numpy as np

from calculus import scipy_simpson

def test_scipy_simpson():
    """
    test the scipy implementation for simpson method
    """
    # Define the range for integration
    a, b = -1, 1

    # Compute the integral using Simpson's rule
    result = scipy_simpson(scipy_simpson, a, b)

    # Expected result (integral of x^3 + 1 from -1 to 1 is 2)
    expected_result = 2.0

    # Use an assertion to check if the result is close to the expected value
    assert np.isclose(result, expected_result, atol=1e-6), \
        f"Expected {expected_result}, but got {result}"
