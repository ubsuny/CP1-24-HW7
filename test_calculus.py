"""
test_calculus.py
"""

import numpy as np

from calculus import scipy_simpson
# Define the function to integrate outside the test function

def test_scipy_simpson():
    """
    test the scipy implementation for simpson method
    """
    def test_func(x):
    return x**3 + 1  # Function to integrate
    
    # Expected result (integral of x^3 + 1 from -1 to 1 is 2)
    expected_result = 2.0
    result = scipy_simpson(test_func, a, b)
    print(f"Result: {result}")

    # Use an assertion to check if the result is close to the expected value
    assert np.isclose(result, expected_result, atol=1e-6), \
        f"Expected {expected_result}, but got {result}"
