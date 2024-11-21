"""
test_calculus.py
"""

import numpy as np

from calculus import scipy_simpson
# Define the function to integrate outside the test function
def test_func(x):
    return x**3 + 1  # Function to integrate

def test_scipy_simpson():
    """
    test the scipy implementation for simpson method
    """

    # Debugging output
    a, b = -1, 1  # Define a and b here
    x = np.linspace(a, b, 100)  # Or any n you choose
    y = test_func(x) # Now test_func is accessible
    print(f"x values: {x}")
    print(f"y values: {y}")

    result = scipy_simpson(test_func, a, b)
    print(f"Result: {result}")
    # Use an assertion to check if the result is close to the expected value
    assert np.isclose(result, expected_result, atol=1e-6), \
        f"Expected {expected_result}, but got {result}"
