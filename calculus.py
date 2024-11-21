"""
calculus.py
this module contain implementation of scipy for simpson
"""
import numpy as np
from scipy.integrate import simpson

# General function to integrate
def scipy_simpson(f, a, b, n=100):
    """
    Integrate a function `f` over the interval [a, b] using Simpson's rule.
    
    Parameters:
        f (function): The function to integrate.
        a (float): The lower limit of the integration.
        b (float): The upper limit of the integration.
        n (int): Number of points to sample for Simpson's rule. Default is 100.
        
    Returns:
        float: The approximate integral value.
    """
    # Create an array of x values between a and b with n points
    x = np.linspace(a, b, n)
    
    # Evaluate the function at each x point
    y = f(x)
    
    # Use Simpson's rule to approximate the integral
    return simpson(y, x)
