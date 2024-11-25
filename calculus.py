"""
calculus.py
This module implements different integration and root finding algorithms
"""
import ctypes
import math
import numpy as np
from scipy import optimize
import scipy as sp
from scipy.integrate import simpson

# General function to integrate
def wrapper_simpson(f, a, b, n=100):
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
    # Ensure the number of intervals is even
    if n % 2 == 1:
        n += 1  # Make n even by adding 1 if it's odd

    # Generate the x values (evenly spaced points) between a and b
    x = np.linspace(a, b, n+1)

    # Evaluate the function at the x points
    y = f(x)

    # Apply Simpson's rule
    result = simpson(y, x=x)

    return result

# import matplotlib.pyplot as plt

def dummy():
    """ 
    dummy function for template file
    """
    return 0

def simpsons_rule(func, a, b, n):
    """
    Approximate the integral of `func` from `a` to `b` using Simpson's Rule.

    Parameters:
        func (callable): The function to integrate.
        a (float): The start point of the interval.
        b (float): The end point of the interval.
        n (int): The number of subintervals (must be even).

    Returns:
        float: The approximate integral of the function.
    """
    if n % 2 != 0:
        raise ValueError("The number of subintervals `n` must be even.")
    if n <= 0:
        raise ValueError("The number of subintervals `n` must be positive.")

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [func(xi) for xi in x]

    integral = y[0] + y[-1]
    integral += 4 * sum(y[i] for i in range(1, n, 2))
    integral += 2 * sum(y[i] for i in range(2, n - 1, 2))
    integral *= h / 3
    return integral

# Function that uses the tangent method for root-finding
def root_tangent(function, fprime, x0):
    """
    A function that takes a function, its derivative, and an initial guess
    to estimate the root closest to that initial guess

    Parameters:
    Inputs:
    function (function): a function that defines a mathematically specific functional form
    fprime (function): a function that defines the mathematically functional form of the
                        function's derivative
    x0: an initial guess that's as close as possible to one of the roots
    Outputs:
    (number): the desired root (zero) of the function
    """
    return optimize.newton(function, x0, fprime)

def tangent_pure_python(func, fprime, x0, tol=1e-6, maxiter=50):
    """
    Pure Python implementation of the Newton-Raphson (tangent) method
    for root finding.

    Parameters:
    func : function
        The function for which the root is to be found.
    fprime : function
        The derivative of the function.
    x0 : float
        Initial guess for the root.
    tol : float, optional
        The convergence tolerance (default is 1e-6).
    maxiter : int, optional
        The maximum number of iterations (default is 50).

    Returns:
    dict
        A dictionary containing:
        - 'root': The estimated root if convergence is achieved.
        - 'converged': Boolean indicating whether the method converged.
        - 'iterations': The number of iterations performed.
    """
    for i in range(maxiter):
        # Evaluate function and its derivative at the current guess
        f_val = func(x0)
        fprime_val = fprime(x0)
        # Handle division by zero
        if abs(fprime_val) < 1e-12:
            return {
                "root": None,
                "converged": False,
                "iterations": i,
                "message": "Derivative too close to zero, division by zero encountered."
            }
        # Update the guess using the Newton-Raphson formula
        x_next = x0 - f_val / fprime_val
        # Check for convergence
        if abs(x_next - x0) < tol:
            return {
                "root": x_next,
                "converged": True,
                "iterations": i + 1
            }
        # Update the current guess
        x0 = x_next
    # If max iterations are reached without convergence
    return {
        "root": None,
        "converged": False,
        "iterations": maxiter,
        "message": "Maximum iterations reached without convergence."}

def a_trap(y, d):
    """
    trap takes in y as an array of y values
    and d, the separation between each y-value
    to use the numpy trapezoid function to 
    return the integral
    """
    return np.trapezoid(y,dx=d)



def adapt(func, bounds, d, sens):
    """
    adapt uses adaptive trapezoidal integration
    to integrate a function over boundaries.
    func must be function which outputs a list
    of its values.
    bounds is a list of length two which defines
    the lower and upper bounds of integration
    d defines the number of points between
    lower and upper bound to use with integration
    sens must be a number; it defines the 
    sensitivity the adapt function will have to 
    how the function changes. 
    """
    #The x is defined as a linspace which is used to define
    #the second derivative of the function for each point x
    #between the bounds
    x=np.linspace(bounds[0], bounds[1], d+1)
    dx=x[1]-x[0]
    dydx=0
    if func=="x^3+1":
        dydx=d3(x)
    elif func=="exp(-1/x)":
        dydx=d1(x)
    elif func=="cos(1/x)":
        dydx=d2(x)

    loopx=enumerate(x)
    summer=0
    #a loop is run through x. Each second derivative is used to
    #define the number of indices of new_x, which is a
    #list defining a number of points inbetween 2 x values
    #then, trapezoidal integration is conducted over new_x
    #and each integration is summed with eachother to produce
    #the total integral.
    for count, val in loopx:
        if count!=len(x)-1:
            new_x=np.linspace(val, x[count+1], 2*(int(np.abs(sens*dydx[count]))+1))
            new_y=[]
            if func=="x^3+1":
                new_y=func3(new_x)
            elif func=="exp(-1/x)":
                new_y=func1(new_x)
            elif func=="cos(1/x)":
                new_y=func2(new_x)
            summer+=a_trap(new_y, dx/((2*(int(np.abs(sens*dydx[count]))+1))-1))
    return summer

def trapezoid_numpy(func, l_lim, u_lim, steps=10000):
    '''
    This function implements trapezoidal rule using numpy wrapper function
    by evaluating the integral of the input function over the limits given
    in the input number of steps and gives as output the integral value in numpy 
    floating point decimal. If the input function is infinite at any point that point
    is modified to a slightly higher value.

    Parameters:
    - func: integrand (could be a custom defined function or a standard function like np.sin)
    - l_lim: lower limit of integration
    - u_lim: upper limit of integration
    - steps: number of steps (default value 10000)

    Returns:
    - integral of the input function using numpy.trapezoid function

    Once the integral value is calculated, it can be compared graphically as follows:

    integral_function = np.zeros(len(x))    # a zero array for integral at each point

    for i in range(1, len(x)):
        # calculate the integral at each x for plotting
        integral_function[i] = np.trapezoid(y[:i+1], x[:i+1])

    # plotting the original and integrated functions
    plt.plot(x, y, label = '$f(x)$')
    # plt.plot(x[:-1], integral_function, label = '$\\int_a^b f(x) dx$')
    plt.plot(x, integral_function,
             label = 'shaded area under $y = f(x)$: $\\int f(t) dt = $'+str(integral_value))
    plt.fill_between(x, y, 0)
    plt.ylabel('y(x)')
    plt.xlabel('x')
    plt.title('integral using numpy trapezoidal method; steps = '+str(steps))
    plt.legend()
    '''

    # check if the integrand is infinite at lower limit, if yes slightly change the limit
    try:
        func(l_lim)
    except ZeroDivisionError:
        l_lim += 0.000000001

    # check if the integrand is infinite at upper limit, if yes slightly change the limit
    try:
        func(u_lim)
    except ZeroDivisionError:
        u_lim += 0.000000001

    x = np.linspace(l_lim, u_lim, steps+1)  # create a linear grid between upper and lower limit

    for i, xi in enumerate((x)):
        # check if the integrand is infinite at any x, if yes slightly change the x value
        try:
            func(xi)
        except ZeroDivisionError:
            x[i] += 0.000000001

    y = func(x) # evaluate the function on the modified grid
    integral_value = np.trapezoid(y, x) # calculate the integral using numpy
    return integral_value

def trapezoid_scipy(func, l_lim, u_lim, steps=10000):
    '''
    This function implements trapezoidal rule using scipy wrapper function
    by evaluating the integral of the input function over the limits given
    in the input number of steps and gives as output the integral value in numpy 
    floating point decimal. If the input function is infinite at any point that point
    is modified to a slightly higher value.

    Parameters:
    - func: integrand(could be a custom defined function or a standard function like np.sin)
    - l_lim: lower limit of integration
    - u_lim: upper limit of integration
    - steps: number of steps (default value 10000)

    Returns:
    - integral of the input function using scipy.integrate.trapezoid function

    Once the integral value is calculated, it can be compared graphically as follows:

       integral_function = np.zeros(len(x))    # a zero array for integral at each point

    for i in range(1, len(x)):
        # calculate the integral at each x for plotting
        integral_function[i] = sp.integrate.trapezoid(y[:i+1], x[:i+1])

    # plotting the original and integrated functions
    plt.plot(x, y, label = '$f(x)$')
    plt.fill_between(x, y, 0)
    # plt.plot(x[:-1], integral_function, label = '$\\int_a^b f(x) dx$')
    plt.plot(x, integral_function,
             label = 'shaded area under $y = f(x)$: $\\int f(t) dt = $'+str(integral_value))
    plt.title('integral using scipy trapezoidal method; steps = '+str(steps))
    plt.ylabel('y(x)')
    plt.xlabel('x')
    plt.legend()
    '''

    # check if the integrand is infinite at lower limit, if yes slightly change the limit
    try:
        func(l_lim)
    except ZeroDivisionError:
        l_lim += 0.000000001

    # check if the integrand is infinite at upper limit, if yes slightly change the limit
    try:
        func(u_lim)
    except ZeroDivisionError:
        u_lim += 0.000000001

    x = np.linspace(l_lim, u_lim, steps+1)  # create a linear grid between upper and lower limit

    for i, xi in enumerate((x)):
        # check if the integrand is infinite at any x, if yes slightly change the x value
        try:
            func(xi)
        except ZeroDivisionError:
            x[i] += 0.000000001

    y = func(x)    # evaluate the function on the modified grid
    integral_value = sp.integrate.trapezoid(y, x)   # calculate the integral using numpy
    return integral_value
