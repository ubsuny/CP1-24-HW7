"""
calculus.py
This module implements different integration and root finding algorithms
"""
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


def a_trap(y, d):
    """
    trap takes in y as an array of y values
    and d, the separation between each y-value
    to use the numpy trapezoid function to 
    return the integral
    """
    return np.trapezoid(y,dx=d)

def func3(x):
    """
    func3 acts as the 
    x^3+1 function
    """
    return x**3+1

def d3(x):
    """
    d3 is the second derivative of 
    function func3
    """
    return 6*x

def func1(x):
    """
    func1 acts as the 
    exp(-1/x) function
    """
    return np.exp(-1/x)

def d1(x):
    """
    d1 is the second derivative of 
    function func1
    """
    return (1/x**2)*np.exp(-1/x)

def func2(x):
    """
    func2 acts as the 
    cos(1/x) function
    """
    return np.cos(1/x)

def d2(x):
    """
    d2 is the second derivative of 
    function func2
    """
    return (-1/x**2)*np.cos(1/x)

def adapt(func, bounds, d, sens):
    """
    adapt uses adaptive trapezoidal integration
    to integrate a function over boundaries.
    func must be a str defining the function
    to be integrated. May be:
    x^3+1
    exp(-1/x)
    cos(1/x)
    bounds is a list of length two which defines
    the lower and upper bounds of integration
    d defines the number of points between
    lower and upper bound to use with integration
    sens must be a number; it defines the 
    sensitivity the adapt function will have to 
    how the function changes. 
    """
    #The x is defined as a linspace which is used to define
    #the derivative of the function for each point x
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

def secant_wrapper(func, x0, x1, args=(), maxiter=50):
    """
    Wrapper for the secant method using scipy.optimize.root_scalar.

    Parameters:
    func : The function for which the root is to be found.
    x0 : The first initial guess (Ideally close to, but less than, the root)
    x1 : The second initial guess (Ideally close to, but greater than, the root)
    args : Additional arguments to pass to the function. Must be a tuple
    tol : Optional tolerance deciding when to stop function (default is 1e-6).
    maxiter : The maximum number of iterations if convergence is not met (default is 50).

    Returns:
    A dictionary containing:
        - 'root': The estimated root.
        - 'converged': Boolean indicating whether the method converged.
        - 'iterations': Number of iterations performed.
        - 'function_calls': Number of function evaluations performed.
    """

    #Use the secant method
    res = optimize.root_scalar(
        func,
        args=args,
        method="secant",
        x0=x0,
        x1=x1,
        xtol=1e-6,
        maxiter=maxiter)

    #Return a callable dictionary
    return {
        "root": res.root if res.converged else None,
        "converged": res.converged,
        "iterations": res.iterations,
        "function_calls": res.function_calls}
