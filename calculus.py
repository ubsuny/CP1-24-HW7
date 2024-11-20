"""
This module implements different integration and root finding algorithms
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def dummy():
    """ 
    dummy function for template file
    """
    return 0

def trapezoid_numpy(func, l_lim, u_lim, steps=10000):
    '''
    Function to implement trapezoidal rule using numpy and plot the result
    '''
    x = np.linspace(l_lim, u_lim, steps+1)  # create a linear grid between upper and lower limit
    y = func(x) # evaluate the function on the grid
    integral_value = np.trapezoid(y, x) # calculate the integral using numpy

    integral_function = np.zeros(len(x))    # a zero array for integral at each point

    for i in range(1, len(x)):
        # calculate the integral at each x for plotting
        integral_function[i] = np.trapezoid(y[:i+1], x[:i+1])

    #dx = np.diff(x)
    #integral_function = np.cumsum(dx * (y[:-1] + y[1:]) / 2)

    # plotting the original and integrated functions
    plt.plot(x, y, label = '$f(x)$')
    # plt.plot(x[:-1], integral_function, label = '$\\int_a^b f(x) dx$')
    plt.plot(x, integral_function, label = '$\\int f(t) dt$: area under $y = f(x)$')
    plt.ylabel('y(x)')
    plt.xlabel('x')
    plt.legend()

    return integral_value

def trapezoid_scipy(f, a, b, n=10000):
    '''
    Function to implement trapezoidal rule using scipy and plot the result
    '''
    x = np.linspace(a, b, n+1)  # create a linear grid between upper and lower limit
    y = f(x)    # evaluate the function on the grid
    integral_value = sp.integrate.trapezoid(y, x)   # calculate the integral using numpy

    integral_function = np.zeros(len(x))    # a zero array for integral at each point

    for i in range(1, len(x)):
        # calculate the integral at each x for plotting
        integral_function[i] = sp.integrate.trapezoid(y[:i+1], x[:i+1])

    # dx = np.diff(x)
    # integral_function = np.cumsum(dx * (y[:-1] + y[1:]) / 2)

    # plotting the original and integrated functions
    plt.plot(x, y, label = '$f(x)$')
    # plt.plot(x[:-1], integral_function, label = '$\\int_a^b f(x) dx$')
    plt.plot(x, integral_function, label = '$\\int f(t) dt$: area under $y = f(x)$')
    plt.ylabel('y(x)')
    plt.xlabel('x')
    plt.legend()

    return integral_value
