"""
This module implements different integration and root finding algorithms
"""

import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt

def dummy():
    """ 
    dummy function for template file
    """
    return 0

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
