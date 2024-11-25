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
    by evaluating the integral of the input function over the limits given, and
    in the input number of steps and gives as output the integral value in numpy 
    floating point decimal. If the input function is infinite at any point that point
    is modified to a slightly different value.

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

    # check if the integrand is infinite at lower limit, if yes slightly increase the limit
    try:
        func(l_lim)
    except ZeroDivisionError:
        l_lim += 0.000000001

    # check if the integrand is infinite at upper limit, if yes slightly decrease the limit
    try:
        func(u_lim)
    except ZeroDivisionError:
        u_lim -= 0.000000001

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
    by evaluating the integral of the input function over the limits given, and
    in the input number of steps and gives as output the integral value in numpy 
    floating point decimal. If the input function is infinite at any point that point
    is modified to a slightly different value.

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

    # check if the integrand is infinite at lower limit, if yes slightly increase the limit
    try:
        func(l_lim)
    except ZeroDivisionError:
        l_lim += 0.000000001

    # check if the integrand is infinite at upper limit, if yes slightly decrease the limit
    try:
        func(u_lim)
    except ZeroDivisionError:
        u_lim -= 0.000000001

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

def trapezoid(f, a, b, n):
    """
    Compute the trapezoidal approximation of an integral.
    Parameters:
    f (callable): Function to integrate.
    a (float): Lower bound of integration.
    b (float): Upper bound of integration.
    n (int): Number of subdivisions.
    """
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]
    return (h / 2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])


def adaptive_trap_py(f, a, b, tol, remaining_depth=10):
    """
    Compute an integral using the adaptive trapezoid method.
    Parameters:
    f (callable): Function to integrate.
    a (float): Lower bound of integration.
    b (float): Upper bound of integration.
    tol (float): Tolerance for stopping condition.
    remaining_depth (int): Remaining recursion depth to avoid infinite recursion.
    """
    integral1 = trapezoid(f, a, b, n=1)

    integral2 = trapezoid(f, a, b, n=2)

    if abs(integral2 - integral1) < tol or remaining_depth <= 0:
        return integral2

    mid = (a + b) / 2
    left_integral = adaptive_trap_py(f, a, mid, tol / 2, remaining_depth - 1)
    right_integral = adaptive_trap_py(f, mid, b, tol / 2, remaining_depth - 1)

    return left_integral + right_integral

def trapezoid_python(func, l_lim, u_lim, steps=10000):
    '''
    This function implements trapezoidal rule by a pure python implementation
    by evaluating the integral of the input function over the limits given, and
    in the input number of steps and gives as output the integral value in numpy 
    floating point decimal. If the input function is infinite at any point that point
    is modified to a slightly different value.

    Parameters:
    - func: integrand (could be a custom defined function or a standard function like np.sin)
    - l_lim: lower limit of integration
    - u_lim: upper limit of integration
    - steps: number of steps (default value 10000)

    Returns:
    - integral of the input function using numpy.trapezoid function

    The plotting functionality can be incorporated just like in trapezoid_numpy()
    or trapezoid_scipy()
    '''

    # check if the integrand is infinite at lower limit, if yes slightly increase the limit
    try:
        func(l_lim)
    except ZeroDivisionError:
        l_lim += 0.000000001

    # check if the integrand is infinite at upper limit, if yes slightly decrease the limit
    try:
        func(u_lim)
    except ZeroDivisionError:
        u_lim -= 0.000000001

    x = np.linspace(l_lim, u_lim, steps+1)  # create a linear grid between upper and lower limit

    for i, xi in enumerate((x)):
        # check if the integrand is infinite at any x, if yes slightly change the x value
        try:
            func(xi)
        except ZeroDivisionError:
            x[i] += 0.000000001

    y = func(x) # evaluate the function on the modified grid
    h = (u_lim - l_lim)/steps   # step size
    # calculate the integral using the trapezoidal algorithm
    integral_value = (h/2)*(y[0] + y[-1] + 2 * np.sum(y[1:-1]))
    return integral_value

def secant_wrapper(func, x0, x1, args=(), maxiter=50):
    """
    Wrapper for the secant method using scipy.optimize.root_scalar.

    Parameters:
    func : The function for which the root is to be found.
    x0 : The first initial guess (Ideally close to, but less than, the root)
    x1 : The second initial guess (Ideally close to, but greater than, the root)
    args : Additional arguments to pass to the function. Must be a tuple
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

# Root Finding with Bisection Method
def bisection_wrapper(func, a, b, tol=1e-6, max_iter=1000):
    """
    Wrapper for SciPy's `bisect` function.

    Parameters:
        func: The function for which to find the root.
        a: The start of the interval.
        b: The end of the interval.
        tol: The tolerance level for convergence. Defaults to 1e-6.
        max_iter: Maximum number of iterations. Defaults to 1000.

    Returns:
        Root: The approximate root of the function.

    Raises:
        ValueError: If func(a) and func(b) do not have opposite signs or if
                    the function encounters undefined values (singularities).
    """
    small_value_threshold = 1e-3  # Threshold for detecting singularities in sin(x)

    try:
        # Check if sin(a) or sin(b) are very small (near zero)
        if abs(math.sin(a)) < small_value_threshold:
            raise ValueError(f"Singularity detected: division by zero in function at x = {a}.")
        if abs(math.sin(b)) < small_value_threshold:
            raise ValueError(f"Singularity detected: division by zero in function at x = {b}.")

        # Call the SciPy bisect method if no errors were raised
        root = optimize.bisect(func, a, b, xtol=tol, maxiter=max_iter)

    except ValueError as e:
        raise ValueError(f"SciPy bisect failed: {e}") from e

    return root

def bisection_pure_python(func, a, b, tol=1e-6, max_iter=1000):
    """
    Pure Python implementation of the bisection method.
    Finds the root of func within the interval [a, b].

    Parameters:
        func: The function for which to find the root.
        a: The start of the interval.
        b: The end of the interval.
        tol: The tolerance level for convergence. Defaults to 1e-6.
        max_iter: Maximum number of iterations. Default is 1000.

    Returns:
        Root: The approximate root of the function.

    Raises:
        ValueError: If no root is detected in the initial interval.
        RuntimeError: If the method exceeds the maximum number of iterations.
    """

    # Check if the initial interval is valid
    if func(a) * func(b) >= 0:
        raise ValueError("The function must have opposite signs at a and b.")

    # Check for singularity or undefined values in the function at the endpoints
    if abs(math.sin(a)) < 1e-12 or abs(math.sin(b)) < 1e-12:  # Stricter threshold for singularity
        raise ValueError(f"Singularity detected: sin(a) = {math.sin(a)}, sin(b) = {math.sin(b)}")

    root = (a + b) / 2
    iteration_count = 0  # Track the number of iterations

    while (b - a) / 2 > tol:
        # Check for maximum iterations
        if iteration_count >= max_iter:
            raise RuntimeError(f"Bisection method exceeded maximum iterations ({max_iter}).")

        iteration_count += 1
        root = (a + b) / 2
        value_at_root = func(root)

        # If the function value at root is 0, return the root as an exact solution
        if value_at_root == 0:
            break

        # If func(root) is too large, it indicates a singularity
        if abs(value_at_root) > 1e10:  # Set a threshold for large values
            raise ValueError(f"Singularity detected: func(root) = {value_at_root}")

        # Narrow the interval
        if func(a) * value_at_root < 0:
            b = root
        else:
            a = root

    return root

def secant_pure_python(func, x0, x1, args=(), maxiter=50):
    """
    Pure Python method for the secant method of root finding.

    Parameters:
    func : The function for which the root is to be found.
    x0 : The first initial guess (Ideally close to, but less than, the root)
    x1 : The second initial guess (Ideally close to, but greater than, the root)
    args : Additional arguments to pass to the function. Must be a tuple
    maxiter : The maximum number of iterations if convergence is not met (default is 50).

    Returns:
    A dictionary containing:
        - 'root': The estimated root.
        - 'converged': Boolean indicating whether the method converged.
        - 'iterations': Number of iterations performed.
    """

    for i in range(maxiter):
        #Evaluate function values
        f0 = func(x0, *args)
        f1 = func(x1, *args)
        #Prevent division by zero. If division by zero occurs, return this dict
        # If no division by zero, move to next dict
        if abs(f1 - f0) < 1e-12:
            return {
                "root": None,
                "converged": False,
                "iterations": i}
        #Secant method formula
        x_next = x1 - f1 * (x1 - x0) / (f1 - f0)
        #Check convergence. If converged return this dict
        #  Add one to iterations until iteration = maxiter
        if abs(x_next - x1) < 1e-6:
            return {
                "root": x_next,
                "converged": True,
                "iterations": i + 1}
        #Update guesses for next iteration
        x0, x1 = x1, x_next
    #If max iterations are reached without convergence return this dict
    return {
        "root": None,
        "converged": False,
        "iterations": maxiter}

def ctypes_stub():
    """    
    This method demonstrates the usage of a ctypes wrapper to interact with a C++
    shared library (DLL).

    The shared library (calculus.dll) currently provides two simple example functions:
    1. verify_arguments: Validates whether a given number is non-negative.
    2. calculate_square: Computes the square of a number if it is non-negative, returning
    NAN for invalid input.
    """
    # Load the DLL
    dll = ctypes.CDLL("./lib_calculus.so")


    # Define function signatures
    dll.verify_arguments.argtypes = [ctypes.c_double]
    dll.verify_arguments.restype = ctypes.c_bool

    dll.calculate_square.argtypes = [ctypes.c_double]
    dll.calculate_square.restype = ctypes.c_double

    # Test the functions
    try:
        # Test valid input
        result = dll.calculate_square(4.0)
        print(f"Square of 4.0: {result}")  # Should print 16.0

        # Test invalid input
        result = dll.calculate_square(-4.0)
        if math.isnan(result):
            print("Square of -4.0: Invalid input (returned NAN)")
        else:
            print(f"Square of -4.0: {result}")

        # Verify arguments
        print(f"verify_arguments(4.0): {dll.verify_arguments(4.0)}")  # True
        print(f"verify_arguments(-4.0): {dll.verify_arguments(-4.0)}")  # False

    except OSError as e:
        # Specific exception for issues loading the DLL or accessing its symbols
        print(f"OS error: {e}")

    except ValueError as e:
        # Exception for issues with invalid inputs or conversions
        print(f"Value error: {e}")

    except TypeError as e:
        # Exception for type mismatch errors
        print(f"Type error: {e}")

def ctypes_invoke_with_floats(callback, a, b):
    """
    Calls the C++ function invoke_with_floats, passing a Python callback function
    and two float arguments.

    Parameters:
        callback (function): A Python function that takes one float argument and returns a float.
        a (float): The first float input.
        b (float): The second float input.

    Returns:
        float: The result of the callback applied to (a + b).
    """
    # Load the DLL
    lib_path = "./lib_calculus.so"  # Ensure the correct library file name and path
    calculus = ctypes.CDLL(lib_path)
    # Define ctypes function signature if not already defined
    callback_function = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    calculus.invoke_with_floats.argtypes = [callback_function, ctypes.c_double, ctypes.c_double]
    calculus.invoke_with_floats.restype = ctypes.c_double

    # Wrap the Python callback with ctypes
    wrapped_callback = callback_function(callback)
    return calculus.invoke_with_floats(wrapped_callback, a, b)

def secant_root(callback, x0, x1, tol, max_iter):
    """
    Finds the root of a function using the secant method.

    Parameters:
        callback (function): A Python function representing the equation f(x).
        x0 (float): The first initial guess for the root.
        x1 (float): The second initial guess for the root.
        tol (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations allowed.

    Returns:
        float: The approximate root if convergence is achieved; otherwise, NAN.
    """
    # Load the shared library
    lib_path = "./lib_calculus.so"  # Ensure the correct library file name and path
    calculus = ctypes.CDLL(lib_path)

    # Define the ctypes callback function type
    callback_function = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    # Define the argument and return types for the secant_root function
    calculus.secant_root.argtypes = [
        callback_function,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
    ]
    calculus.secant_root.restype = ctypes.c_double

    # Wrap the Python callback function
    wrapped_callback = callback_function(callback)

    # Invoke the C++ function and return the result
    return calculus.secant_root(wrapped_callback, x0, x1, tol, max_iter)



def calculate_integrals():
    """
    Calculate integrals of the three given functions using all available algorithms.
    Print the results for each function and algorithm.
    """
    print("Calculating integrals for all functions using all algorithms...\n")

    # List of functions and their integration intervals
    functions = [
        (func1, "exp(-1/x)", 0.01, 10),
        (func2, "cos(1/x)", 0.01, 3 * np.pi),
        (func3, "x³ + 1", -1, 1)
    ]

    # Algorithms to use
    algorithms = {
        "Simpson's Rule": lambda f, a, b: wrapper_simpson(f, a, b, 1000),
        "Trapezoidal Rule": lambda f, a, b: trapezoid(f, a, b, 1000),
        "Adaptive Trapezoidal Rule": lambda f, a, b: adaptive_trap_py(
            f, a, b, tol=1e-6, remaining_depth=10
        ),
    }

    # Iterate over each function and apply all algorithms
    for func, name, a, b in functions:
        print(f"Function: {name} on [{a}, {b}]")
        for algo_name, algo in algorithms.items():
            try:
                result = algo(func, a, b)
                print(f"{algo_name}: {result:.6f}")
            except ZeroDivisionError as e:
                print(f"{algo_name}: Division by zero error - {e}")
            except ValueError as e:
                print(f"{algo_name}: Invalid value error - {e}")
            except OverflowError as e:
                print(f"{algo_name}: Overflow error - {e}")
        print("\n")


if __name__ == "__main__":
    calculate_integrals()
