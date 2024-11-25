"""
This script visualizes the integration of mathematical functions
using various numerical methods (e.g., Simpson's rule, Trapezoidal rule).
"""

import numpy as np
import matplotlib.pyplot as plt
import calculus as calc


def plot_integral(integration_details, n=1000):
    """
    Generate a plot showing the function and the shaded area of the integral.

    Parameters:
    - integration_details: Dictionary containing all function and method details.
        Expected keys: "func", "name", "a", "b", "method_func", "method_name".
    - n: Number of steps (default is 1000).
    """
    function = integration_details["func"]
    func_name = integration_details["name"]
    a, b = integration_details["a"], integration_details["b"]
    method_func = integration_details["method_func"]
    method_name = integration_details["method_name"]

    # Create sanitized file names
    safe_func_name = func_name.replace("(", "").replace(")", "").replace("/", "-").replace(" ", "_")
    safe_method_name = method_name.replace(" ", "_")

    x = np.linspace(a, b, 1000)
    y = function(x)

    # Compute integral
    try:
        result = (
            method_func(function, a, b, n)
            if "adaptive" not in method_name.lower()
            else method_func(function, a, b, tol=1e-6)
        )
        steps = n if "adaptive" not in method_name.lower() else "adaptive"
    except (ValueError, TypeError) as e:
        print(f"Error integrating {func_name} using {method_name}: {e}")
        return

    # Plot the function
    plt.plot(x, y, label=f"${func_name}(x)$")
    plt.fill_between(x, y, alpha=0.3, label=f"Area = {result:.6f}, Steps = {steps}")
    plt.title(f"Integration of ${func_name}(x)$ using {method_name}")
    plt.xlabel("x")
    plt.ylabel(f"${func_name}(x)$")
    plt.legend()

    # Save the figure with a sanitized file name
    plt.savefig(f"{safe_func_name}_{safe_method_name}.png")
    plt.close()


if __name__ == "__main__":
    function_list = [
        {"func": calc.func1, "name": "exp(-1/x)", "a": 0.01, "b": 10},
        {"func": calc.func2, "name": "cos(1/x)", "a": 0.01, "b": 3 * np.pi},
        {"func": calc.func3, "name": "x^3+1", "a": -1, "b": 1},
    ]

    method_list = [
        {"method_func": calc.wrapper_simpson, "method_name": "Simpson's Rule"},
        {"method_func": calc.trapezoid, "method_name": "Pure Python Trapezoid"},
        {"method_func": calc.trapezoid_python, "method_name": "Python Trapezoid"},
        {"method_func": calc.trapezoid_numpy, "method_name": "NumPy Trapezoid"},
        {"method_func": calc.trapezoid_scipy, "method_name": "SciPy Trapezoid"},
        {"method_func": calc.adaptive_trap_py, "method_name": "Adaptive Trapezoid"},
    ]

    # Generate plots for each function-method combination
    for func_details in function_list:
        for method_details in method_list:
            details = {**func_details, **method_details}
            plot_integral(details)
