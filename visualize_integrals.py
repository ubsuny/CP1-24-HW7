import numpy as np
import matplotlib.pyplot as plt
import calculus as calc

def plot_integral(func, func_name, a, b, method, method_name, n=1000):
    """
    Generate a plot showing the function and the shaded area of the integral.

    Parameters:
    - func: Function to integrate.
    - func_name: Name of the function for labeling the plot.
    - a: Lower boundary of integration.
    - b: Upper boundary of integration.
    - method: Integration method (e.g., calc.wrapper_simpson, calc.trapezoid).
    - method_name: Name of the integration method for labeling the plot.
    - n: Number of steps (default is 1000).
    """
    # Create sanitized file name
    safe_func_name = func_name.replace("(", "").replace(")", "").replace("/", "-").replace(" ", "_")
    safe_method_name = method_name.replace(" ", "_")

    x = np.linspace(a, b, 1000)
    y = func(x)
    
    # Compute integral
    try:
        result = method(func, a, b, n) if 'adaptive' not in method_name.lower() else method(func, a, b, tol=1e-6)
        steps = n if 'adaptive' not in method_name.lower() else "adaptive"
    except Exception as e:
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

# Example usage:
if __name__ == "__main__":
    functions = [
        (calc.func1, "exp(-1/x)", 0.01, 10),
        (calc.func2, "cos(1/x)", 0.01, 3 * np.pi),
        (calc.func3, "x^3+1", -1, 1),
    ]

    methods = [
        (calc.wrapper_simpson, "Simpson's Rule"),
        (calc.trapezoid, "Pure Python Trapezoid"),
        (calc.trapezoid_python, "Python Trapezoid"),
        (calc.trapezoid_numpy, "NumPy Trapezoid"),
        (calc.trapezoid_scipy, "SciPy Trapezoid"),
        (calc.adaptive_trap_py, "Adaptive Trapezoid"),
    ]

    # Generate plots for each function-method combination
    for func, func_name, a, b in functions:
        for method, method_name in methods:
            plot_integral(func, func_name, a, b, method, method_name)
