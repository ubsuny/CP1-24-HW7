import numpy as np
import matplotlib.pyplot as plt
import calculus as calc

def plot_integral(func, func_name, a, b, method, n=1000):
    """
    Generate a plot showing the function and the shaded area of the integral.

    Parameters:
    - func: Function to integrate.
    - func_name: Name of the function for labeling the plot.
    - a: Lower boundary of integration.
    - b: Upper boundary of integration.
    - method: Integration method (e.g., calc.simpson).
    - n: Number of steps (default is 1000).
    """
    # Create sanitized file name
    safe_func_name = func_name.replace("(", "").replace(")", "").replace("/", "-").replace(" ", "_")

    x = np.linspace(a, b, 1000)
    y = func(x)
    result, steps = method(func, a, b, n)

    # Plot the function
    plt.plot(x, y, label=f"${func_name}(x)$")
    plt.fill_between(x, y, alpha=0.3, label=f"Area = {result:.6f}, Steps = {steps}")
    plt.title(f"Integration of ${func_name}(x)$ using Simpson's Rule")
    plt.xlabel("x")
    plt.ylabel(f"${func_name}(x)$")
    plt.legend()

    # Save the figure with a sanitized file name
    plt.savefig(f"{safe_func_name}.png")
    plt.close()

# Example usage:
if __name__ == "__main__":
    plot_integral(calc.func1, "exp(-1/x)", 0.01, 10, calc.simpson)
    plot_integral(calc.func2, "cos(1/x)", 0.01, 3 * np.pi, calc.simpson)
    plot_integral(calc.func3, "x^3+1", -1, 1, calc.simpson)
