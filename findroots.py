"""
Root-Finding Module for Mathematical Functions
This module implements methods for root-finding from calculus.py, including the secant and
bisection methods, using both SciPy and pure Python implementations. It also provides
utilities for function visualization and accuracy analysis of root-finding methods.
Functions:
    - func_1_safe: Defines the function y(x) = 1/sin(x) with singularities handled.
    - func_2: Defines the function y(x) = tanh(x).
    - func_3: Defines the function y(x) = sin(x).
    - calculate_accuracy: Computes the number of correct decimal digits in a root approximation.
    - apply_methods: Applies root-finding methods to a function over a specified interval.
    - plot_function_with_roots: Plots the function over an interval and marks the found roots.
    - find_roots: Main function to test root-finding on predefined functions and intervals.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from calculus import secant_wrapper, bisection_wrapper, bisection_pure_python, secant_pure_python

# Define the functions
# --------------------------------------------------------------------------------
def func_1_safe(x):
    """
    y(x) = 1/sin(x) with singularities handled.
    """
    epsilon = 1e-6  # Threshold to avoid singularity
    sin_x = math.sin(x)
    if abs(sin_x) < epsilon:
        raise ValueError(f"Singularity detected near x = {x}")
    return 1 / sin_x

def func_2(x):
    """
    y(x) = tanh(x).
    """
    return np.tanh(x)

def func_3(x):
    """
    y(x) = sin(x).
    """
    return math.sin(x)

# Calculate the accuracy of roots
# --------------------------------------------------------------------------------
def calculate_accuracy(approx, true_root):
    """
    Calculate the number of correct decimal digits in the approximation of the root.
    """
    if approx is None or true_root is None:
        return 0
    if approx == 0 and true_root == 0:
        return float('inf')  # Perfect match for zero roots
    try:
        return -int(math.log10(abs(approx - true_root)))
    except ValueError:
        return 0

# Function to apply the root finding methods
# --------------------------------------------------------------------------------
def apply_methods(func, interval, description, true_root=None, filename=None):
    """
    Apply root-finding methods to the function over a specified interval.
    """
    print(f"\nFinding roots for {description}")
    a, b = interval
    roots = []  # To store all found roots

    # Define the root-finding methods
    methods = [
        {"name": "Secant (scipy)", "func": secant_wrapper},
        {"name": "Bisection (scipy)", "func": bisection_wrapper},
        {"name": "Bisection (pure Python)", "func": bisection_pure_python},
        {"name": "Secant (pure Python)", "func": secant_pure_python},
    ]

    # Iterate over the methods and apply them
    for method in methods:
        try:
            result = method["func"](func, a, b)
            root = result if isinstance(result, (float, int)) else result.get('root')
            if root is not None:
                roots.append(root)
            accuracy = calculate_accuracy(root, true_root)
            if isinstance(result, dict):  # Check if method returns a dict
                print(
                    f"{method['name']} root: {root} | Converged: {result.get('converged', 'N/A')} |"
                    f"Accuracy: {accuracy} digits"
                )
            else:
                print(f"{method['name']} root: {root} | Accuracy: {accuracy} digits")
        except ValueError as e:
            print(f"{method['name']} error: {e}")

    # Remove None values and duplicates from the roots list
    roots = list(set(filter(lambda x: x is not None, roots)))

    # Optionally plot the function with identified roots
    if filename:
        plot_function_with_roots(func, interval, roots, filename, description)

    return roots

# Plot functions, mark roots, and save as png file
# --------------------------------------------------------------------------------
def plot_function_with_roots(func, interval, roots, filename, description):
    """
    Plot the function over the given interval and mark the roots found.
    """
    x = np.linspace(interval[0], interval[1], 1000)
    y = np.array([func(val) for val in x])

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f"{description}", color="blue")
    plt.axhline(0, color='red', linestyle='--', label="y=0")

    # Mark roots
    if roots:
        for root in roots:
            plt.plot(root, func(root), 'o', label=f"Root at x={root:.4f}", markersize=8)
    else:
        print("No valid roots found to mark on the plot.")

    # Generate the plot
    plt.title(f"Function: {description}")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")

# Main function
# --------------------------------------------------------------------------------
def find_roots():
    """
    Main function to find roots for multiple functions using various methods and compare accuracies.
    """
    # Root finding for y(x) = 1/sin(x) with singularities handled
    apply_methods(func_1_safe, (0.5, 1.5), "y(x) = 1/sin(x) (singularities handled)",
                  filename="root_1_sinx.png")

    # Root finding for y(x) = tanh(x)
    apply_methods(func_2, (-1, 1), "y(x) = tanh(x)", true_root=0.0, filename="root_tanhx.png")

    # Root finding for y(x) = sin(x)
    apply_methods(func_3, (3, 4), "y(x) = sin(x)", true_root=math.pi, filename="root_sinx.png")

if __name__ == "__main__":
    find_roots()
