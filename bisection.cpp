#include <iostream>
#include <cmath>
#include <stdexcept>
#include <functional>

extern "C" {

/**
 * Bisection method to find the root of a function within a given interval.
 * 
 * @param func - The function whose root is being calculated.
 * @param a - The start of the interval.
 * @param b - The end of the interval.
 * @param tol - The tolerance level for stopping the iteration.
 * @param max_iter - The maximum number of iterations.
 * 
 * @return The root of the function within the given interval.
 */
double bisection(double (*func)(double), double a, double b, double tol, int max_iter) {
    if (func(a) * func(b) >= 0) {
        throw std::invalid_argument("Function values at the endpoints must have opposite signs.");
    }

    double root = 0.0;
    int iteration = 0;

    while ((b - a) / 2 > tol) {
        if (iteration >= max_iter) {
            // throw std::runtime_error("Maximum number of iterations reached.");
            // Instead of throwing an exception, return NaN
            return std::numeric_limits<double>::quiet_NaN();
        }

        root = (a + b) / 2;
        double value_at_root = func(root);

        if (value_at_root == 0.0) {
            // Exact root found
            break;
        }

        if (func(a) * value_at_root < 0) {
            b = root;
        } else {
            a = root;
        }

        iteration++;
    }

    return root;
}

/**
 * Wrapper function for passing from ctypes (no function pointer for func)
 */
double bisection_ctypes(double (*func)(double), double a, double b, double tol, int max_iterations) {
    return bisection(func, a, b, tol, max_iterations);
}

}
