/**
 * @file calculus.cpp
 * @brief C++ implementation of a simple module for performing mathematical operations,
 * intended for use with a ctypes wrapper in Python.
 *
 * This module provides two functions:
 * - verify_arguments: Validates input to ensure it meets specific criteria.
 * - calculate_square: Computes the square of a number, with input validation.
 *
 * Usage:
 * Compile this file into a shared library for Python:
 * g++ -shared -o lib_calculus.so -fPIC lib_calculus.cpp
 */

#include <cmath>         // For NAN (Not-a-Number) representation
#include <stdbool.h>     // For compatibility with C-style bool
#include <iostream>      // IO handling

typedef double (*CallbackFunction)(double);

extern "C" {
    /**
     * @brief Verifies if the provided argument is non-negative.
     *
     * This function checks whether the given input meets a basic criterion:
     * it must be non-negative. Used primarily to validate inputs for other
     * mathematical functions in the module.
     *
     * @param x The input value to verify.
     * @return true if the input is non-negative, false otherwise.
     */
    bool verify_arguments(double x) {
        return x >= 0;
    }

    /**
     * @brief Calculates the square of a given number.
     *
     * This function computes the square of the provided input if the input
     * is valid (non-negative). If the input is invalid (negative), the
     * function returns NAN (Not-a-Number) as a special error indicator.
     *
     * @param x The input value to square.
     * @return The square of the input if valid; otherwise, NAN.
     */
    double calculate_square(double x) {
        if (!verify_arguments(x)) {
            return NAN; // Return NAN for invalid input
        }
        return x * x; // Return the square of the input
    }

    /**
     * Applies a Python math function (via callback) to two floats.
     * The result is the evaluation of `callback(a + b)`.
     */
    double invoke_with_floats(CallbackFunction callback, double a, double b) {
        if (callback == nullptr) {
            return NAN; // Handle invalid callback
        }
        double sum = a + b;
        return callback(sum);
    }

    /**
     * @brief Implements the secant method for root-finding.
     *
     * The secant method is an iterative numerical technique to approximate the root
     * of a nonlinear function using two initial guesses and the slope between them.
     *
     * @param callback The function whose root is to be found.
     * @param x0 The first initial guess for the root.
     * @param x1 The second initial guess for the root.
     * @param tol The tolerance for convergence and division checks.
     * @param max_iter The maximum number of iterations allowed. If convergence is not
     *                 achieved within this limit, the method returns NaN.
     * @return The approximate root of the function if convergence is achieved; otherwise, NAN.
     */
    double secant_root(CallbackFunction callback, double x0, double x1, double tol, int max_iter) {
        // Validate the callback function
        if (callback == nullptr) {
            std::cerr << "Invalid callback function." << std::endl;
            return NAN;
        }

        // Evaluate the function at initial guesses
        double f_x0 = callback(x0);
        double f_x1 = callback(x1);

        // Check if initial guesses are valid
        if (std::isnan(f_x0) || std::isnan(f_x1)) {
            std::cerr << "Function returned NaN for initial guesses." << std::endl;
            return NAN;
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            // Calculate the difference in function values (denominator)
            double denominator = f_x1 - f_x0;

            // Check for division by a near-zero denominator
            if (fabs(denominator) < tol) {
                std::cerr << "Division by zero detected: |f_x1 - f_x0| < tol." << std::endl;
                return NAN; // Explicitly return NaN to indicate the issue
            }

            // Apply the secant method formula
            double x2 = x1 - f_x1 * (x1 - x0) / denominator;

            // Check for convergence
            if (fabs(x2 - x1) < tol) {
                return fabs(x2) < tol ? 0.0 : x2;  // Explicitly return 0.0 if near zero
            }

            // Update variables for the next iteration
            x0 = x1;
            f_x0 = f_x1;
            x1 = x2;
            f_x1 = callback(x1);

            // Check for invalid function output (e.g., NaN)
            if (std::isnan(f_x1)) {
                std::cerr << "Function returned NaN at x1=" << x1 << std::endl;
                return NAN;
            }
        }

        // If maximum iterations are reached without convergence
        std::cerr << "Secant method did not converge within the maximum number of iterations." << std::endl;
        return NAN; // Return NaN to indicate failure
    }

}

typedef double (*CallbackFunction)(double);

extern "C" {
    /**
     * @brief Implements the trapezoidal rule for numerical integration.
     *
     * This function approximates the integral of a given callback function
     * over a specified interval [a, b] using the trapezoidal rule.
     *
     * @param callback The function to integrate.
     * @param a The lower limit of integration.
     * @param b The upper limit of integration.
     * @param n The number of subdivisions (must be positive).
     * @return The approximate value of the integral, or NAN if an error occurs.
     */
    double trapezoidal_rule(CallbackFunction callback, double a, double b, int n) {
        if (n <= 0) {
            std::cerr << "Error: Number of subdivisions must be positive. Received n = " << n << std::endl;
            return NAN;
        }

        if (callback == nullptr) {
            std::cerr << "Error: Callback function is null." << std::endl;
            return NAN;
        }

        double h = (b - a) / n;  // Step size
        double integral = 0.0;

        // Pre-check for divide by zero in endpoints
        if (a == 0 || b == 0) {
            std::cerr << "Error: Divide by zero detected at endpoint(s) a or b." << std::endl;
            return NAN;
        }

        // Check and evaluate endpoints
        try {
            double fa = callback(a);
            double fb = callback(b);

            // Check for NAN or infinity
            if (std::isnan(fa) || std::isnan(fb)) {
                std::cerr << "Error: Endpoint evaluations are invalid (NAN)." << std::endl;
                return NAN;
            }
            if (std::isinf(fa) || std::isinf(fb)) {
                std::cerr << "Error: Endpoint evaluations are invalid (infinity)." << std::endl;
                return NAN;
            }
            integral = 0.5 * (fa + fb);  // Contribution from endpoints
        } catch (const std::exception& e) {
            std::cerr << "Error: Exception occurred during callback evaluation at endpoints: " << e.what() << std::endl;
            return NAN;
        } catch (...) {
            std::cerr << "Error: Unknown exception occurred during callback evaluation at endpoints." << std::endl;
            return NAN;
        }

        // Safely evaluate interior points
        for (int i = 1; i < n; ++i) {
            double x_i = a + i * h;

            // Pre-check for divide by zero
            if (x_i == 0) {
                std::cerr << "Error: Divide by zero detected at x = " << x_i << std::endl;
                return NAN;
            }

            try {
                double f_xi = callback(x_i);

                // Check for NAN or infinity
                if (std::isnan(f_xi)) {
                    std::cerr << "Error: Invalid evaluation at interior point x = " << x_i << std::endl;
                    return NAN;
                }
                if (std::isinf(f_xi)) {
                    std::cerr << "Error: Function approaches infinity at x = " << x_i << std::endl;
                    return NAN;
                }
                integral += f_xi;  // Sum contributions
            } catch (const std::exception& e) {
                std::cerr << "Error: Exception occurred during callback evaluation at x = " << x_i << ": " << e.what() << std::endl;
                return NAN;
            } catch (...) {
                std::cerr << "Error: Unknown exception occurred during callback evaluation at x = " << x_i << std::endl;
                return NAN;
            }
        }

        // Apply step size and return the final integral
        return h * integral;
    }
}
