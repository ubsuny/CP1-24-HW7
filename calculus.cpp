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
 * g++ -shared -o calculus.dll -fPIC calculus.cpp
 */

#include <cmath>    // For NAN (Not-a-Number) representation
#include <stdbool.h> // For compatibility with C-style bool

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
}
