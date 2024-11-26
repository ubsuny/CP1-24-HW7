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
#include <iostream>
#include <cmath>    // For NAN (Not-a-Number) representation
#include <stdbool.h> // For compatibility with C-style bool

extern "C" {
   double trapezoidal_rule(double (*func)(double), double a, double b, int n) {
      /**
       * this function takes in a function, and upper and lower bound, and
       * the number of points in between those bounds for integration and
       * carries out trapezoidal summation to integrate.
       */
         double h = (b - a) / n;  // Step size
         double sum = 0.5 * (func(a) + func(b));  // Start with the first and last terms

         // Sum the middle terms
         for (int i = 1; i < n; i++) {
             double x = a + i * h;  // Calculate the current x value
             sum += func(x);
         }

         // Multiply by step size
         sum *= h;

         return sum;
     }

     double second_derivative(double (*func)(double), double x, double h) {
       /**
        * this function takes in a function, x values
        * and a value h defining dx and carries out the
        * second derivative of the function at x
        */
    
         return (func(x + h) - 2 * func(x) + func(x - h)) / (h * h);
     }

     float adapt_c(double (*func)(double), double a, double b, int n, int sens){
       /**
        * This function uses an adaptive algorithm to carry out 
        * trapezoidal integration. It does this by taking the second
        * derivative at n points inbetween the upper and lower bounds
        * and using that second derivative to expand the summation
        * such that more trapezoids are used to integrate along 
        * more curved parts of the function. With this, the funciton
        * takes in a function, upper and lower bounds, the number 
        * of points to take derivatives at and an integer, called sens
        * which defines how sensitive the algorithm is to curvature. 
        * Higher sens means more sensitivity and greater accuracy.
        */
         double h = (b - a) / n; 
         double sum=0;
         for (int i = 1; i < n; i++) {
             double x = a + i * h;  
             int der=(second_derivative(func, x,h)+1)*sens;
             der =abs(der);
             if (x!=b){
               sum+=trapezoidal_rule(func, x, x+h, der);
             }
         }
         return sum;
     }
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
