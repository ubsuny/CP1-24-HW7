/**
 * @file tangent_root_finder.cpp
 * @brief C++ implementation of root-finding with the tangent method
 */

#include <cmath>
#include <stdbool.h>

extern "C" {
    // Function pointer types for function and derivative
    typedef double (*func)(double);
    typedef double (*deriv)(double);
    typedef struct {
        bool converged;
        double root;
    } Result;
    /**
     * @brief Finds the root of a function using the tangent method.
     * @param f The function to find its root.
     * @param fprime Its derivative.
     * @param x0 The initial guess.
     * @param tol The tolerance to the accepted values
     * @param max_iter The max. number of iterations before terminating the algorithm.
     * @return a struct of two variables, one to inform about convergence and the other is the root.
     */
    Result cpp_root_tangent(func f, deriv fprime, double x0, double tol = 1e-6, int max_iter = 100) {
        Result result;
        result.converged = false;
        result.root = x0;
        double x = x0;
        for (int i = 0; i < max_iter; i++) {
            double fx = f(x);
            double fpx = fprime(x);
            if (fabs(fpx) < 1e-10) { // Prevents division by zero
                return result;
            }
            double x_next = x - fx / fpx;
            if (fabs(x_next - x) < tol) {
                result.converged = true;
                result.root = x_next;
                return result;
            }
            x = x_next;
        }
        return result;
    }
}
