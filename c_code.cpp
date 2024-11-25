#include <iostream>
#include <list>
#include <cmath>
using namespace std;

extern "C"{
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
}




