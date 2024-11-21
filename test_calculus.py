"""
test_calculus.py
"""

import unittest
import math
import calculus as calc

def test_dummy():
    """ Unit test for dummy function
    just the same test function that Dr Thomay made"""
    assert calc.dummy() == 0

class TestBisectionMethods(unittest.TestCase):
    """
    Unit tests for bisection root-finding methods implemented in three ways:
    - SciPy wrapper implementation
    - Pure Python implementation
    - C++ ctypes implementation

    Tests are performed on multiple mathematical functions, including:
    - y(x) = 1 / sin(x), which has singularities where sin(x) = 0
    - y(x) = tanh(x), a smooth function with a root at x = 0

    Additionally, tests include edge cases for:
    - Invalid intervals where func(a) and func(b) do not have opposite signs
    - Proper handling of singularities and undefined values
    """

    def test_wrapper_tanh(self):
        """Test SciPy wrapper implementation with tanh(x)."""
        def func(x):
            return math.tanh(x)
        root = calc.bisection_wrapper(func, -1, 1, tol=1e-6)
        self.assertAlmostEqual(root, 0.0, places=6)

    def test_wrapper_1_over_sin(self):
        """Test SciPy wrapper implementation with 1/sin(x)."""
        def func(x):
            if math.sin(x) == 0:
                raise ValueError("Function undefined where sin(x) = 0.")
            return 1 / math.sin(x)
        root = calc.bisection_wrapper(func, 3, 4, tol=1e-6)  # Root near pi
        self.assertAlmostEqual(root, math.pi, places=6)

    def test_pure_python_tanh(self):
        """Test pure Python implementation with tanh(x)."""
        def func(x):
            return math.tanh(x)
        root = calc.bisection_pure_python(func, -1, 1, tol=1e-6)
        self.assertAlmostEqual(root, 0.0, places=6)

    def test_pure_python_1_over_sin(self):
        """Test pure Python implementation with 1/sin(x)."""
        def func(x):
            if math.sin(x) == 0:
                raise ValueError("Function undefined where sin(x) = 0.")
            return 1 / math.sin(x)
        root = calc.bisection_pure_python(func, 3, 4, tol=1e-6)
        self.assertAlmostEqual(root, math.pi, places=6)

    def test_ctypes_tanh(self):
        """Test C++ ctypes implementation with tanh(x)."""
        def func(x):
            return math.tanh(x)
        root = calc.bisection_ctypes(func, -1, 1, tol=1e-6)
        self.assertAlmostEqual(root, 0.0, places=6)

    def test_ctypes_1_over_sin(self):
        """Test C++ ctypes implementation with 1/sin(x)."""
        def func(x):
            if math.sin(x) == 0:
                raise ValueError("Function undefined where sin(x) = 0.")
            return 1 / math.sin(x)
        root = calc.bisection_ctypes(func, 3, 4, tol=1e-6)
        self.assertAlmostEqual(root, math.pi, places=6)

    def test_invalid_interval(self):
        """Test for invalid intervals where func(a) and func(b) do not have opposite signs."""
        def func(x):
            return math.tanh(x)
        with self.assertRaises(ValueError):
            calc.bisection_wrapper(func, -1, -0.5, tol=1e-6)
        with self.assertRaises(ValueError):
            calc.bisection_pure_python(func, -1, -0.5, tol=1e-6)
        with self.assertRaises(ValueError):
            calc.bisection_ctypes(func, -1, -0.5, tol=1e-6)

    def test_singularities(self):
        """Test handling of singularities for 1/sin(x)."""
        def func(x):
            if math.sin(x) == 0:
                raise ValueError("Function undefined where sin(x) = 0.")
            return 1 / math.sin(x)
        with self.assertRaises(ValueError):
            calc.bisection_wrapper(func, math.pi - 0.1, math.pi + 0.1, tol=1e-6)
        with self.assertRaises(ValueError):
            calc.bisection_pure_python(func, math.pi - 0.1, math.pi + 0.1, tol=1e-6)
        with self.assertRaises(ValueError):
            calc.bisection_ctypes(func, math.pi - 0.1, math.pi + 0.1, tol=1e-6)

# if __name__ == "__main__":
#     unittest.main()
